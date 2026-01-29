import torch
import numpy as np
import ot
import ot.gmm
import sklearn.mixture


def ot_emd_transfer(alpha, cf, styles, style_weights=None, max_samples=10000):
    """
    Exact OT using Linear Programming (ot.emd).
    Uses subsampling + Nearest Neighbor extension for large images.
    """
    if cf.dim() == 3: cf = cf.unsqueeze(0)
    B, C, H, W = cf.shape
    N = B * H * W
    
    cf_view = cf.view(B, C, -1)
    cf_flat = cf_view.permute(0, 2, 1).reshape(-1, C).double() # (N, C)
    
    if isinstance(styles, torch.Tensor): styles = [styles]
    if style_weights is None: style_weights = [1.0/len(styles)] * len(styles)
    
    y_list = []
    for i, sf in enumerate(styles):
        if sf.dim() == 3: sf = sf.unsqueeze(0)
        sf_flat = sf.view(sf.size(0), C, -1).permute(0, 2, 1).reshape(-1, C).double()
        
        # Subsample style right away if needed per style (heuristic)
        # But we'll handle global subsampling below.
        y_list.append(sf_flat)
        

    Y_list = []
    target_samples = min(N, max_samples)
    
    for i, sf_flat in enumerate(y_list):
        n_s = sf_flat.size(0)
        n_needed = int(style_weights[i] * target_samples)
        if n_needed > 0:
            if n_s > n_needed:
                idx = torch.randperm(n_s, device=cf.device)[:n_needed]
                Y_list.append(sf_flat[idx])
            else:
                Y_list.append(sf_flat) 
    
    if not Y_list: return cf
    Y = torch.cat(Y_list, dim=0) # (M, C)
    M = Y.size(0)
    
    # 3. Subsample Content
    if N > max_samples:
        indices = torch.randperm(N, device=cf.device)[:max_samples]
        X_sampled = cf_flat[indices] # (K, C)
    else:
        indices = torch.arange(N, device=cf.device)
        X_sampled = cf_flat
        
    K_samp = X_sampled.size(0)
    
    #  Compute OT 
    # Move to CPU for POT
    Xs_np = X_sampled.cpu().numpy()
    Ys_np = Y.cpu().numpy()
    
    # Uniform weights
    a = np.ones((K_samp,)) / K_samp
    b = np.ones((M,)) / M
    

    # Cost Matrix
    M_cost = ot.dist(Xs_np, Ys_np, metric='euclidean') # (K, M)
    
    # Solve
    # P is (K, M) sparse matrix
    P = ot.emd(a, b, M_cost)
    

    
    Targets_np = (P @ Ys_np) * K_samp
    Targets = torch.from_numpy(Targets_np).to(cf.device)
    
    Displacement = Targets - X_sampled # (K, C)
    
    # Apply to Full Content
    # If N == K_samp, we just apply directly.
    # If N > K_samp, we perform Nearest Neighbor interpolation of the Displacement field.
    
    if N == K_samp:

        # Actually easier: calculate result tensor
        res_flat = cf_flat.clone()
        res_flat[indices] = Targets
    else:
        # NN Extension
        # For every pixel in N, find closest pixel in X_sampled.
        # Calculate distances
        
        # We need to assign a displacement to every pixel `cf_flat[i]`.
        # Find nearest `X_sampled[j]`.
        # displacement[i] = Displacement[j]
        
        # X_sampled is (K, C). cf_flat is (N, C).
        
        NN_BATCH = 4096
        res_flat = torch.zeros_like(cf_flat)
        
        X_sampled_float = X_sampled.float()
        
        for i in range(0, N, NN_BATCH):
            end = min(i + NN_BATCH, N)
            batch_x = cf_flat[i:end].float()
            
            # Distance to samples
            dists = torch.cdist(batch_x, X_sampled_float) # (batch, K)
            
            # Align
            nearest_idx = torch.argmin(dists, dim=1) # (batch,)
            
            batch_disp = Displacement[nearest_idx]
            
            res_flat[i:end] = batch_x.double() + batch_disp
            
    # Reshape and Interpolate
    res_flat = res_flat.float()
    pushed = res_flat.view(B, H, W, C).permute(0, 3, 1, 2)
    
    res = (1-alpha) * cf.float() + alpha * pushed
    return res

def gmm_transfer(alpha, cf, styles, style_weights=None, K=5):
    """
    Apply GMM-based Wasserstein Style Transfer.
    
    Args:
        alpha: float
        cf: content features tensor (C, H, W)
        styles: list of style features tensors
        style_weights: list of weights
        K: number of GMM components
    """
    # 0. Setup
    if isinstance(styles, torch.Tensor):
        styles = [styles]
        style_weights = [1.0]

    if style_weights is None:
        style_weights = [1.0/len(styles)] * len(styles)
        
    cf = cf.double()
    channels = cf.size(0)
    cfv = cf.view(channels, -1) # C x N
    
    # Subsampling for GMM fitting to improve speed
    MAX_SAMPLES = 25000
    
    def fit_gmm(features, n_components):
        # features: (N, C) numpy
        n_samples = features.shape[0]
        if n_samples > MAX_SAMPLES:
            rng = np.random.RandomState(42) # fixed seed
            indices = rng.choice(n_samples, MAX_SAMPLES, replace=False)
            features_subset = features[indices]
        else:
            features_subset = features
            
        gmm = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type='full', reg_covar=1e-5)
        gmm.fit(features_subset)
        return gmm

    def ensure_pd(cov, eps=1e-5):
        # Ensure covariance is positive definite
        # cov: (K, C, C) or (C,C)
        if cov.ndim == 2:
            cov = (cov + cov.T) / 2
            cov = cov + np.eye(cov.shape[0]) * eps
        elif cov.ndim == 3:
            for k in range(cov.shape[0]):
                cov[k] = (cov[k] + cov[k].T) / 2
                cov[k] = cov[k] + np.eye(cov.shape[1]) * eps
        return cov

    # 1. Fit GMM to Content
    x_np_full = cfv.T.cpu().numpy()
    c_gmm = fit_gmm(x_np_full, K)
    
    # 2. Fit/Construct GMM for Style (Barycenter)
    s_weights_list = []
    s_means_list = []
    s_covs_list = []
    
    for i, sf in enumerate(styles):
        sf = sf.double()
        sfv = sf.view(channels, -1)
        s_np_full = sfv.T.cpu().numpy()
        
        s_gmm = fit_gmm(s_np_full, K)
        
        s_weights_list.append(s_gmm.weights_ * style_weights[i])
        s_means_list.append(s_gmm.means_)
        s_covs_list.append(s_gmm.covariances_)
        
    # Combine parameters to form the Target GMM
    ws = np.concatenate(s_weights_list)
    ms = np.concatenate(s_means_list)
    Cs = np.concatenate(s_covs_list)
    
    ws = ws / ws.sum()
    
    wc = c_gmm.weights_
    mc = c_gmm.means_
    Cc = c_gmm.covariances_
    
    Cs = ensure_pd(Cs)
    Cc = ensure_pd(Cc)
    
    # 3. Compute OT Plan between GMM components
    # Calculate Cost Matrix (Wasserstein distance between Gaussians)
    
    def gaussian_w2_dist(m1, C1, m2, C2):
        # m: (D,), C: (D, D)
        # W2^2 = ||m1-m2||^2 + Tr(C1 + C2 - 2(C1^1/2 C2 C1^1/2)^1/2)
        diff = m1 - m2
        term1 = torch.sum(diff**2)
        
        C1_sqrt = sqrtm(C1)
        term2 = torch.trace(C1 + C2 - 2 * sqrtm(C1_sqrt @ C2 @ C1_sqrt))
        return term1 + term2

    # Prepare GPU tensors for components

    
    # Create Cost Matrix M (K x K)
    M = torch.zeros((K, K), device=cf.device, dtype=torch.float64)
    
    
    t_mc = torch.from_numpy(mc).to(cf.device).double()
    t_ms = torch.from_numpy(ms).to(cf.device).double()
    t_Cc = torch.from_numpy(Cc).to(cf.device).double()
    t_Cs = torch.from_numpy(Cs).to(cf.device).double()
    
    for i in range(K):
        for j in range(K):
            M[i, j] = gaussian_w2_dist(t_mc[i], t_Cc[i], t_ms[j], t_Cs[j])
            
    # Solve OT with POT 
    # weights need to be numpy
    M_np = M.cpu().numpy()
    P = ot.emd(wc, ws, M_np) # (K, K) transport plan
    
    # 4. Apply Map (GPU)
    # T(x) = \sum_{i,j} \frac{P_{ij} p(i|x)}{w_i} T_{i->j}(x)
    # where T_{i->j}(x) = m_j + A_{ij}(x - m_i)
    # A_{ij} = C_i^{-1/2} (C_i^{1/2} C_j C_i^{1/2})^{1/2} C_i^{-1/2}
    
    # Precompute A_{ij} matrices
    A_matrices = torch.zeros((K, K, channels, channels), device=cf.device, dtype=torch.float64)
    for i in range(K):
        Ci = t_Cc[i]
        Ci_sqrt = sqrtm(Ci)
        Ci_sqrt_inv = torch.inverse(Ci_sqrt)
        
        for j in range(K):
            Cj = t_Cs[j]
            # Monge map for Gaussians
            mid = sqrtm(Ci_sqrt @ Cj @ Ci_sqrt)
            A_ij = Ci_sqrt_inv @ mid @ Ci_sqrt_inv
            A_matrices[i, j] = A_ij
            
    # Calculate membership probabilities p(i|x) for each pixel
    # log p(x|i) = -0.5 * (x-mu_i)^T inv(Sigma_i) (x-mu_i) - 0.5 log det(Sigma_i) + const
    
    # cfv is (C, N), dim 0 is channels
    x = cfv.T # (N, C)
    N_pixels = x.shape[0]
    
    log_probs = torch.zeros((N_pixels, K), device=cf.device, dtype=torch.float64)
    
    for i in range(K):
        mu = t_mc[i] # (C,)
        sigma = t_Cc[i] # (C, C)
        
        try:
            L = torch.linalg.cholesky(sigma)
            log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))
            # Mahalanobis distance
            # (x - mu) @ inv(sigma) @ (x-mu).T
            # = || L^-1 (x-mu) ||^2
            
            centered = x - mu.unsqueeze(0) # (N, C)
            # solve L y = centered^T  -> y = L^-1 centered^T
            y = torch.linalg.solve_triangular(L, centered.T, upper=False) # (C, N)
            mahalanobis = torch.sum(y**2, dim=0) # (N,)
            
            log_probs[:, i] = torch.log(torch.tensor(wc[i])) - 0.5 * (mahalanobis + log_det)
            
        except Exception as e:

            pass

    # Normalize to get posteriors p(i|x)
    # Use log-sum-exp for stability
    max_log = torch.max(log_probs, dim=1, keepdim=True)[0]
    # Handle cases where all probs are -inf (shouldn't happen with valid GMM)
    probs = torch.exp(log_probs - max_log)
    probs = probs / (torch.sum(probs, dim=1, keepdim=True) + 1e-10) # (N, K)
    
    # Construct the displacement
    # We want y = \sum_{i,j} \frac{P_{ij}}{w_i} p(i|x) (m_j + A_{ij}(x - m_i))
    #           = \sum_i p(i|x) \sum_j \frac{P_{ij}}{w_i} (m_j + A_{ij}(x - m_i))
    
    y = torch.zeros_like(x)
    
    for i in range(K):
        # Contribution of component i
        # weighting: gamma_i(x) = p(i|x)
        gamma_i = probs[:, i].unsqueeze(1) # (N, 1)
        
        # If gamma_i is effectively 0 for all pixels, skip
        if torch.sum(gamma_i) < 1e-6:
            continue
            
        centered_i = x - t_mc[i] # (N, C)
        
        # Inner sum over j
        # term_i = \sum_j (P_{ij}/w_i) (m_j + A_{ij} centered_i)
        #        = (\sum_j P'_{ij} m_j) + (\sum_j P'_{ij} A_{ij}) centered_i
        
        # P'_{ij} = P_{ij} / w_i
        w_i = wc[i]
        if w_i < 1e-10: continue
        
        P_row = torch.from_numpy(P[i]).to(cf.device).double() / w_i # (K,)
        
        # Weighted mean target: sum_j P'_{ij} m_j
        mean_target_i = torch.sum(P_row.unsqueeze(1) * t_ms, dim=0) # (C,)
        
        # Weighted Transform: sum_j P'_{ij} A_{ij}
        # A_matrices[i] is (K, C, C)
        # Broadcast P_row: (K, 1, 1)
        A_target_i = torch.sum(P_row.view(K, 1, 1) * A_matrices[i], dim=0) # (C, C)
        
        # Map for cluster i: m'_i + A'_i(x - m_i)
        mapped_i = mean_target_i + (centered_i @ A_target_i.T)
        
        y += gamma_i * mapped_i
        
    pushed = y.T.view_as(cf).float() # (C, H, W) or (C, H, W)
    res = (1-alpha) * cf.float() + alpha * pushed
    
    if cf.dim() == 3:
        res = res.unsqueeze(0)
        
    return res.float()

def sqrtm(M):
    """Compute the square root of a positive semidefinite matrix"""
    # Small regularization closely following the original code's logic
    # but arguably we should ensure symmetry before SVD if needed.
    # For now, sticking to previous implementation logic.
    _, s, v = M.svd()
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

def get_mean_cov(features):
    """
    Compute mean and covariance of features.
    
    Args:
        features: (B, C, H, W) or (C, H, W) tensor
    Returns:
        mean: (C, 1)
        cov: (C, C)
        reshaped_features: (C, H*W) centered
    """
    if features.dim() == 4:
        features = features.squeeze(0)
    
    c_channels, c_width, c_height = features.size(0), features.size(1), features.size(2)
    features_v = features.view(c_channels, -1).double() # Work in double precision

    mean = torch.mean(features_v, 1, keepdim=True)
    features_v_centered = features_v - mean
    
    cov = torch.mm(features_v_centered, features_v_centered.t()).div((c_width * c_height) - 1)
    
    return mean, cov, features_v_centered, features.view(c_channels, -1).double()

def wasserstein_barycenter_cov(covs, weights, max_iter=20, tol=1e-6):
    """
    Compute the Wasserstein Barycenter of covariances using fixed point iteration.
    For N=2, uses the closed-form geodesic.
    
    Args:
        covs: List of (C, C) covariance matrices
        weights: List of weights summing to 1
    """
    if len(covs) == 2:
        # Use closed-form geodesic
        # Sigma_t = [(1-t)I + tT] Sigma_0 [(1-t)I + tT]
        # where T is the optimal transport map from Sigma_0 to Sigma_1
        # and t = w_1 (assuming w_0 + w_1 = 1)
        
        S0 = covs[0]
        S1 = covs[1]
        t = weights[1]
        
        S0_sqrt = sqrtm(S0)
        S0_sqrt_inv = torch.inverse(S0_sqrt)
        
        # Transport Map T_{0->1}
        # T = S0^{-1/2} (S0^{1/2} S1 S0^{1/2})^{1/2} S0^{-1/2}
        
        middle = sqrtm(S0_sqrt @ S1 @ S0_sqrt)
        T = S0_sqrt_inv @ middle @ S0_sqrt_inv
        
        # Interpolation Map T_t = (1-t)I + tT
        eye = torch.eye(S0.size(0), device=S0.device, dtype=S0.dtype)
        Tt = (1-t) * eye + t * T
        
        # Sigma_t = T_t S0 T_t
        sigma = Tt @ S0 @ Tt
        
        return sigma

    # Initial guess: simple weighted sum (Fréchet mean likely close)
    sigma = torch.zeros_like(covs[0])
    for i, C in enumerate(covs):
        sigma += weights[i] * C
        
    for k in range(max_iter):
        sigma_sqrt = sqrtm(sigma)
        
        # Fixed Point Iteration (Alvarez-Esteban et al. 2016)
        # S_{k+1} = S_k^{-1/2} [ \sum_j w_j (S_k^{1/2} \Sigma_j S_k^{1/2})^{1/2} ]^2 S_k^{-1/2}
        
        sigma_sqrt_inv = torch.inverse(sigma_sqrt)
        
        inner_sum = torch.zeros_like(sigma)
        for i, C in enumerate(covs):
             inner_sum += weights[i] * sqrtm(sigma_sqrt @ C @ sigma_sqrt)
             
        sigma_new = sigma_sqrt_inv @ (inner_sum @ inner_sum) @ sigma_sqrt_inv
        
        diff = torch.norm(sigma_new - sigma)
        sigma = sigma_new
        if diff < tol:
            break
            
    return sigma

def gaussian_transfer(alpha, cf, styles, style_weights=None):
    """
    Args:
        alpha: float
        cf: content features tensor
        styles: list of style features tensors, OR single tensor
        style_weights: list of weights (must sum to 1), optional.
    """
    if isinstance(styles, torch.Tensor):
        styles = [styles]
        style_weights = [1.0]
    
    if style_weights is None:
        style_weights = [1.0/len(styles)] * len(styles)
    
    # 1. Compute Content Statistics
    c_mean, c_cov, c_centered, cfv_full = get_mean_cov(cf)
    c_width = cf.size(1) if cf.dim()==3 else cf.size(2)
    c_height = cf.size(2) if cf.dim()==3 else cf.size(3)
    
    
    # 2. Compute Style Statistics (Barycenter)
    s_means = []
    s_covs = []
    
    for sf in styles:
        m, C, _, _ = get_mean_cov(sf)
        s_means.append(m)
        s_covs.append(C)
        
    # Barycenter Mean
    target_mean = torch.zeros_like(s_means[0])
    for i, m in enumerate(s_means):
        target_mean += style_weights[i] * m
        
    # Barycenter Covariance
    if len(styles) == 1:
        target_cov = s_covs[0]
    else:
        target_cov = wasserstein_barycenter_cov(s_covs, style_weights)
        
    
    # 3. Compute Transport Map
    # Map from Content Gaussian (c_mean, c_cov) to Target Gaussian (target_mean, target_cov)
    # T(x) = target_mean + A(x - c_mean)
    # A = c_cov^{-1/2} (c_cov^{1/2} target_cov c_cov^{1/2})^{1/2} c_cov^{-1/2}
    
    c_cov_sqrt = sqrtm(c_cov)
    c_cov_sqrt_inv = torch.inverse(c_cov_sqrt)
    
    A = c_cov_sqrt_inv @ sqrtm(c_cov_sqrt @ target_cov @ c_cov_sqrt) @ c_cov_sqrt_inv
    
    # Apply Map
    # (A @ centered_content) + target_mean
    # Shapes: A is (C,C), centered_content is (C, Pixels)
    
    pushed_features = (target_mean + torch.mm(A, c_centered))
    
    # Reshape back to image
    pushed = pushed_features.view_as(cf).float()
    
    # Interpolate with original content
    res = (1-alpha) * cf.float() + alpha * pushed
    
    if cf.dim() == 4:
         # ensure we return 4D if input was 4D
         pass
    elif cf.dim() == 3:
         res = res.unsqueeze(0)
         
    return res
