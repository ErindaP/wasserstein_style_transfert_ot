import torch

def sqrtm(M):
    """Compute the square root of a positive semidefinite matrix"""
    # Small regularization closely following the original code's logic
    # but arguably we should ensure symmetry before SVD if needed.
    # For now, sticking to previous implementation logic.
    _, s, v = M.svd()
    # truncate small components
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
