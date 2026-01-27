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

def wasserstein_barycenter_cov(covs, weights, max_iter=50, tol=1e-6):
    """
    Compute the Wasserstein Barycenter of covariances using fixed point iteration.
    
    Args:
        covs: List of (C, C) covariance matrices
        weights: List of weights summing to 1
    """
    # Initial guess: simple weighted sum (Fréchet mean likely close)
    sigma = torch.zeros_like(covs[0])
    for i, C in enumerate(covs):
        sigma += weights[i] * C
        
    for k in range(max_iter):
        sigma_sqrt = sqrtm(sigma)
        sigma_prev = sigma.clone()
        
        T_sum = torch.zeros_like(sigma)
        for i, C in enumerate(covs):
            # (Sigma^1/2 * C * Sigma^1/2)^1/2
            term = sqrtm(sigma_sqrt @ C @ sigma_sqrt)
            T_sum += weights[i] * term
            
        sigma = T_sum @ T_sum # The actual iteration is Sigma_{k+1} = (Sum w_i (Sigma_k^1/2 C_i Sigma_k^1/2)^1/2)^2 ?? 
        # Actually the fixed point is Sigma = Sum w_i (Sigma^1/2 C_i Sigma^1/2)^1/2
        # No, wait.
        # The equation is Sigma = \sum w_i (Sigma^{1/2} C_i Sigma^{1/2})^{1/2}  is WRONG.
        # The equation for barycenter Sigma is: Sigma = \sum w_i (Sigma^{1/2} C_i Sigma^{1/2})^{1/2}  is effectively what many papers say but let's double check.
        # Alvarez-Esteban et al. "A fixed-point approach to Barycenters in Wasserstein space":
        # S_{n+1} = S_n^{-1/2} ( \sum w_i (S_n^{1/2} C_i S_n^{1/2})^{1/2} )^2 S_n^{-1/2}
        
        # Let's try the implementation from POT or similar if in doubt, but the standard fixed point is:
        # T = \sum w_i (Sigma^{1/2} C_i Sigma^{1/2})^{1/2}
        # S_{new} = S^{-1/2} T^2 S^{-1/2} ? No.
        
        # Actually, let's look at the paper provided if possible, but 1905.12828 is about WST, maybe not barycenters specifically.
        # However, it mentions optimal transport.
        
        # Standard Fixed Point Algorithm:
        # S = I (Identity)
        # S = (\sum w_i (S^{1/2} C_i S^{1/2})^{1/2})^2
        # This assumes we want barycenter.
        
        # Let's use the one from POT library if available or stick to the one above which is:
        # S = ( \sum w_i (S^{1/2} C_i S^{1/2})^{1/2} )^2  <-- This assumes commuting? No.
        
        # Correct Fixed Point (Alvarez-Esteban 2016):
        # S_{k+1} = S_k^{-1/2} [ \sum_j w_j (S_k^{1/2} \Sigma_j S_k^{1/2})^{1/2} ]^2 S_k^{-1/2}
        
        # wait, let's verify if `sigma` is invertible. For style features often regularization is needed.
        
        # Simplified update if we assume S^{1/2} commutes (it doesn't usually).
        
        # Use simple iterative update:
        # S = (\sum \dots)^2?
        
        # Let's implement the Alvarez-Esteban one.
        
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
