import torch

def sqrtm(M):
    """Compute the square root of a positive semidefinite matrix"""
    _, s, v = M.svd()
    # truncate small components
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

def gaussian_transfer(alpha, cf, sf):
    """Mroueh, Y. (2019). Wasserstein style transfer"""
    # Approximate content features by Gaussian
    cf = cf.double()
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    cfv = cf.view(c_channels, -1)

    c_mean = torch.mean(cfv, 1)
    c_mean = c_mean.unsqueeze(1).expand_as(cfv)
    cfv = cfv - c_mean

    c_covm = torch.mm(cfv, cfv.t()).div((c_width * c_height) - 1)
    c_covm_sqrt = sqrtm(c_covm)
    c_covm_sqrt_inv = torch.inverse(c_covm_sqrt)
    
    # Approximate style features by Gaussian
    sf = sf.double()
    _, s_width, s_heigth = sf.size(0), sf.size(1), sf.size(2)
    sfv = sf.view(c_channels, -1)

    s_mean = torch.mean(sfv, 1)
    s_mean = s_mean.unsqueeze(1).expand_as(sfv)
    sfv = sfv - s_mean
    s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_heigth) - 1)

    A = c_covm_sqrt_inv @ sqrtm(c_covm_sqrt @ s_covm @ c_covm_sqrt) @ c_covm_sqrt_inv
    pushed = (s_mean + A @ cfv).view_as(cf)

    res = (1-alpha) * cf + alpha * pushed
    return res.float().unsqueeze(0)
