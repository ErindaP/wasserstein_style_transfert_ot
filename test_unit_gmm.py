import torch
import time
from wst import gmm_transfer

def test():
    print("Testing GMM Transfer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = 0.5
    # Create random features similar to Relu4_1 (512x64x64)
    # But let's use smaller for speed: 64 channels, 64x64 spatial
    C = 64
    H = 64
    W = 64
    
    cf = torch.randn(C, H, W).to(device)
    styles = [torch.randn(C, H, W).to(device)]
    style_weights = [1.0]
    K = 3
    
    start = time.time()
    res = gmm_transfer(alpha, cf, styles, style_weights, K)
    end = time.time()
    
    print(f"Success! Output shape: {res.shape}")
    print(f"Time taken: {end - start:.4f}s")

if __name__ == "__main__":
    test()
