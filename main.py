import argparse
import torch
import os
from model import MultiLevelStyleTransfer
from utils import load_image, save_image
import time

def main():
    parser = argparse.ArgumentParser(description="Wasserstein Style Transfer (Gaussian)")
    parser.add_argument("content", type=str, help="Path to content image")
    parser.add_argument("style", type=str, help="Path to style image")
    parser.add_argument("--out", type=str, default="output.jpg", help="Output path")
    parser.add_argument("--alpha", type=float, default=0.5, help="Style transfer intensity (0-1)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    print(f"Device: {args.device}")
    
    # Load images
    # We don't resize by default, assuming user provides images of compatible size or VGG handles it.
    # But VGG usually works better if images are similar size.
    content = load_image(args.content).to(args.device)
    style = load_image(args.style).to(args.device)
    
    print(f"Content size: {content.shape}")
    print(f"Style size: {style.shape}")
    
    # Initialize model
    model = MultiLevelStyleTransfer(alpha=args.alpha, device=args.device)
    
    # Run style transfer
    start = time.time()
    result = model(content, style)
    end = time.time()
    
    print(f"Time: {end - start:.2f}s")
    
    # Save result
    save_image(result, args.out)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
