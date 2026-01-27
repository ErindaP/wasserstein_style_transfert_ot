import argparse
import torch
import os
from model import MultiLevelStyleTransfer
from utils import load_image, save_image
import time

def main():
    parser = argparse.ArgumentParser(description="Wasserstein Style Transfer (Gaussian)")
    parser.add_argument("content", type=str, help="Path to content image")
    parser.add_argument("styles", nargs='+', type=str, help="Path to style image(s)")
    parser.add_argument("--out", type=str, default="output.jpg", help="Output path")
    parser.add_argument("--alpha", type=float, default=0.5, help="Style transfer intensity (0-1)")
    parser.add_argument("--style_weights", type=float, nargs='+', help="Weights for each style image (must sum to 1)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    print(f"Device: {args.device}")
    
    # Load content
    content = load_image(args.content).to(args.device)
    print(f"Content size: {content.shape}")
    
    # Load styles
    styles = []
    for s_path in args.styles:
        s_img = load_image(s_path).to(args.device)
        styles.append(s_img)
        print(f"Loaded style: {s_path} {s_img.shape}")
        
    # Process weights
    if args.style_weights:
        if len(args.style_weights) != len(styles):
            print("Error: Number of weights must match number of style images.")
            return
        # Normalize weights
        total_weight = sum(args.style_weights)
        weights = [w / total_weight for w in args.style_weights]
    else:
        weights = [1.0 / len(styles)] * len(styles)
    
    print(f"Style Weights: {weights}")
    
    # Initialize model
    model = MultiLevelStyleTransfer(alpha=args.alpha, style_weights=weights, device=args.device)
    
    # Run style transfer
    start = time.time()
    # Pass list of styles
    result = model(content, styles)
    end = time.time()
    
    print(f"Time: {end - start:.2f}s")
    
    # Save result
    save_image(result, args.out)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
