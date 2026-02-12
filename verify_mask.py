
import os
import cv2
import numpy as np
import argparse
from glob import glob

def verify_mask(data_root, output_path="mask_check.png"):
    # Find a mask file - Robust Search
    # Try multiple patterns
    patterns = [
        os.path.join(data_root, "matte", "train", "*.png"),
        os.path.join(data_root, "matte", "*.png"),
        os.path.join(data_root, "*", "matte", "train", "*.png"), # dataset/unbraid/matte/train
        os.path.join(data_root, "*", "matte", "*.png"),          # dataset/unbraid/matte
    ]
    
    mask_files = []
    for p in patterns:
        files = glob(p)
        if files:
            mask_files.extend(files)
            break
            
    if not mask_files:
        print(f"No masks found in {data_root} using patterns: {patterns}")
        return

    # Pick the first one
    mask_path = mask_files[0]
    print(f"Processing mask: {mask_path}")
    
    # Load
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Failed to load mask.")
        return

    # 1. Resize to 1024 (Training Size)
    mask_resized = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    
    # 2. Binarize
    _, mask_bin = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
    
    # 3. Dilate (Expand)
    # Kernel size 15 is roughly 1.5% of 1024px, seems reasonable for hairline
    kernel_size = 15 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    
    # 4. Blur (Soften)
    # Sigma 10 gives a nice gradient over ~30 pixels
    mask_soft = cv2.GaussianBlur(mask_dilated, (0, 0), sigmaX=10)
    
    # Normalize for visualization (0-255)
    
    # Create comparison
    # Stack: Original (Resized) | Diff (Expanded Area) | Final Soft Mask
    diff = cv2.absdiff(mask_dilated, mask_bin)
    
    # Visualizing the gradient: normalize mask_soft
    
    concat = np.hstack((mask_resized, diff, mask_soft))
    
    cv2.imwrite(output_path, concat)
    print(f"Saved comparison to {output_path}")
    print("Left: Original, Center: Added Area, Right: Final Soft Mask")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset") # Default relative path
    args = parser.parse_args()
    
    verify_mask(args.data_root)
