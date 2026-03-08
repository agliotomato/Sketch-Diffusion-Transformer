
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import kornia
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from PIL import Image
import os

def visualize_hybrid_loss_components(image_path, sketch_path, mask_path=None):
    # 1. Load and Preprocess
    img = Image.open(image_path).convert("RGB").resize((1024, 1024))
    sketch = Image.open(sketch_path).convert("RGB").resize((1024, 1024))
    
    # Binarize Sketch (Same as our dataset logic)
    sketch_np = np.array(sketch)
    gray = cv2.cvtColor(sketch_np, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) < 127: # Black bg
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    else: # White bg
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    sketch_binary = Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

    # Convert to Tensors
    to_tensor = transforms.ToTensor()
    img_t = to_tensor(img).unsqueeze(0) # [1, 3, 1024, 1024]
    sketch_t = to_tensor(sketch_binary).unsqueeze(0)
    
    # 2. Gaussian Shape Components (11, 21, 31)
    shape_kernels = [11, 21, 31]
    blurred_imgs = []
    for k in shape_kernels:
        sigma = k / 4.0
        blurred = gaussian_blur(img_t, kernel_size=[k, k], sigma=[sigma, sigma])
        blurred_imgs.append(blurred[0].permute(1, 2, 0).numpy())

    # 3. Sobel Gradient Components (3, 5, 7)
    grad_kernels = [3, 5, 7]
    edge_maps = []
    for k in grad_kernels:
        if k > 3:
            p = gaussian_blur(img_t, kernel_size=[k, k], sigma=[(k-1)/4.0, (k-1)/4.0])
        else:
            p = img_t
        grad = kornia.filters.sobel(p)
        # Normalize for visualization
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        edge_maps.append(grad[0].permute(1, 2, 0).numpy())

    # 4. Plotting
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Row 1: Original & Binarized Sketch
    axes[0, 0].imshow(np.array(img))
    axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(np.array(sketch))
    axes[0, 1].set_title("Original Sketch")
    axes[0, 2].imshow(np.array(sketch_binary))
    axes[0, 2].set_title("Binarized Sketch (Input)")

    # Row 2: Gaussian Blurs (Shape Guidance)
    axes[1, 0].imshow(blurred_imgs[0])
    axes[1, 0].set_title("Gaussian 11x11")
    axes[1, 1].imshow(blurred_imgs[1])
    axes[1, 1].set_title("Gaussian 21x21")
    axes[1, 2].imshow(blurred_imgs[2])
    axes[1, 2].set_title("Gaussian 31x31")

    # Row 3: Sobel Edges (Gradient Guidance)
    axes[2, 0].imshow(edge_maps[0], cmap='gray')
    axes[2, 0].set_title("Sobel (Direct)")
    axes[2, 1].imshow(edge_maps[1], cmap='gray')
    axes[2, 1].set_title("Sobel (via 5x5 Blur)")
    axes[2, 2].imshow(edge_maps[2], cmap='gray')
    axes[2, 2].set_title("Sobel (via 7x7 Blur)")

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("hybrid_loss_visualization.png")
    print("✅ Visualization saved as 'hybrid_loss_visualization.png'")
    plt.show()

if __name__ == "__main__":
    # Test with sample files found in the root
    img_path = "braid_1.png"
    sketch_path = "braid_1_sketch.png"
    if os.path.exists(img_path) and os.path.exists(sketch_path):
        visualize_hybrid_loss_components(img_path, sketch_path)
    else:
        print(f"Sample files not found in current directory.")
