import torch
from dataset_sd35 import HairInpaintingDataset
from torchvision import transforms
from PIL import Image
import os

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def main():
    dataset = HairInpaintingDataset(data_root="dataset3", size=512, mode="train")
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Dataset empty. Exiting.")
        return
        
    os.makedirs("test_outputs", exist_ok=True)
    
    # Try getting a few samples
    for i in range(min(4, len(dataset))):
        sample = dataset[i]
        
        # Denormalize and save target
        target = denormalize(sample["pixel_values"])
        target_img = transforms.ToPILImage()(target)
        target_img.save(f"test_outputs/target_{i}.png")
        
        # Save mask
        mask = sample["masks"].squeeze(0)
        mask_img = transforms.ToPILImage()(mask)
        mask_img.save(f"test_outputs/mask_{i}.png")
        
        # Denormalize and save cond
        cond = denormalize(sample["conditioning_pixel_values"])
        cond_img = transforms.ToPILImage()(cond)
        cond_img.save(f"test_outputs/cond_{i}.png")
        
        print(f"Saved sample {i}")

if __name__ == "__main__":
    main()
