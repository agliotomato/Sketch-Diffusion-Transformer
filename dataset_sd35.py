
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HairInpaintingDataset(Dataset):
    def __init__(self, data_root, size=1024, mode="train"):
        self.data_root = data_root
        self.size = size
        self.mode = mode
        
        # Paths assuming SketchHairSalon structure
        # data_root/img (Target)
        # data_root/matte (Mask)
        # data_root/sketch (Color Sketch - assuming this folder contains the color-coded sketches)
        
        # Note: User mentioned "Color Sketch" is the input. 
        # In the provided 'data/color_coding.py', it generates this on the fly or saves it?
        # The user said "I received img, matte, and color sketch". 
        # So we assume there is a folder for 'color_sketch' OR we generate it if only raw sketch exists.
        # Let's assume there is a folder named 'sketch' which contains the color sketches based on user context,
        # or we might need to assume 'input_1' style naming.
        # Let's verify directory structure later, but for now standard names:
        
        self.img_dir = os.path.join(data_root, "img")
        self.mask_dir = os.path.join(data_root, "matte")
        self.cond_dir = os.path.join(data_root, "sketch") # Color Sketch
        
        # Collect files
        self.image_names = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_names.sort()

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # [-1, 1] for SD
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Load Images
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        cond_path = os.path.join(self.cond_dir, img_name)
        
        # 1. Target Image (RGB)
        target_img = Image.open(img_path).convert("RGB")
        
        # 2. Mask (L)
        # Mask needs to be 1 for Hair (Loss region), 0 for Face (No Loss).
        # Usually mattes are White(255) for hair.
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")
        else:
            # Create dummy mask if missing (Full training?)
            mask_img = Image.new("L", target_img.size, 255)

        # 3. Condition Image (Color Sketch) (RGB)
        if os.path.exists(cond_path):
            cond_img = Image.open(cond_path).convert("RGB")
        else:
            cond_img = Image.new("RGB", target_img.size, (128, 128, 128))

        # Apply Transforms
        pixel_values = self.transform(target_img)
        conditioning_pixel_values = self.transform(cond_img)
        mask = self.mask_transform(mask_img)
        
        # Binarize Mask for hard masking (Optional but recommended)
        mask = (mask > 0.5).float()

        return {
            "pixel_values": pixel_values,      # Target (Original)
            "conditioning_pixel_values": conditioning_pixel_values, # Input (Color Sketch)
            "masks": mask,                    # Loss Mask
            "prompt": "A hairstyle"           # Dummy prompt (or load from txt if avail)
        }
