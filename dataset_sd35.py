
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HairInpaintingDataset(Dataset):
    def __init__(self, data_root, size=1024, mode="train", category=None):
        self.data_root = data_root
        self.size = size
        self.mode = mode
        self.category = category
        
        self.image_paths = []
        self.mask_paths = []
        self.cond_paths = []

        # Determine categories to load
        # specific category: ['braid'] or ['unbraid']
        # None (all): ['braid', 'unbraid'] if present, else look in root
        if category:
            categories = [category]
        else:
            # Check for subdirectories
            subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
            if "braid" in subdirs or "unbraid" in subdirs:
                categories = [d for d in subdirs if d in ["braid", "unbraid"]]
            else:
                categories = ["."] # Root

        for cat in categories:
            if cat == ".":
                cat_root = data_root
            else:
                cat_root = os.path.join(data_root, cat)
            
            img_dir = os.path.join(cat_root, "img", mode)
            mask_dir = os.path.join(cat_root, "matte", mode)
            cond_dir = os.path.join(cat_root, "sketch", mode)
            
            if not os.path.exists(img_dir):
                print(f"Warning: {img_dir} does not exist. Skipping category {cat}")
                continue

            names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            names.sort()
            
            for name in names:
                self.image_paths.append(os.path.join(img_dir, name))
                self.mask_paths.append(os.path.join(mask_dir, name))
                self.cond_paths.append(os.path.join(cond_dir, name))

        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {data_root} with structure {{braid/unbraid}}/img/{mode} or direct img/{mode}")

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load Images
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        cond_path = self.cond_paths[idx]
        
        # 1. Target Image (RGB)
        target_img = Image.open(img_path).convert("RGB")
        
        # 2. Mask (L)
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                mask_img = np.zeros((self.size, self.size), dtype=np.uint8) + 255
            else:
                mask_img = cv2.resize(mask_img, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        else:
            mask_img = np.zeros((self.size, self.size), dtype=np.uint8) + 255
            
        # Soft Masking Strategy
        _, mask_bin = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)
        mask_soft = cv2.GaussianBlur(mask_dilated, (0, 0), sigmaX=10)
        mask_tensor = torch.from_numpy(mask_soft.astype(np.float32) / 255.0).unsqueeze(0)

        # 3. Condition Image (Color Sketch) (RGB)
        if os.path.exists(cond_path):
            cond_img = Image.open(cond_path).convert("RGB")
        else:
            cond_img = Image.new("RGB", target_img.size, (128, 128, 128))

        # Apply Transforms
        pixel_values = self.transform(target_img)
        conditioning_pixel_values = self.transform(cond_img)
        
        return {
            "pixel_values": pixel_values,      
            "conditioning_pixel_values": conditioning_pixel_values, 
            "masks": mask_tensor,             
            "prompt": "A hairstyle"           
        }
