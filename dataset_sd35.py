
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
        
        self.braid_image_paths = []
        self.braid_mask_paths = []
        self.braid_cond_paths = []
        
        self.unbraid_image_paths = []
        self.unbraid_mask_paths = []
        self.unbraid_cond_paths = []

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
                if cat == "braid":
                    self.braid_image_paths.append(os.path.join(img_dir, name))
                    self.braid_mask_paths.append(os.path.join(mask_dir, name))
                    self.braid_cond_paths.append(os.path.join(cond_dir, name))
                elif cat == "unbraid":
                    self.unbraid_image_paths.append(os.path.join(img_dir, name))
                    self.unbraid_mask_paths.append(os.path.join(mask_dir, name))
                    self.unbraid_cond_paths.append(os.path.join(cond_dir, name))
                else: 
                    # If root data without distinction
                    self.unbraid_image_paths.append(os.path.join(img_dir, name))
                    self.unbraid_mask_paths.append(os.path.join(mask_dir, name))
                    self.unbraid_cond_paths.append(os.path.join(cond_dir, name))

        self.braid_len = len(self.braid_image_paths)
        self.unbraid_len = len(self.unbraid_image_paths)
        
        if self.braid_len == 0 and self.unbraid_len == 0:
            print(f"Warning: No images found in {data_root} with structure {{braid/unbraid}}/img/{mode} or direct img/{mode}")

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ])
        
        # We handle mask transforms manually in getitem now to combine with affine
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        # Oversample smaller dataset to match 1:1 ratio
        max_len = max(self.braid_len, self.unbraid_len)
        if self.category is None and self.braid_len > 0 and self.unbraid_len > 0:
            return max_len * 2
        else:
            return max(self.braid_len, self.unbraid_len)

    def __getitem__(self, idx):
        # Determine whether to pick from braid or unbraid to maintain 1:1 ratio
        if self.category is None and self.braid_len > 0 and self.unbraid_len > 0:
            if idx % 2 == 0:
                # Pick braid (oversampled if necessary)
                actual_idx = (idx // 2) % self.braid_len
                img_path = self.braid_image_paths[actual_idx]
                mask_path = self.braid_mask_paths[actual_idx]
                cond_path = self.braid_cond_paths[actual_idx]
            else:
                # Pick unbraid (oversampled if necessary)
                actual_idx = (idx // 2) % self.unbraid_len
                img_path = self.unbraid_image_paths[actual_idx]
                mask_path = self.unbraid_mask_paths[actual_idx]
                cond_path = self.unbraid_cond_paths[actual_idx]
        else:
            # Fallback for single category
            if self.braid_len > 0:
                actual_idx = idx % self.braid_len
                img_path = self.braid_image_paths[actual_idx]
                mask_path = self.braid_mask_paths[actual_idx]
                cond_path = self.braid_cond_paths[actual_idx]
            else:
                actual_idx = idx % self.unbraid_len
                img_path = self.unbraid_image_paths[actual_idx]
                mask_path = self.unbraid_mask_paths[actual_idx]
                cond_path = self.unbraid_cond_paths[actual_idx]

        # 1. Target Image (RGB)
        target_img = Image.open(img_path).convert("RGB")
        target_img = target_img.resize((self.size, self.size), Image.BILINEAR)
        pixel_values = transforms.ToTensor()(target_img)
        pixel_values = transforms.Normalize([0.5], [0.5])(pixel_values)

        # 2. Mask (L) and Condition Image (RGB) Setup
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")
        else:
            mask_img = Image.new("L", (self.size, self.size), color=255)

        if os.path.exists(cond_path):
            cond_img = Image.open(cond_path).convert("RGB")
        else:
            cond_img = Image.new("RGB", target_img.size, (128, 128, 128))

        mask_img = mask_img.resize((self.size, self.size), Image.NEAREST)
        cond_img = cond_img.resize((self.size, self.size), Image.BILINEAR)

        # --- Data Augmentation on Condition and Mask only ---
        import torchvision.transforms.functional as TF
        import random
        
        if self.mode == "train":
            # Random Horizontal Flip (50% chance)
            if random.random() > 0.5:
                mask_img = TF.hflip(mask_img)
                cond_img = TF.hflip(cond_img)
            
            # Random Affine (Rotation and Translation)
            angle = random.uniform(-15, 15)
            # max translation 50 pixels
            translate_x = int(random.uniform(-50, 50))
            translate_y = int(random.uniform(-50, 50))
            
            # Use basic translate for simpler operations if PyTorch versions complain
            mask_img = TF.affine(mask_img, angle=angle, translate=[translate_x, translate_y], scale=1.0, shear=0)
            cond_img = TF.affine(cond_img, angle=angle, translate=[translate_x, translate_y], scale=1.0, shear=0)


        # Apply Soft Masking Strategy dynamically using OpenCV 
        mask_np_resized = np.array(mask_img)
        _, mask_bin = cv2.threshold(mask_np_resized, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)
        mask_soft = cv2.GaussianBlur(mask_dilated, (0, 0), sigmaX=10)
        mask_tensor = torch.from_numpy(mask_soft.astype(np.float32) / 255.0).unsqueeze(0)

        # 3. Transform final condition image
        conditioning_pixel_values = transforms.ToTensor()(cond_img)
        conditioning_pixel_values = transforms.Normalize([0.5], [0.5])(conditioning_pixel_values)
        
        return {
            "pixel_values": pixel_values,      
            "conditioning_pixel_values": conditioning_pixel_values, 
            "masks": mask_tensor,             
            "prompt": "A hairstyle"           
        }
