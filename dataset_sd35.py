
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
        else:
            print(f"✅ [Dataset] Successfully loaded {len(self.image_paths)} images for category '{category if category else 'All'}'.")
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
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
        target_img = target_img.resize((self.size, self.size), Image.BILINEAR)
        # Note: ToTensor and Normalize moved below augmentation!

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
        cond_img = cond_img.resize((self.size, self.size), Image.NEAREST)

        # --- Data Augmentation on All Inputs (MUST be spatially aligned) ---
        import torchvision.transforms.functional as TF
        import random
        
        if self.mode == "train":
            # Random Horizontal Flip (50% chance)
            if random.random() > 0.5:
                target_img = TF.hflip(target_img)
                mask_img = TF.hflip(mask_img)
                cond_img = TF.hflip(cond_img)
            
            # Random Affine (Rotation and Translation)
            angle = random.uniform(-15, 15)
            # max translation 50 pixels
            translate_x = int(random.uniform(-50, 50))
            translate_y = int(random.uniform(-50, 50))
            
            # Apply identically to maintain perfect spatial alignment!
            # Target (RGB) uses BILINEAR, Mask/Sketch (Categorical) use NEAREST
            target_img = TF.affine(target_img, angle=angle, translate=[translate_x, translate_y], scale=1.0, shear=0, interpolation=TF.InterpolationMode.BILINEAR)
            mask_img = TF.affine(mask_img, angle=angle, translate=[translate_x, translate_y], scale=1.0, shear=0, interpolation=TF.InterpolationMode.NEAREST)
            cond_img = TF.affine(cond_img, angle=angle, translate=[translate_x, translate_y], scale=1.0, shear=0, interpolation=TF.InterpolationMode.NEAREST)

        # Transform final target image
        pixel_values = transforms.ToTensor()(target_img)
        pixel_values = transforms.Normalize([0.5], [0.5])(pixel_values)

        # Directly convert the soft mask to a tensor (0.0 ~ 1.0)
        # This preserves the predicted matte's natural soft boundaries
        mask_tensor = transforms.ToTensor()(mask_img)

        # 3. Transform final condition image
        conditioning_pixel_values = transforms.ToTensor()(cond_img)
        conditioning_pixel_values = transforms.Normalize([0.5], [0.5])(conditioning_pixel_values)
        
        return {
            "pixel_values": pixel_values,      
            "conditioning_pixel_values": conditioning_pixel_values, 
            "masks": mask_tensor,             
            "prompt": "A hairstyle"           
        }
