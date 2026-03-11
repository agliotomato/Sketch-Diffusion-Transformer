import os
import random
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms


class HairInpaintingDataset(Dataset):
    def __init__(self, data_root, size=1024, mode="train", category=None):
        self.data_root = data_root
        self.size = size
        self.mode = mode

        self.image_paths = []
        self.mask_paths = []
        self.sketch_paths = []

        if category:
            categories = [category]
        else:
            subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
            if "braid" in subdirs or "unbraid" in subdirs:
                categories = [d for d in subdirs if d in ["braid", "unbraid"]]
            else:
                categories = ["."]

        for cat in categories:
            cat_root = data_root if cat == "." else os.path.join(data_root, cat)

            img_dir    = os.path.join(cat_root, "img",    mode)
            mask_dir   = os.path.join(cat_root, "matte",  mode)
            sketch_dir = os.path.join(cat_root, "sketch", mode)

            if not os.path.exists(img_dir):
                print(f"Warning: {img_dir} not found. Skipping category '{cat}'.")
                continue

            names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
            for name in names:
                self.image_paths.append(os.path.join(img_dir, name))
                self.mask_paths.append(os.path.join(mask_dir, name))
                self.sketch_paths.append(os.path.join(sketch_dir, name))

        print(f"[Dataset] Loaded {len(self.image_paths)} images (category={category or 'all'}, mode={mode})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load ---
        image = Image.open(self.image_paths[idx]).convert("RGB").resize((self.size, self.size), Image.BILINEAR)

        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size), Image.BILINEAR) \
            if os.path.exists(mask_path) else Image.new("L", (self.size, self.size), 255)

        sketch_path = self.sketch_paths[idx]
        sketch = Image.open(sketch_path).convert("RGB").resize((self.size, self.size), Image.BILINEAR) \
            if os.path.exists(sketch_path) else Image.new("RGB", (self.size, self.size), (0, 0, 0))

        # --- Augmentation (train only) ---
        if self.mode == "train":
            image, mask, sketch = self._augment(image, mask, sketch)

        # --- hair_target = image × matte ---
        matte_tensor = transforms.ToTensor()(mask)          # (1, H, W), [0, 1]
        image_tensor = transforms.ToTensor()(image)         # (3, H, W), [0, 1]
        hair_target  = image_tensor * matte_tensor          # (3, H, W), hair region only

        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        pixel_values  = normalize(hair_target)                           # (3, H, W), [-1, 1]
        sketch_tensor = normalize(transforms.ToTensor()(sketch))         # (3, H, W), [-1, 1]

        return {
            "pixel_values": pixel_values,   # hair patch only
            "sketch":       sketch_tensor,
            "matte":        matte_tensor,   # (1, H, W), [0, 1]
        }

    # ------------------------------------------------------------------

    def _augment(self, image, mask, sketch):
        # 1. Geometry — synchronized across all three
        if random.random() > 0.5:
            image  = TF.hflip(image)
            mask   = TF.hflip(mask)
            sketch = TF.hflip(sketch)

        angle = random.uniform(-10, 10)
        tx    = int(random.uniform(-0.1 * self.size, 0.1 * self.size))
        ty    = int(random.uniform(-0.1 * self.size, 0.1 * self.size))
        scale = random.uniform(0.9, 1.1)

        image  = TF.affine(image,  angle=angle, translate=[tx, ty], scale=scale, shear=0,
                           interpolation=TF.InterpolationMode.BILINEAR)
        mask   = TF.affine(mask,   angle=angle, translate=[tx, ty], scale=scale, shear=0,
                           interpolation=TF.InterpolationMode.NEAREST)
        sketch = TF.affine(sketch, angle=angle, translate=[tx, ty], scale=scale, shear=0,
                           interpolation=TF.InterpolationMode.NEAREST)

        # 2. Sketch color randomization — color = structural cue, not appearance
        sketch = TF.adjust_hue(sketch, random.uniform(-0.5, 0.5))

        # 3. Sketch thickness jitter — robustness to line thickness
        op = random.random()
        k  = random.choice([3, 5])
        if op < 0.33:
            sketch = sketch.filter(ImageFilter.MaxFilter(size=k))       # dilation
        elif op < 0.66:
            sketch = sketch.filter(ImageFilter.MinFilter(size=k))       # erosion
        else:
            sketch = sketch.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # 4. Matte boundary perturbation — boundary robustness
        bk = random.choice([3, 5, 7])
        if random.random() < 0.33:
            mask = mask.filter(ImageFilter.MaxFilter(size=bk))          # dilation
        elif random.random() < 0.66:
            mask = mask.filter(ImageFilter.MinFilter(size=bk))          # erosion
        mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))

        # 5. Hair appearance augmentation — decouple structure from appearance
        image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))
        image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        image = TF.adjust_hue(image,        random.uniform(-0.1, 0.1))

        return image, mask, sketch
