import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from dataset_sd35 import HairInpaintingDataset
import numpy as np

def verify():
    model_name = "stabilityai/stable-diffusion-3.5-medium"
    data_root = "./dataset3"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading VAE from {model_name}...")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float16).to(device)
    vae.requires_grad_(False)

    print("Loading dataset...")
    dataset = HairInpaintingDataset(data_root, size=1024, mode="train", category="braid")
    # Take a small batch
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))

    pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
    sketches = batch["conditioning_pixel_values"].to(device, dtype=torch.float16)

    with torch.no_grad():
        # Encode
        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        sketch_latents = vae.encode(sketches).latent_dist.sample() * vae.config.scaling_factor

    print("\n--- Statistics ---")
    print(f"VAE Scaling Factor: {vae.config.scaling_factor}")
    
    print(f"Image Pixel  - Mean: {pixel_values.mean():.4f}, Std: {pixel_values.std():.4f}")
    print(f"Sketch Pixel - Mean: {sketches.mean():.4f}, Std: {sketches.std():.4f}")
    
    print(f"Image Latent  - Mean: {latents.mean():.4f}, Std: {latents.std():.4f}")
    print(f"Sketch Latent - Mean: {sketch_latents.mean():.4f}, Std: {sketch_latents.std():.4f}")

    # Latent magnitude check
    ratio = sketch_latents.std() / latents.std()
    print(f"\nStd Ratio (Sketch/Image): {ratio:.4f}")
    
    if ratio < 0.2 or ratio > 5.0:
        print("⚠️ Warning: Latent scales are significantly different!")
    else:
        print("✅ Latent scales are within a reasonable range.")

if __name__ == "__main__":
    verify()
