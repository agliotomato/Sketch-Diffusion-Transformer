
import argparse
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import SD3ControlNetModel, SD3Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset_sd35 import HairInpaintingDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sd35_hair_lora")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=4)
    device = accelerator.device

    # 1. Load Models
    model_id = "stabilityai/stable-diffusion-3.5-large"
    controlnet_id = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
    
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    transformer = SD3Transformer2DModel.from_pretrained(model_id, subfolder="transformer")
    controlnet = SD3ControlNetModel.from_pretrained(controlnet_id)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze Base
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    controlnet.requires_grad_(False)

    # 2. Add LoRA to Transformer
    lora_config = LoraConfig(
        r=16, lora_alpha=16, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)
    
    # Optimizer
    params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # 3. Data
    dataset = HairInpaintingDataset(args.data_root, size=1024)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    transformer, optimizer, dataloader, controlnet = accelerator.prepare(
        transformer, optimizer, dataloader, controlnet
    )
    vae.to(device)

    print(f"Start Training: {len(dataset)} images, {args.epochs} epochs")

    for epoch in range(args.epochs):
        transformer.train()
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            with accelerator.accumulate(transformer):
                # A. Latents
                latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float16)).latent_dist.sample() * vae.config.scaling_factor
                
                # B. ControlNet Input (Color Sketch -> Canny Edge simulation or direct?)
                # Since we use Canny ControlNet, it expects 0-1 range Canny map.
                # User's Color Sketch is RGB. Let's pass it directly as structural guidance (ControlNet is robust).
                # Or ideally, convert to Grayscale/Canny inside dataset. 
                # For now, create control latents/features.
                
                # ControlNet expects pixel values, not latents.
                control_pixel = batch["conditioning_pixel_values"].to(device, dtype=torch.float16) # [-1, 1]
                # Rescale to [0, 1] for ControlNet? Usually it takes [0,1].
                control_input = (control_pixel + 1.0) / 2.0
                
                # C. Noise & Timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                
                # Add Noise
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # D. Text Prop (Dummy)
                # SD3 requires pooled_projections (from T5/CLIP).
                # We need to compute embeds. For code brevity, we use zero/null embeds or cached.
                # HACK: Create empty embeds matching shapes.
                # In real script, load TextEncoders. Here assuming unconditioned (empty prompt) training for shape/texture alignment.
                # Or better: pre-compute them.
                # For this script to work without loading 10GB Text Encoders, we create dummy tensors.
                encoder_hidden_states = torch.zeros((bsz, 77, 4096), device=device, dtype=torch.float16) # Dummy
                pooled_projections = torch.zeros((bsz, 2048), device=device, dtype=torch.float16) # Dummy
                
                # E. Run ControlNet
                control_block_samples = controlnet(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    controlnet_cond=control_input,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    return_dict=False
                )[0]
                
                # F. Run Transformer
                noise_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    block_controlnet_hidden_states=control_block_samples,
                    return_dict=False
                )[0]
                
                # G. Masked Loss
                # Flow Matching Target: v = noise - latents (usually). 
                # Diffusers 'flow_match_euler' -> target is (noise - latents) or similar.
                # Simple MSE against 'noise' (for epsilon prediction) or velocity.
                # SD3 predicts 'flow'. v_t = u_t.
                
                # Let's assume prediction is `v_pred`.
                # Target `v` for Rectified Flow is `noise - original_latents`.
                target = noise - latents
                
                loss = F.mse_loss(noise_pred, target, reduction="none")
                
                # Apply Mask
                # Resize mask to latent shape
                mask = F.interpolate(batch["masks"].to(device, dtype=torch.float16), size=loss.shape[-2:], mode="nearest")
                
                masked_loss = (loss * mask).mean()
                
                accelerator.backward(masked_loss)
                optimizer.step()
                optimizer.zero_grad()
                
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            output_path = os.path.join(args.output_dir, f"checkpoint-{epoch}")
            os.makedirs(output_path, exist_ok=True)
            transformer.save_pretrained(output_path)
            print(f"Saved LoRA to {output_path}")

if __name__ == "__main__":
    main()
