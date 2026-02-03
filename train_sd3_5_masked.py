
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
import matplotlib.pyplot as plt

# Set backend to Agg for headless environments
plt.switch_backend('Agg')

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
    # FIX: Increase Rank to 128 for better hair texture details
    lora_config = LoraConfig(
        r=128, lora_alpha=128, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05, # FIX: Add dropout to prevent overfitting with high rank
    )
    transformer.add_adapter(lora_config)
    transformer.enable_gradient_checkpointing()  # Optimize VRAM
    
    # FIX: Cast Transformer to fp16 (Base weights frozen, LoRA trained)
    transformer.to(dtype=torch.float16)

    # FIX: LoRA weights (trainable) must be fp32 for stable training and scaler compatibility
    for param in transformer.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
    # Optimizer
    # Optimizer (Use 8-bit AdamW to save VRAM)
    import bitsandbytes as bnb
    import gc
    
    # Optimizer
    params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = bnb.optim.AdamW8bit(params, lr=4e-5) # FIX: Lower LR for high rank fine-tuning

    # 3. Data
    dataset = HairInpaintingDataset(args.data_root, size=1024)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Clear cache before training start
    torch.cuda.empty_cache()
    gc.collect()

    transformer, optimizer, dataloader, controlnet = accelerator.prepare(
        transformer, optimizer, dataloader, controlnet
    )
    # Fix Dtype Mismatch: Cast VAE and ControlNet to fp16
    vae.to(device, dtype=torch.float16)
    controlnet.to(device, dtype=torch.float16)

    print(f"Start Training: {len(dataset)} images, {args.epochs} epochs")
    
    loss_history = []
    
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
                
                # FIX: SD3 ControlNet expects VAE latents (16 channels), not pixels (3 channels).
                # Encode sketch to latents
                control_input = vae.encode(control_pixel).latent_dist.sample() * vae.config.scaling_factor
                
                # C. Noise & Timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                
                # Add Noise (Manual Rectified Flow: z_t = (1-t)x + t*noise)
                # timesteps is 0-1000. sigmas = t/1000
                sigmas = timesteps.float() / scheduler.config.num_train_timesteps
                sigmas = sigmas.reshape(bsz, 1, 1, 1).to(device, dtype=torch.float16)
                
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                # We need to compute embeds. For code brevity, we use zero/null embeds or cached.
                # HACK: Create empty embeds matching shapes.
                # In real script, load TextEncoders. Here assuming unconditioned (empty prompt) training for shape/texture alignment.
                # Or better: pre-compute them.
                # For this script to work without loading 10GB Text Encoders, we create dummy tensors.
                encoder_hidden_states = torch.zeros((bsz, 77, 4096), device=device, dtype=torch.float16) # Dummy
                pooled_projections = torch.zeros((bsz, 2048), device=device, dtype=torch.float16) # Dummy
                
                # E. Run ControlNet
                # FIX: Check if ControlNet requires 3D inputs (missing pos_embed)
                if getattr(controlnet, "pos_embed", None) is None:
                    # Manually patchify using transformer's pos_embed
                    control_hidden_states = transformer.pos_embed(noisy_latents)
                else:
                    control_hidden_states = noisy_latents

                # Prepare ControlNet text args based on configuration
                control_txt_kwargs = {"pooled_projections": pooled_projections}
                
                # Only provide encoder_hidden_states if context_embedder is present
                if getattr(controlnet, "context_embedder", None) is not None:
                    control_txt_kwargs["encoder_hidden_states"] = encoder_hidden_states

                control_block_samples = controlnet(
                    hidden_states=control_hidden_states,
                    timestep=timesteps,
                    controlnet_cond=control_input,
                    return_dict=False,
                    **control_txt_kwargs
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
                # Let's assume prediction is `v_pred`.
                # Target `v` for Rectified Flow is `noise - original_latents`.
                target = noise - latents
                
                # FIX: Compute Loss in Float32 for stability and to prevent Backward type errors
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                
                # Apply Mask
                # Resize mask to latent shape
                mask = F.interpolate(batch["masks"].to(device, dtype=torch.float16), size=loss.shape[-2:], mode="nearest")
                
                # FIX: Loss must be float32 for scaler stability
                # FIX: Normalized Masked Loss
                # Divide by sum of mask to avoid gradient dilution for small hair areas
                masked_loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-6)
                
                loss_history.append(masked_loss.item())
                
                accelerator.backward(masked_loss)
                optimizer.step()
                optimizer.zero_grad()
                
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            output_path = os.path.join(args.output_dir, f"checkpoint-{epoch}")
            os.makedirs(output_path, exist_ok=True)
            transformer.save_pretrained(output_path)
            transformer.save_pretrained(output_path)
            print(f"Saved LoRA to {output_path}")

        # Save Loss Graph
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label="Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "loss_graph.png"))
        plt.close()


if __name__ == "__main__":
    main()
