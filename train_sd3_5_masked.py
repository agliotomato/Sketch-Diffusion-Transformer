import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import SD3Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# Set backend to Agg for headless environments
plt.switch_backend('Agg')

from dataset_sd35 import HairInpaintingDataset

def modify_pos_embed(transformer):
    """
    Expands the input channels of the transformer's pos_embed from 16 to 32.
    Initializes the new 16 channels with zeros (Weight Surgery).
    """
    # Channel 0-15: Noisy Latents
    # Channel 16-31: Sketch Latents
    
    old_proj = transformer.pos_embed.proj
    
    # Create new Conv2d with 32 input channels
    new_proj = nn.Conv2d(
        in_channels=32,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None
    )
    
    # Initialize weights
    # 1. Copy original weights to first 16 channels (Preserve Pre-trained Knowledge)
    new_proj.weight.data[:, :16, :, :] = old_proj.weight.data
    
    # 2. Zero-init new 16 channels (Weight Surgery for gradual adaptation)
    new_proj.weight.data[:, 16:, :, :] = 0.0
    
    # Copy bias if exists
    if old_proj.bias is not None:
        new_proj.bias.data = old_proj.bias.data
        
    # Replace the layer in transformer
    transformer.pos_embed.proj = new_proj
    
    print("Successfully modified pos_embed: 16 -> 32 channels (New channels 0-initialized).")
    return transformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sd35_sketch_hair_lora")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--category", type=str, default=None, 
                       help="Dataset category to use (e.g., 'braid', 'unbraid'). Default: None (Load both if available)")
    parser.add_argument("--lambda_shape", type=float, default=0.0, 
                       help="Weight for Shape Reconstruction Loss (Gradient Loss).")
    parser.add_argument("--loss_space", type=str, default="latent", choices=["latent", "pixel"],
                       help="Space to calculate Gradient Loss: 'latent' (faster) or 'pixel' (better for braids).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint directory to resume from (must contain pos_embed_weights.pt and adapter weights).")
    
    args = parser.parse_args()
    
    # Create Output Directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log Stage
    if args.lambda_shape > 0:
        print(f"Running Specialization Stage: Lambda Shape {args.lambda_shape} in {args.loss_space} Space")
    else:
        print("Running Generalization Stage: lambda_shape = 0")

    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device

    # 1. Load Models
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae", torch_dtype=torch.float16)
    transformer = SD3Transformer2DModel.from_pretrained(args.model_name, subfolder="transformer", torch_dtype=torch.float16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    # 2. Modify Architecture (Weight Surgery)
    # This must be done BEFORE adding LoRA
    transformer = modify_pos_embed(transformer)

    # Freeze Base Model
    vae.requires_grad_(False)
    transformer.requires_grad_(False) # Default freeze all

    # 3. Unfreeze pos_embed to allow learning the new sketch condition
    transformer.pos_embed.proj.requires_grad_(True)

    # 4. Add LoRA
    # Rank 128 for detailed hair texture
    lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "add_k_proj", "add_v_proj", "add_q_proj", "to_add_out"
        ],
        # layers_to_transform=[i for i in range(24)], # Removed to allow suffix matching to work
        lora_dropout=0.05,
    )
    transformer = get_peft_model(transformer, lora_config) # Use get_peft_model
    transformer.enable_gradient_checkpointing() # Optimize VRAM

    # Resume from Checkpoint
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        
        # 1. Load LoRA Weights
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        
        adapter_path = os.path.join(args.resume_from_checkpoint, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            adapters_weights = load_file(adapter_path)
            set_peft_model_state_dict(transformer, adapters_weights)
            print(f"Successfully loaded LoRA weights from {adapter_path}")
        else:
            print(f"Warning: adapter_model.safetensors not found in {args.resume_from_checkpoint}")
            
        # 2. Load pos_embed Weights
        pos_embed_path = os.path.join(args.resume_from_checkpoint, "pos_embed_weights.pt")
        if os.path.exists(pos_embed_path):
            pos_state_dict = torch.load(pos_embed_path, map_location="cpu")
            # Handle PeftModel wrapping
            # PeftModel -> base_model -> model -> pos_embed (usually)
            # Or simplified access if attributes forwarded
            try:
                # Try direct loading (if forwarded)
                transformer.pos_embed.load_state_dict(pos_state_dict)
            except AttributeError:
                # Fallback to internal structure
                # This handles the case where simple forwarding doesn't work for modules
                transformer.base_model.model.pos_embed.load_state_dict(pos_state_dict)
                
            print(f"Successfully loaded pos_embed weights from {pos_embed_path}")
        else:
            print(f"Warning: pos_embed_weights.pt not found in {args.resume_from_checkpoint}")

    # Cast Transformer to fp16
    # Note: accelerate handles mixed precision, but explicit cast helps with some layers if not fully handled
    transformer.to(device, dtype=torch.float16)

    # Ensure trainable params are fp32 for stability
    for param in transformer.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # Optimizer (8-bit AdamW)
    import bitsandbytes as bnb
    params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = bnb.optim.AdamW8bit(params, lr=args.learning_rate) # Use args.learning_rate

    # Datasets
    dataset = HairInpaintingDataset(args.data_root, size=1024, category=args.category) # Added category
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Prepare with Accelerator
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    vae.to(device, dtype=torch.float16)

    class GradientLoss(torch.nn.Module):
        def __init__(self, mode="latent", vae=None):
            super().__init__()
            self.mode = mode
            self.vae = vae
            
            # Sobel kernel
            kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.register_buffer("kernel_x", kernel_x)
            self.register_buffer("kernel_y", kernel_y)
            
            # Gaussian Kernel for mask blurring (Sigma 10 approx)
            # Create a large kernel size to support sigma 10 (size ~ 6*sigma)
            from torchvision.transforms import GaussianBlur
            self.blur = GaussianBlur(kernel_size=(61, 61), sigma=10.0)

        def get_gradients(self, img):
            b, c, h, w = img.shape
            img_reshaped = img.view(b * c, 1, h, w)
            
            grad_x = torch.nn.functional.conv2d(img_reshaped, self.kernel_x, padding=1)
            grad_y = torch.nn.functional.conv2d(img_reshaped, self.kernel_y, padding=1)
            
            grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            return grad.view(b, c, h, w)

        def forward(self, pred, target, mask=None, z_t=None, sigmas=None):
            # Input Handling
            if self.mode == "pixel":
                # Reconstruct z0: x_pred = z_t - sigma * v_pred
                # pred is v_pred
                z0_pred = z_t - sigmas * pred
                # Target z0 is implicitly 'target' which was passed as 'latents' (original) in this branch?
                # Wait, caller passes 'target' as 'noise - latents' (velocity) usually.
                # If we want Pixel Loss, we need the GT Image or GT Latent.
                # Let's adjust call signature in training loop or reconstruct here.
                
                # We need to decode z0_pred.
                # VAE Decode requires float32 sometimes or good precision.
                # Provide VAE in eval mode.
                
                # Decode Prediction
                z0_pred_scaled = z0_pred / self.vae.config.scaling_factor
                # Ensure dtype matches VAE (float16)
                z0_pred_scaled = z0_pred_scaled.to(self.vae.dtype)
                pixel_pred = self.vae.decode(z0_pred_scaled, return_dict=False)[0]
                
                # Decode Target (We assume target passed here is the GT Latent for Pixel Mode flexibility, 
                # OR we reconstruct GT from inputs. 
                # Actually, easier to pass GT Latents as 'target' if mode is pixel from the loop.
                # Let's handle 'target' argument polymorphism in the loop.)
                target_scaled = target / self.vae.config.scaling_factor
                # Ensure dtype matches VAE (float16)
                target_scaled = target_scaled.to(self.vae.dtype)
                pixel_gt = self.vae.decode(target_scaled, return_dict=False)[0]
                
                # Calculate Gradients in Pixel Space
                grad_pred = self.get_gradients(pixel_pred.float()) # Compute grad in float32 for stability
                grad_gt = self.get_gradients(pixel_gt.float())
                
                # Resize Mask to Pixel Space
                if mask is not None:
                    mask = F.interpolate(mask, size=pixel_pred.shape[-2:], mode="nearest")
                    # Apply Gaussian Blur to Mask here if not already done?
                    # User requested Soft Masking. We can do it here or earlier.
                    # Let's apply it here to ensure it affects the loss weights.
                    mask = self.blur(mask)
            
            else: # Latent Space
                grad_pred = self.get_gradients(pred)
                grad_gt = self.get_gradients(target)
                
                if mask is not None:
                    mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
                    mask = self.blur(mask)

            loss = torch.abs(grad_pred - grad_gt)
            
            if mask is not None:
                loss = (loss * mask).sum() / (mask.sum() * loss.shape[1] + 1e-6)
            else:
                loss = loss.mean()
                
            return loss

    gradient_criterion = GradientLoss(mode=args.loss_space, vae=vae).to(accelerator.device) # Pass vae and mode

    print(f"Start Training: {len(dataset)} images, {args.num_epochs} epochs") # Use args.num_epochs
    print(f"  - Category: {args.category if args.category else 'All'}")
    print(f"  - Lambda Shape: {args.lambda_shape}")
    print(f"  - Loss Space: {args.loss_space}")
    loss_history = []

    for epoch in range(args.num_epochs): # Use args.num_epochs
        transformer.train()
        epoch_loss = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16) # [B, 3, 1024, 1024]
                masks = batch["masks"].to(accelerator.device, dtype=torch.float16) # [B, 1, 1024, 1024]
                sketches = batch["conditioning_pixel_values"].to(accelerator.device, dtype=torch.float16) # [B, 3, 1024, 1024]

                # ... (Keep existing latent encoding code) ...
                # VAE Encoding (Latents)
                with torch.no_grad():
                    # 1. Image Latents [B, 16, 128, 128]
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    
                    # 2. Sketch Latents [B, 16, 128, 128]
                    sketch_latents = vae.encode(sketches).latent_dist.sample() * vae.config.scaling_factor

                # Noise & Timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise (Forward Diffusion)
                # Manual Rectified Flow Noise Addition
                sigmas = timesteps.float() / scheduler.config.num_train_timesteps
                sigmas = sigmas.reshape(bsz, 1, 1, 1).to(device, dtype=torch.float16)
                
                # Noisy Target (Input Channel 0-15)
                # Apply noise ONLY to the masked region (Hair) with SOFT MASK
                # This prevents hard edge artifacts and allows seamless blending
                
                # 1. Prepare Soft Mask for Noise Injection
                # Resize mask to latent size [128, 128]
                mask_latents = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")
                
                # Apply Gaussian Blur to Mask (Soft Masking)
                # We use the blur kernel from gradient_criterion
                mask_latents_blurred = gradient_criterion.blur(mask_latents) 
                
                # 2. Apply Noise with Soft Mask
                # Background (1-mask) stays clean (latents)
                # Foreground (mask) gets noisy
                noisy_latents_full = (1.0 - sigmas) * latents + sigmas * noise
                noisy_latents = (1.0 - mask_latents_blurred) * latents + mask_latents_blurred * noisy_latents_full
                
                # ... (Keep existing CFG code) ...
                # CFG: Drop sketch condition with prob.
                if random.random() < 0.15:
                    sketch_latents = torch.zeros_like(sketch_latents)

                # Concatenate Inputs (32 channels)
                model_input = torch.cat([noisy_latents, sketch_latents], dim=1)

                # Prediction (v_prediction or epsilon)
                # Dummy text embeddings (Unconditional training focus on structure/texture)
                encoder_hidden_states = torch.zeros((bsz, 77, 4096), device=device, dtype=torch.float16)
                pooled_projections = torch.zeros((bsz, 2048), device=device, dtype=torch.float16)

                model_pred = transformer(
                    hidden_states=model_input,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states, 
                    pooled_projections=pooled_projections,
                    return_dict=False
                )[0]

                # Ground Truth (Target)
                # Flow Matching: target = noise - latents (or similar, widely used in SD3)
                # But we use scheduler.get_velocity() or similar if available. 
                # For simplicity, let's assume `model_pred` tries to verify `noise` (if epsilon) or `v`
                # In SD3, usually target is calculated based on noise/original. 
                # Let's rely on Diffuser's simplistic assumption for now or calculate v manually:
                # v_t = alpha_t * noise - sigma_t * x_0 (approx)
                # Standard practice:
                
                # If scheduler is FlowMatchEulerDiscreteScheduler (SD3)
                # target = noise - latents
                target_v = noise - latents # Renamed to target_v
                
                # Soft Masking for Loss (Already calculated above)
                # We reuse 'mask_latents_blurred' for loss weighting as well

                # Calculate MSE Loss
                loss_mse = F.mse_loss(model_pred.float(), target_v.float(), reduction="none")
                loss_mse = (loss_mse * mask_latents_blurred).sum() / (mask_latents_blurred.sum() * model_pred.shape[1] + 1e-6)
                
                # Shape Reconstruction Loss (Gradient Loss)
                loss_shape = torch.tensor(0.0, device=accelerator.device)
                if args.lambda_shape > 0:
                    if args.loss_space == "pixel":
                        loss_shape = gradient_criterion(
                            pred=model_pred.float(), 
                            target=latents.float(), # GT Latents for Pixel comparison (decoded inside)
                            mask=batch["masks"].to(device, dtype=torch.float16), # High-res mask
                            z_t=noisy_latents.float(), 
                            sigmas=sigmas.float()
                        )
                    else: # Latent Space
                        loss_shape = gradient_criterion(
                            pred=model_pred.float(), 
                            target=target_v.float(), 
                            mask=mask_latents # Pass unblurred mask_latents to gradient_criterion, it will blur it
                        )
                
                # Total Loss
                loss = loss_mse + args.lambda_shape * loss_shape

                # Backprop
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                
                # Debug Print for Zero Loss
                if step % 100 == 0:
                    print(f"Step {step}: Total Loss={loss.item():.6f}, MSE={loss_mse.item():.6f}, Shape={loss_shape.item():.6f}")
                    print(f"       Mask Mean={mask_latents_blurred.mean().item():.6f}, Max={mask_latents_blurred.max().item():.6f}")
                    print(f"       Pred Mean={model_pred.mean().item():.6f}, Target Mean={target_v.mean().item():.6f}")

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f} (MSE: {loss_mse.item():.4f}, Shape: {loss_shape.item():.4f})") # Use args.num_epochs

        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            stage_name = 'stage2' if args.lambda_shape > 0 else 'stage1'
            output_path = os.path.join(args.output_dir, f"{stage_name}_checkpoint-{epoch+1}") # Added stage_name
            os.makedirs(output_path, exist_ok=True)
            
            # Save Transformer (LoRA Adapters)
            transformer.save_pretrained(output_path)
            
            # Save Modified pos_embed manually
            # Since PEFT save_pretrained only saves adapters, we need to save the modified input layer
            # so we can reload it properly for inference.
            # Save Modified pos_embed manually
            # Since PEFT save_pretrained only saves adapters, we need to save the modified input layer
            # so we can reload it properly for inference.
            # We must use proper unwrap method from Accelerator
            unwrapped_model = accelerator.unwrap_model(transformer)
            # If PeftModel, we might need to access .base_model.model?
            # PeftModel usually forwards unknown attrs. Let's try direct access first.
            if hasattr(unwrapped_model, 'pos_embed'):
                pos_embed_state = unwrapped_model.pos_embed.state_dict()
            else:
                # If unwrapped is PeftModel and pos_embed is not forwarded correctly (it should be though)
                # Try accessing base_model (PeftModel -> base_model -> model -> pos_embed)
                # But actually PeftModel usually has base_model attribute.
                # Let's hope direct access works or fall back.
                # Wait, error was AttributeError "SD3Transformer2DModel object has no attribute 'unwrap_model'".
                # This confirms transformer was indeed wrapper which forwarded to base.
                # So unwrapped_model via accelerator should be fine.
                pos_embed_state = unwrapped_model.pos_embed.state_dict()
            torch.save(pos_embed_state, os.path.join(output_path, "pos_embed_weights.pt"))
            
            print(f"Saved Checkpoint & pos_embed to {output_path}")

        # Plot Loss
        if accelerator.is_main_process:
            plt.figure(figsize=(10, 5))
            plt.plot(loss_history, label="Training Loss")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title(f"Training Loss ({args.loss_space} gradient)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(args.output_dir, "loss_graph.png"))
            plt.close()

if __name__ == "__main__":
    main()
