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
import kornia
import lpips

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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--category", type=str, required=True, choices=["unbraid", "braid"],
                       help="Dataset category to use ('unbraid' for Stage 1, 'braid' for Stage 2).")
    parser.add_argument("--lambda_shape", type=float, default=1.0, 
                       help="Weight for Shape Reconstruction Loss (Gaussian Volume).")
    parser.add_argument("--lambda_gradient", type=float, default=0.5, 
                       help="Weight for Multi-scale Gradient Loss (Thin edges).")
    parser.add_argument("--lambda_lpips", type=float, default=0.5, 
                       help="Weight for LPIPS Perceptual Loss (Texture).")
    parser.add_argument("--loss_space", type=str, default="pixel", choices=["latent", "pixel"],
                       help="Space to calculate Hybrid Loss: 'latent' (faster) or 'pixel' (better for braids). MUST be pixel for LPIPS.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint directory to resume from (must contain pos_embed_weights.pt and adapter weights).")
    
    args = parser.parse_args()
    
    # Create Output Directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log Stage
    if args.lambda_shape > 0 or args.lambda_gradient > 0 or args.lambda_lpips > 0:
        print(f"Running Specialization Stage: L_Shape {args.lambda_shape}, L_Grad {args.lambda_gradient}, L_LPIPS {args.lambda_lpips} in {args.loss_space} Space")
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

    class HybridLoss(torch.nn.Module):
        def __init__(self, mode="pixel", vae=None):
            super().__init__()
            self.mode = mode
            self.vae = vae
            
            from torchvision.transforms import GaussianBlur
            # 1. Mask Blurring for Background Blending
            self.mask_blur = GaussianBlur(kernel_size=(61, 61), sigma=10.0)
            
            # 2. Shape Blurring for L1 Shape Reconstruction (Low Frequency)
            # We use multiple scales to capture coarse and medium silhouettes
            self.shape_kernels = [11, 21, 31]
            
            # 3. LPIPS for Perceptual Texture Matching
            if self.mode == "pixel":
                print("Loading LPIPS VGG Model...")
                self.lpips_loss = lpips.LPIPS(net='vgg').eval()
                # Disable gradients for LPIPS network itself
                for param in self.lpips_loss.parameters():
                    param.requires_grad = False
            else:
                self.lpips_loss = None
                print("WARNING: Selected 'latent' loss space, LPIPS cannot be computed!")

        def get_multi_scale_shape_loss(self, pred, target, mask=None):
            loss = 0.0
            from torchvision.transforms.functional import gaussian_blur
            for k in self.shape_kernels:
                # Sigma is proportional to kernel size for consistent blur level
                sigma = k / 4.0
                shape_p = gaussian_blur(pred.float(), kernel_size=[k, k], sigma=[sigma, sigma])
                shape_t = gaussian_blur(target.float(), kernel_size=[k, k], sigma=[sigma, sigma])
                
                l1_diff = torch.abs(shape_p - shape_t)
                
                if mask is not None:
                    l1_diff = (l1_diff * mask.float()).sum() / (mask.float().sum() * l1_diff.shape[1] + 1e-6)
                else:
                    l1_diff = l1_diff.mean()
                loss += l1_diff
                
            return loss / len(self.shape_kernels)
            
        def get_multi_scale_gradient_loss(self, pred, target, mask=None):
            # Compute multi-scale Sobel gradients (High Frequency)
            loss = 0.0
            # Normalize gradients per kernel size to roughly same scale
            scales = [3, 5, 7] 
            for k in scales:
                # Kornia's SpatialGradient doesn't natively support multiple kernel sizes easily out of box in all versions.
                # Let's use a safer manual convolution approach for multi-scale sobel if needed, 
                # or just use F.conv2d directly.
                # Since kornia.filters.SpatialGradient defaults to 3x3, we can build custom kernels or just 
                # use kornia.filters.sobel for the default and calculate.
                
                # A robust way without relying on complex kornia multi-scale APIs:
                # Just use kornia.filters.sobel directly which applies 3x3.
                # To get "multi-scale" we can blur before applying the 3x3 sobel!
                # Scale 3: raw image (detail)
                # Scale 5: slightly blurred image (thicker structures)
                # Scale 7: more blurred image (overall boundaries)
                
                if k > 3:
                     # Simulate larger kernel by blurring first
                     from torchvision.transforms.functional import gaussian_blur
                     p = gaussian_blur(pred, kernel_size=[k, k], sigma=[(k-1)/4.0, (k-1)/4.0])
                     t = gaussian_blur(target, kernel_size=[k, k], sigma=[(k-1)/4.0, (k-1)/4.0])
                else:
                     p, t = pred, target
                     
                grad_p = kornia.filters.sobel(p)
                grad_t = kornia.filters.sobel(t)
                
                diff = torch.abs(grad_p - grad_t)
                
                if mask is not None:
                     diff = (diff * mask.float()).sum() / (mask.float().sum() * diff.shape[1] + 1e-6)
                else:
                     diff = diff.mean()
                     
                loss += diff
                
            return loss / len(scales)

        def forward(self, pred, target, mask=None, z_t=None, sigmas=None):
            loss_shape = torch.tensor(0.0, device=pred.device)
            loss_gradient = torch.tensor(0.0, device=pred.device)
            loss_lpips_val = torch.tensor(0.0, device=pred.device)

            if self.mode == "pixel":
                # Reconstruct z0: x_pred = z_t - sigma * v_pred
                z0_pred = z_t - sigmas * pred
                
                # Decode Prediction
                z0_pred_scaled = z0_pred / self.vae.config.scaling_factor
                z0_pred_scaled = z0_pred_scaled.to(torch.float16)
                pixel_pred = self.vae.decode(z0_pred_scaled, return_dict=False)[0]
                
                # Decode Target
                target_scaled = target / self.vae.config.scaling_factor
                target_scaled = target_scaled.to(torch.float16)
                pixel_gt = self.vae.decode(target_scaled, return_dict=False)[0]
                
                pixel_mask = F.interpolate(mask, size=pixel_pred.shape[-2:], mode="nearest") if mask is not None else None
                pixel_mask_f32 = pixel_mask.float() if pixel_mask is not None else None
                
                # 1. Shape Volume Extraction (Multi-scale Gaussian Loss)
                loss_shape = self.get_multi_scale_shape_loss(pixel_pred, pixel_gt, pixel_mask_f32)
                loss_shape = loss_shape.to(pred.dtype)
                
                # 2. Multi-scale Gradient Extraction (Sobel Loss)
                loss_gradient = self.get_multi_scale_gradient_loss(pixel_pred.float(), pixel_gt.float(), pixel_mask_f32)
                loss_gradient = loss_gradient.to(pred.dtype)

                # 3. LPIPS Loss (Texture)
                # LPIPS expects inputs in [-1, 1], our vae decoder outputs roughly [-1, 1] 
                # (SD VAE decodes standardized latents to [-1, 1] image space)
                if self.lpips_loss is not None:
                    l_lpips = self.lpips_loss(pixel_pred.float(), pixel_gt.float()) # returns [B, 1, 1, 1]
                    # We might want to weight by mask if possible. 
                    # LPIPS is spatial if spatial=True, but by default it returns average.
                    loss_lpips_val = l_lpips.mean().to(pred.dtype)

            else: # Latent Space (Fallback, LPIPS not supported)
                mask_f32 = F.interpolate(mask, size=pred.shape[-2:], mode="nearest").float() if mask is not None else None
                loss_shape = self.get_multi_scale_shape_loss(pred, target, mask_f32)
                loss_shape = loss_shape.to(pred.dtype)
                
                loss_gradient = self.get_multi_scale_gradient_loss(pred.float(), target.float(), mask_f32)
                loss_gradient = loss_gradient.to(pred.dtype)

            return loss_shape, loss_gradient, loss_lpips_val

    hybrid_criterion = HybridLoss(mode=args.loss_space, vae=vae).to(accelerator.device) # Pass vae and mode

    print(f"Start Training: {len(dataset)} images, {args.num_epochs} epochs")
    print(f"  - Category: {args.category if args.category else 'All'}")
    print(f"  - Lambda Shape: {args.lambda_shape}, Gradient: {args.lambda_gradient}, LPIPS: {args.lambda_lpips}")
    print(f"  - Loss Space: {args.loss_space}")
    loss_history = []

    initial_epoch = 0
    if args.resume_from_checkpoint:
        # Try to parse epoch from checkpoint name (e.g., "checkpoint-10")
        try:
            # Assuming format "checkpoint-{epoch}"
            ckpt_name = os.path.basename(args.resume_from_checkpoint)
            if "checkpoint-" in ckpt_name:
                initial_epoch = int(ckpt_name.split("-")[-1])
                print(f"Resumed from epoch {initial_epoch}")
        except ValueError:
            print("Could not parse epoch from checkpoint name, starting from 0")

    for epoch in range(initial_epoch, args.num_epochs): # Use args.num_epochs
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
                
                # Prepare Soft Mask
                mask_latents = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")
                mask_latents_blurred = hybrid_criterion.mask_blur(mask_latents) 
                
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
                mask_f32 = mask_latents_blurred.float()
                loss_mse = (loss_mse * mask_f32).sum() / (mask_f32.sum() * model_pred.shape[1] + 1e-6)
                
                # Hybrid Loss Computation
                loss_shape, loss_gradient, loss_lpips = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
                
                use_hybrid = args.lambda_shape > 0 or args.lambda_gradient > 0 or args.lambda_lpips > 0
                if use_hybrid:
                    if args.loss_space == "pixel":
                        loss_shape, loss_gradient, loss_lpips = hybrid_criterion(
                            pred=model_pred.float(), 
                            target=latents.float(), 
                            mask=batch["masks"].to(device, dtype=torch.float16),
                            z_t=noisy_latents.float(), 
                            sigmas=sigmas.float()
                        )
                    else: # Latent Space
                        loss_shape, loss_gradient, loss_lpips = hybrid_criterion(
                            pred=model_pred.float(), 
                            target=target_v.float(), 
                            mask=mask_latents 
                        )
                
                # --- Time-dependent Guidance (3-step Scheduling) ---
                w_shape = torch.full((bsz,), args.lambda_shape, device=device)
                w_grad = torch.full((bsz,), args.lambda_gradient, device=device)
                w_lpips = torch.full((bsz,), args.lambda_lpips, device=device)
                
                if use_hybrid:
                    # Step 1: Layout (t=700~1000) -> Boost Shape (x1.5)
                    mask_step1 = (timesteps >= 700) & (timesteps <= 1000)
                    w_shape = torch.where(mask_step1, w_shape * 1.5, w_shape)
                    
                    # Step 2: Structure (t=300~700) -> Boost Grad (x3.0)
                    mask_step2 = (timesteps >= 300) & (timesteps < 700)
                    w_grad = torch.where(mask_step2, w_grad * 3.0, w_grad)
                    
                    # --- Exponential Time-dependent LPIPS (from FlowMapSR) ---
                    # Formula: lambda = 5 * exp(-4 * s), where s is sigma (0 to 1)
                    # This naturally boosts LPIPS at low noise (t -> 0)
                    s_vals = sigmas.flatten().float()
                    w_lpips = args.lambda_lpips * 5.0 * torch.exp(-4.0 * s_vals)
                    
                # Calculate mean weights for the batch to formulate final scalar loss
                curr_w_shape = w_shape.mean().item()
                curr_w_grad = w_grad.mean().item()
                curr_w_lpips = w_lpips.mean().item()
                
                # Total Loss Formulation
                loss = loss_mse + \
                       (curr_w_shape * loss_shape) + \
                       (curr_w_grad * loss_gradient) + \
                       (curr_w_lpips * loss_lpips)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                
                # Debug Print for Zero Loss
                if step % 100 == 0:
                    print(f"Step {step}[t={timesteps[0].item()}]: Tot={loss.item():.4f}, MSE={loss_mse.item():.4f}, Shp={loss_shape.item():.4f}, Grd={loss_gradient.item():.4f}, LPI={loss_lpips.item():.4f}")
                    print(f"       Weights: Shape({curr_w_shape:.2f}), Grad({curr_w_grad:.2f}), LPIPS({curr_w_lpips:.2f})")

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f} (MSE: {loss_mse.item():.4f})")

        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            stage_name = 'stage2' if (args.lambda_shape > 0 or args.lambda_gradient > 0) else 'stage1'
            output_path = os.path.join(args.output_dir, f"{stage_name}_checkpoint-{epoch+1}") 
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
