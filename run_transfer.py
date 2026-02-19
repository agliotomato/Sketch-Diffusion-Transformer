import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from diffusers import SD3Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from peft import PeftModel

def apply_affine_transform(image_pil, scale, x_offset, y_offset, target_size, interpolation=cv2.INTER_LINEAR):
    """
    Applies scaling and translation to an image, placing it on a canvas of target_size.
    image_pil: Source image (PIL)
    scale: Float scale factor
    x_offset, y_offset: Translation in pixels
    target_size: (width, height) tuple
    """
    # Convert to numpy for cv2
    img = np.array(image_pil)
    
    # Get original dimensions
    h, w = img.shape[:2]
    
    # 1. Scale
    new_w = int(w * scale)
    new_h = int(h * scale)
    if new_w <= 0 or new_h <= 0:
        # Handle too small scale
        return Image.new(image_pil.mode, target_size)
        
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # 2. Translate (Place on Canvas)
    # Canvas is target_size
    canvas_w, canvas_h = target_size
    
    # Create blank canvas
    if len(img.shape) == 3: # RGB
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    else: # Grayscale/L
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        
    # Calculate placement coordinates
    # We assume x, y are offsets from top-left (0,0) or center?
    # Let's assume (0,0) is top-left of canvas, and x,y is where top-left of image goes.
    # To make it user friendly, maybe center-to-center is better?
    # Let's stick to top-left alias for now, or simple overlay.
    # User said "move it", so x, y offset from (0,0).
    
    # Startup pos
    start_x = int(x_offset)
    start_y = int(y_offset)
    
    # Clipping logic to paste img_resized into canvas
    # Source coords
    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = new_w, new_h
    
    # Dest coords
    dst_x1 = start_x
    dst_y1 = start_y
    dst_x2 = start_x + new_w
    dst_y2 = start_y + new_h
    
    # Clip
    if dst_x1 < 0:
        src_x1 -= dst_x1
        dst_x1 = 0
    if dst_y1 < 0:
        src_y1 -= dst_y1
        dst_y1 = 0
    if dst_x2 > canvas_w:
        src_x2 -= (dst_x2 - canvas_w)
        dst_x2 = canvas_w
    if dst_y2 > canvas_h:
        src_y2 -= (dst_y2 - canvas_h)
        dst_y2 = canvas_h

    # Check validity
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img_resized[src_y1:src_y2, src_x1:src_x2]
        
    return Image.fromarray(canvas)

# ... (Previous modify_pos_embed function from inference script) ...
def modify_pos_embed(transformer):
    old_proj = transformer.pos_embed.proj
    new_proj = nn.Conv2d(
        in_channels=32,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None
    )
    new_proj.weight.data[:, :16, :, :] = old_proj.weight.data
    new_proj.weight.data[:, 16:, :, :] = 0.0
    if old_proj.bias is not None:
        new_proj.bias.data = old_proj.bias.data
    transformer.pos_embed.proj = new_proj
    return transformer

def main():
    parser = argparse.ArgumentParser(description="Transfer Sketch/Matte to Target Image")
    parser.add_argument("--target", type=str, required=True, help="Target background image path")
    parser.add_argument("--sketch", type=str, required=True, help="Source sketch path")
    parser.add_argument("--matte", type=str, required=True, help="Source matte path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory")
    
    # Transform args
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for sketch/matte")
    parser.add_argument("--x", type=int, default=0, help="X offset in pixels")
    parser.add_argument("--y", type=int, default=0, help="Y offset in pixels")
    
    # Modes
    parser.add_argument("--check", action="store_true", help="Alignment check mode (no inference)")
    parser.add_argument("--output", type=str, default="output_transfer.png", help="Output filename")
    
    # Inference args
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--debug", action="store_true", help="Save debug images (mask, inputs)") # Added debug flag

    args = parser.parse_args()
    
    # 1. Load Images
    target_img = Image.open(args.target).convert("RGB")
    target_size = target_img.size # (W, H)
    
    sketch_img = Image.open(args.sketch).convert("RGB")
    matte_img = Image.open(args.matte).convert("L")
    
    # 3. Apply Transform to Sketch & Matte
    print(f"Applying transform: Scale={args.scale}, X={args.x}, Y={args.y}")
    # Use Area/Linear for Sketch to preserve lines
    sketch_transformed = apply_affine_transform(sketch_img, args.scale, args.x, args.y, target_size, interpolation=cv2.INTER_AREA)
    # Use Nearest Neighbor for Matte to preserve sharp binary edges
    matte_transformed = apply_affine_transform(matte_img, args.scale, args.x, args.y, target_size, interpolation=cv2.INTER_NEAREST)
    
    # 3. Check Mode
    if args.check:
        print("Running Alignment Check...")
        # Create overlay visualization
        # Target + Red Overlay for Matte + Sketch lines
        
        target_np = np.array(target_img)
        matte_np = np.array(matte_transformed)
        sketch_np = np.array(sketch_transformed)
        
        # Red overlay for matte (where matte > 0)
        overlay = target_np.copy()
        mask_bool = matte_np > 127
        
        # Make red where mask is white
        overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([255, 0, 0]) * 0.5
        
        # Adaptive Sketch Detection
        sketch_gray = cv2.cvtColor(sketch_np, cv2.COLOR_RGB2GRAY)
        if np.mean(sketch_gray) < 127:
            # Black Background, White Lines
            edge_bool = sketch_gray > 100
        else:
            # White Background, Black Lines
            edge_bool = sketch_gray < 200
        
        overlay[edge_bool] = np.array([0, 255, 0]) # Green lines
        
        vis_img = Image.fromarray(overlay.astype(np.uint8))
        save_path = args.output
        vis_img.save(save_path)
        print(f"Saved alignment check to {save_path}")
        print("Matte is RED area, Sketch is GREEN lines.")
        return

    # 4. Inference Mode
    if not args.checkpoint:
        print("Error: --checkpoint is required for inference mode.")
        return
        
    print(f"Loading models from {args.checkpoint}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    # Load Models (Similar to inference_sd3_5_masked.py)
    model_name = "stabilityai/stable-diffusion-3.5-medium"
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype).to(device)
    transformer = SD3Transformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
    
    # Modify & Load Weights
    transformer = modify_pos_embed(transformer)
    
    pos_embed_path = os.path.join(args.checkpoint, "pos_embed_weights.pt")
    if os.path.exists(pos_embed_path):
        transformer.pos_embed.load_state_dict(torch.load(pos_embed_path, map_location="cpu"), strict=False)
    else:
        print("Warning: pos_embed_weights.pt not found!")
        
    transformer = PeftModel.from_pretrained(transformer, args.checkpoint)
    transformer.to(device, dtype=dtype)
    transformer.eval()
    
    # Prepare Inputs
    # Resize everything to 1024x1024 for model (if target is not 1024)
    # But usually we want to keep target resolution or crop?
    # SD3.5 is trained on 1024. Let's resize inputs to 1024.
    
    
    process_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    mask_process_transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    image_tensor = process_transform(target_img).unsqueeze(0).to(device, dtype=dtype)
    sketch_tensor = process_transform(sketch_transformed).unsqueeze(0).to(device, dtype=dtype)
    mask_tensor = mask_process_transform(matte_transformed).unsqueeze(0).to(device, dtype=dtype)
    
    # Encode
    with torch.no_grad():
        latents_clean = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor
        sketch_latents = vae.encode(sketch_tensor).latent_dist.sample() * vae.config.scaling_factor
        
        # Null Condition for CFG (Zero Sketch)
        null_sketch_latents = torch.zeros_like(sketch_latents)
        
    encoder_hidden_states = torch.zeros((1, 77, 4096), device=device, dtype=dtype)
    pooled_projections = torch.zeros((1, 2048), device=device, dtype=dtype)
    
    # For CFG, we need batch size 2 for encoder_hidden_states if we want to batch them?
    # Or just run twice. Batching is usually faster.
    # Let's batch [Uncond, Cond]
    
    encoder_hidden_states_batch = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
    pooled_projections_batch = torch.cat([pooled_projections, pooled_projections], dim=0)
    
    # Soft Mask
    mask_latents = F.interpolate(mask_tensor, size=latents_clean.shape[-2:], mode="nearest")
    from torchvision.transforms import GaussianBlur
    blur = GaussianBlur(kernel_size=(61, 61), sigma=10.0)
    mask_latents_blurred = blur(mask_latents)
    
    if args.debug:
        # Save blurred mask check
        save_mask = mask_latents_blurred[0, 0].cpu().float().numpy()
        save_mask = (save_mask * 255).astype(np.uint8)
        Image.fromarray(save_mask).save("debug_soft_mask.png")
        print("Saved debug_soft_mask.png")
    
    # Generation Loop
    scheduler.set_timesteps(args.steps)
    latents = torch.randn_like(latents_clean)
    
    print(f"Starting Inference with Guidance={args.guidance}...")
    
    for i, t in enumerate(scheduler.timesteps):
        # Background Injection (The User's specific question!)
        # (1 - M) * Clean + M * Noisy
        latents_input = (1.0 - mask_latents_blurred) * latents_clean + mask_latents_blurred * latents
        
        if args.debug and i == 0:
            # Visualize the first input to see "Matte Application"
            # Decode latents_input
            with torch.no_grad():
                debug_latents = latents_input / vae.config.scaling_factor
                debug_img = vae.decode(debug_latents, return_dict=False)[0]
                debug_img = (debug_img / 2 + 0.5).clamp(0, 1)
                debug_img = debug_img.cpu().permute(0, 2, 3, 1).float().numpy()
                Image.fromarray((debug_img[0] * 255).astype(np.uint8)).save("debug_step0_input.png")
                print("Saved debug_step0_input.png (clean bg + noisy hair hole)")

        # Prepare Batch for CFG
        # Batch: [Uncond (Null Sketch), Cond (Sketch)]
        latents_input_batch = torch.cat([latents_input, latents_input], dim=0)
        sketch_latents_batch = torch.cat([null_sketch_latents, sketch_latents], dim=0)
        
        model_input = torch.cat([latents_input_batch, sketch_latents_batch], dim=1)
        
        # Timestep
        t_batch = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0).to(device)
        
        with torch.no_grad():
            noise_pred_batch = transformer(
                hidden_states=model_input,
                timestep=t_batch,
                encoder_hidden_states=encoder_hidden_states_batch,
                pooled_projections=pooled_projections_batch,
                return_dict=False
            )[0]
            
        # Split predictions
        noise_pred_uncond, noise_pred_cond = noise_pred_batch.chunk(2)
        
        # Apply CFG
        noise_pred = noise_pred_uncond + args.guidance * (noise_pred_cond - noise_pred_uncond)
            
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        
    # Final Decode
    latents_final = (1.0 - mask_latents_blurred) * latents_clean + mask_latents_blurred * latents
    latents_final = latents_final / vae.config.scaling_factor
    
    with torch.no_grad():
        image = vae.decode(latents_final, return_dict=False)[0]
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    out_img = Image.fromarray((image[0] * 255).astype(np.uint8))
    out_img.save(args.output)
    print(f"Saved result to {args.output}")

if __name__ == "__main__":
    main()
