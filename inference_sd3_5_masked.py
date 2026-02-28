
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SD3Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from peft import PeftModel, LoraConfig
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

def apply_affine_transform(image_pil, scale, x_offset, y_offset, target_size, interpolation=cv2.INTER_LINEAR):
    img = np.array(image_pil)
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    if new_w <= 0 or new_h <= 0:
        return Image.new(image_pil.mode, target_size)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    canvas_w, canvas_h = target_size
    if len(img.shape) == 3: # RGB
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    else: # Grayscale/L
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    start_x, start_y = int(x_offset), int(y_offset)
    src_x1, src_y1, src_x2, src_y2 = 0, 0, new_w, new_h
    dst_x1, dst_y1, dst_x2, dst_y2 = start_x, start_y, start_x + new_w, start_y + new_h
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
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img_resized[src_y1:src_y2, src_x1:src_x2]
    return Image.fromarray(canvas)

# Modify pos_embed to 32 channels (Same as Training)
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
    # Weights will be loaded from checkpoint, so initialization here doesn't matter much
    # but let's keep it consistent
    new_proj.weight.data[:, :16, :, :] = old_proj.weight.data
    new_proj.weight.data[:, 16:, :, :] = 0.0
    if old_proj.bias is not None:
        new_proj.bias.data = old_proj.bias.data
    transformer.pos_embed.proj = new_proj
    return transformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the checkpoint directory (must contain adapter_model.safetensors and pos_embed_weights.pt)")
    parser.add_argument("--image_path", type=str, required=True, help="Original image path")
    parser.add_argument("--mask_path", type=str, required=True, help="Mask image path")
    parser.add_argument("--sketch_path", type=str, required=True, help="Sketch image path")
    parser.add_argument("--prompt", type=str, default="A hairstyle")
    parser.add_argument("--output_path", type=str, default="output.png")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--bg_start_ratio", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print(f"Loading models from {args.checkpoint_dir}...")

    # 1. Load Base Models
    model_name = "stabilityai/stable-diffusion-3.5-medium"
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype).to(device)
    transformer = SD3Transformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
    
    # 2. Modify Architecture
    transformer = modify_pos_embed(transformer)

    # 3. Load Trained Weights
    # Load Input Layer (pos_embed)
    pos_embed_path = os.path.join(args.checkpoint_dir, "pos_embed_weights.pt")
    if os.path.exists(pos_embed_path):
        pos_state_dict = torch.load(pos_embed_path, map_location="cpu")
        try:
            transformer.pos_embed.load_state_dict(pos_state_dict)
        except RuntimeError as e:
            print(f"Error loading pos_embed: {e}")
            print("Attempting loose load...")
            transformer.pos_embed.load_state_dict(pos_state_dict, strict=False)
        print("Loaded pos_embed weights.")
    else:
        print(f"Error: pos_embed_weights.pt not found in {args.checkpoint_dir}")
        return

    # Load LoRA
    # Using PeftModel to load adapter
    transformer = PeftModel.from_pretrained(transformer, args.checkpoint_dir)
    print("Loaded LoRA weights.")
    
    transformer.to(device, dtype=dtype)
    transformer.eval()

    # 4. Prepare Inputs
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])

    image = Image.open(args.image_path).convert("RGB")
    target_size = image.size
    
    # Load and Transform Sketch/Matte
    raw_mask = Image.open(args.mask_path).convert("L")
    raw_sketch = Image.open(args.sketch_path).convert("RGB")
    
    print(f"Applying transforms: scale={args.scale}, x={args.x}, y={args.y}")
    # Use INTER_LINEAR for mask to preserve the soft alpha values from S2M-Net
    mask = apply_affine_transform(raw_mask, args.scale, args.x, args.y, target_size, interpolation=cv2.INTER_LINEAR)
    sketch = apply_affine_transform(raw_sketch, args.scale, args.x, args.y, target_size, interpolation=cv2.INTER_LINEAR)

    pixel_values = transform(image).unsqueeze(0).to(device, dtype=dtype)
    mask_tensor = mask_transform(mask).unsqueeze(0).to(device, dtype=dtype)
    sketch_tensor = transform(sketch).unsqueeze(0).to(device, dtype=dtype)

    # 5. Encode Latents
    with torch.no_grad():
        # Image Latents
        latents_clean = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        
        # Sketch Latents
        sketch_latents = vae.encode(sketch_tensor).latent_dist.sample() * vae.config.scaling_factor

    # 6. Prepare Conditions
    # Text Embeddings (Dummy for Unconditional/Structure focus)
    # Using larger batch size implies CFG, but we use Guidance Scale differently in FlowMatch
    # For simplicity, passing zeros for text (as in training)
    # Or should we generate real embeddings? 
    # Training verified: encoder_hidden_states = torch.zeros((bsz, 77, 4096))
    
    # We will use guidance scale > 1, so we need unconditional and conditional.
    # But since we trained with empty text embeddings, "conditional" here means "sketch conditioned"?
    # Actually, training used `torch.zeros` as text embedding unconditionally.
    # So "classifier-free guidance" on text is not possible unless we trained with text.
    # We trained with Sketch Condition always (except 15% dropout).
    # So we can do CFG on Sketch if we want.
    # Let's start with NO CFG (Standard Generation) first to match training behavior.
    
    encoder_hidden_states = torch.zeros((1, 77, 4096), device=device, dtype=dtype)
    pooled_projections = torch.zeros((1, 2048), device=device, dtype=dtype)

    # 7. Generation Loop (Clean Background Injection)
    # Initial Noise
    torch.manual_seed(args.seed)
    latents = torch.randn_like(latents_clean)
    
    scheduler.set_timesteps(args.num_inference_steps)
    
    # Prepare Soft Mask for blending
    mask_latents = F.interpolate(mask_tensor, size=latents.shape[-2:], mode="nearest")
    from torchvision.transforms import GaussianBlur
    blur = GaussianBlur(kernel_size=(61, 61), sigma=10.0)
    mask_latents_blurred = blur(mask_latents)

    print(f"Starting Inference (Guidance Scale: {args.guidance_scale}, BG Start Ratio: {args.bg_start_ratio})...")
    for i, t in enumerate(scheduler.timesteps):
        # Current Step Ratio (0.0 at first step, 1.0 at last)
        step_ratio = i / len(scheduler.timesteps)
        
        # 1. Clean Background Injection logic with Scheduling
        # If we haven't reached the bg_start_ratio, we keep background as NOISY (current latents)
        # to prevent identity prior from overriding the sketch too early.
        if step_ratio < args.bg_start_ratio:
            # Mask out the background injection - let the model see the noisy background
            latents_input = latents
        else:
            # Normal Background Injection for blending
            latents_input = (1.0 - mask_latents_blurred) * latents_clean + mask_latents_blurred * latents
        
        # 2. CFG: Prepare Conditional and Unconditional Inputs
        # If guidance_scale > 1, we do CFG on the sketch condition
        do_classifier_free_guidance = args.guidance_scale > 1.0
        
        if do_classifier_free_guidance:
            # Conditional: with sketch
            cond_latents = torch.cat([latents_input, sketch_latents], dim=1)
            # Unconditional: without sketch (zeroed)
            uncond_latents = torch.cat([latents_input, torch.zeros_like(sketch_latents)], dim=1)
            
            model_input = torch.cat([uncond_latents, cond_latents], dim=0)
            
            # Batch other conditions
            batch_encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
            batch_pooled_projections = torch.cat([pooled_projections, pooled_projections], dim=0)
            batch_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0).to(device)
        else:
            model_input = torch.cat([latents_input, sketch_latents], dim=1)
            batch_encoder_hidden_states = encoder_hidden_states
            batch_pooled_projections = pooled_projections
            batch_t = t.unsqueeze(0).to(device)

        # 3. Predict
        with torch.no_grad():
            output = transformer(
                hidden_states=model_input,
                timestep=batch_t,
                encoder_hidden_states=batch_encoder_hidden_states,
                pooled_projections=batch_pooled_projections,
                return_dict=False
            )[0]
            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = output.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = output

        # 4. Step
        # Scheduler Step usually expects 'latents' (current noisy state).
        # We should update 'latents' (the loop variable).
        # Important: We only care about the update in the MASKED region.
        # But scheduler updates the whole tensor.
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # 8. Final Decode
    # Final blending to ensure background is pixel-perfect
    latents_final = (1.0 - mask_latents_blurred) * latents_clean + mask_latents_blurred * latents
    latents_final = latents_final / vae.config.scaling_factor
    
    with torch.no_grad():
        image = vae.decode(latents_final, return_dict=False)[0]
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    
    # Save
    image = Image.fromarray((image[0] * 255).astype(np.uint8))
    image.save(args.output_path)
    print(f"Saved result to {args.output_path}")

if __name__ == "__main__":
    main()
