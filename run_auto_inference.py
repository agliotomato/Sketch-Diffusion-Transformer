import torch
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    AutoPipelineForInpainting,
    SD3ControlNetModel
)
from diffusers.utils import load_image

# Attempt to import S2M
try:
    from s2m_wrapper import S2MModel
except ImportError:
    S2MModel = None
    print("Warning: S2M modules not found. Auto-matte generation will be disabled.")

def resize_for_sd3(image, base=64):
    w, h = image.size
    new_w = (w // base) * base
    new_h = (h // base) * base
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def main():
    parser = argparse.ArgumentParser(description="Clean Auto-Inference for SD3.5 with S2M")
    parser.add_argument("--prompt", type=str, default="A detailed photo of a hairstyle", help="Text prompt")
    parser.add_argument("--sketch_path", type=str, required=True, help="Path to input sketch")
    parser.add_argument("--image_path", type=str, required=True, help="Path to original image")
    parser.add_argument("--mask_path", type=str, default=None, help="Path to mask (matte). If None, uses S2M.")
    parser.add_argument("--output_path", type=str, default="output.png", help="Path to save result")
    
    # Model Paths
    parser.add_argument("--s2m_checkpoint", type=str, default="checkpoints/S2M/200_net_G.pth", help="Path to S2M checkpoint")
    parser.add_argument("--sd3_checkpoint", type=str, default="stabilityai/stable-diffusion-3.5-large", help="SD3 Model ID or Path")
    parser.add_argument("--controlnet_checkpoint", type=str, default="stabilityai/stable-diffusion-3.5-large-controlnet-canny", help="ControlNet ID or Path")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA directory (optional)")
    parser.add_argument("--lora_weight", type=float, default=1.0, help="LoRA weight")
    
    # Inference Params
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--control_strength", type=float, default=0.7)

    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f">>> Running on {device} ({dtype})")

    # ---------------------------------------------------------
    # 1. Prepare Inputs
    # ---------------------------------------------------------
    print(f"Loading inputs...")
    sketch_pil = load_image(args.sketch_path).convert("RGB")
    original_pil = load_image(args.image_path).convert("RGB")
    
    sketch_pil = resize_for_sd3(sketch_pil)
    original_pil = resize_for_sd3(original_pil)
    
    # ---------------------------------------------------------
    # 2. Get Mask (Matte)
    # ---------------------------------------------------------
    mask_pil = None
    if args.mask_path:
        print(f"Using provided mask: {args.mask_path}")
        mask_pil = load_image(args.mask_path).convert("RGB")
        mask_pil = resize_for_sd3(mask_pil)
    else:
        print("Mask not provided. Running S2M Auto-Matte...")
        if S2MModel is None:
            raise ImportError("Cannot run S2M: s2m_wrapper not available.")
            
        if not os.path.exists(args.s2m_checkpoint):
            raise FileNotFoundError(f"S2M Checkpoint not found at {args.s2m_checkpoint}")
            
        # Initialize S2M
        s2m = S2MModel(args.s2m_checkpoint, device=device)
        
        # Load sketch as grayscale for S2M
        sketch_cv = cv2.imread(args.sketch_path, 0)
        if sketch_cv is None:
            raise ValueError(f"Failed to read sketch for S2M: {args.sketch_path}")
            
        # Predict
        matte_np = s2m.predict_matte(sketch_cv)
        
        # Save matte
        matte_save_path = args.output_path.replace(".png", "_matte.png")
        cv2.imwrite(matte_save_path, matte_np)
        print(f"    Saved auto-matte to {matte_save_path}")
        
        # Convert to PIL
        mask_pil = Image.fromarray(matte_np).convert("RGB")
        mask_pil = resize_for_sd3(mask_pil)

    # ---------------------------------------------------------
    # 3. Load SD3 Pipeline
    # ---------------------------------------------------------
    print("Loading SD3 + ControlNet...")
    controlnet = SD3ControlNetModel.from_pretrained(
        args.controlnet_checkpoint,
        torch_dtype=dtype
    )
    
    # Use AutoPipeline to handle 'ControlNet + Inpainting' logic automatically
    pipe = AutoPipelineForInpainting.from_pretrained(
        args.sd3_checkpoint,
        controlnet=controlnet,
        torch_dtype=dtype
    )
    pipe.to(device)

    # Load LoRA if requested
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}...")
        try:
             # Try standard loading first
            pipe.load_lora_weights(args.lora_path, weight_name="adapter_model.safetensors", adapter_name="hair_style")
            pipe.set_adapters(["hair_style"], adapter_weights=[args.lora_weight])
            print("    LoRA loaded successfully.")
        except Exception as e:
            print(f"    Warning: Failed to load LoRA: {e}")

    # ---------------------------------------------------------
    # 4. Run Inference
    # ---------------------------------------------------------
    print("Running Inference...")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    
    result = pipe(
        prompt=args.prompt,
        image=original_pil,
        mask_image=mask_pil,
        control_image=sketch_pil,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.control_strength,
        generator=generator
    ).images[0]
    
    result.save(args.output_path)
    print(f"Done! Result saved to {args.output_path}")

if __name__ == "__main__":
    main()
