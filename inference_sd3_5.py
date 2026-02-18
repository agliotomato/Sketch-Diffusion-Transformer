
# Updated for S2M integration
import torch
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusion3ControlNetInpaintingPipeline,
    SD3ControlNetModel,
    StableDiffusion3ControlNetPipeline
)
from diffusers.utils import load_image
from pipeline_sd3_5_ref import ReferenceAttentionControl, register_reference_attention
try:
    from s2m_wrapper import S2MModel
except ImportError as e:
    print(f"Warning: Failed to import s2m_wrapper: {e}")
    S2MModel = None


# Helper to resize for SD3 (Nearest 64)
def resize_for_sd3(image, base=64):
    w, h = image.size
    new_w = (w // base) * base
    new_h = (h // base) * base
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def main():
    parser = argparse.ArgumentParser(description="SD3.5 Sketch-to-Hair Inference with Ref-Attention")
    parser.add_argument("--prompt", type=str, default="A detailed photo of a hairstyle", help="Text prompt (Default: 'A detailed photo of a hairstyle')")
    parser.add_argument("--sketch_path", type=str, required=True, help="Path to sketch input (input_1)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to original image")
    parser.add_argument("--mask_path", type=str, default=None, help="Path to mask image (matte). If None, will be predicted from sketch using S2M")
    parser.add_argument("--color_ref_path", type=str, default=None, help="Path to color reference (input_2). If None, uses original image.")
    parser.add_argument("--output_path", type=str, default="output_sd35_ref.png", help="Path to save result")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to trained LoRA directory (e.g. sd35_hair_lora)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Alias for --lora_path (for backward compatibility)")
    parser.add_argument("--lora_weight", type=float, default=1.0, help="LoRA weight scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control_strength", type=float, default=0.7, help="ControlNet strength")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading SD3.5 Models on {device}...")
    
    controlnet = SD3ControlNetModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-controlnet-canny",
        torch_dtype=dtype
    )
    
    try:
        pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            controlnet=controlnet,
            torch_dtype=dtype
        )
    except Exception as e:
        print(f"Fallback to Txt2Img Pipe: {e}")
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            controlnet=controlnet,
            torch_dtype=dtype
        )
    
    pipe.to(device)

    # Backward compatibility: Map checkpoint_dir to lora_path if needed
    if args.checkpoint_dir and not args.lora_path:
        args.lora_path = args.checkpoint_dir

    # --- LOAD LORA ---
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path} with weight {args.lora_weight}...")
        try:
            # "hair_style"라는 이름으로 어댑터 로드 (PEFT 포맷 명시)
            pipe.load_lora_weights(args.lora_path, weight_name="adapter_model.safetensors", adapter_name="hair_style")
            pipe.set_adapters(["hair_style"], adapter_weights=[args.lora_weight])
            print("Successfully loaded LoRA.")
        except Exception as e:
            print(f"Warning: Failed to load LoRA: {e}")

    # --- HOOK REGISTRATION ---
    # We initialize the controller
    controller = ReferenceAttentionControl(pipe, mode="write")
    register_reference_attention(pipe, controller)
    
    # --- LOAD INPUTS ---
    sketch = resize_for_sd3(load_image(args.sketch_path).convert("RGB"))
    image = resize_for_sd3(load_image(args.image_path).convert("RGB"))
    
    if args.mask_path:
        print(f"Loading mask from {args.mask_path}...")
        mask = resize_for_sd3(load_image(args.mask_path).convert("RGB"))
    else:
        print(">>> Mask path not provided. Attempting to generate matte using S2M model...")
        if S2MModel is None:
            raise ImportError("s2m_wrapper module not found. Cannot auto-generate matte.")
        
        # Path to S2M Checkpoint
        s2m_ckpt = os.path.join("SketchHairSalon", "checkpoints", "S2M", "200_net_G.pth")
        if not os.path.exists(s2m_ckpt):
             # Try checking current directory or global checkpoints
             s2m_ckpt = os.path.join("checkpoints", "S2M", "200_net_G.pth")
        
        s2m_predictor = S2MModel(s2m_ckpt)
        
        # S2M takes grayscale numpy array
        sketch_cv = cv2.imread(args.sketch_path, 0) # Load as grayscale
        if sketch_cv is None:
             raise ValueError(f"Could not load sketch for S2M from {args.sketch_path}")

        predicted_matte = s2m_predictor.predict_matte(sketch_cv)
        
        # Save predicted matte for debug/reference
        matte_save_path = args.output_path.replace(".png", "_matte.png")
        cv2.imwrite(matte_save_path, predicted_matte)
        print(f"    Predicted matte saved to {matte_save_path}")
        
        # Convert to PIL and resize for SD3
        mask = Image.fromarray(predicted_matte).convert("RGB")
        mask = resize_for_sd3(mask)
    


    if args.color_ref_path:
        ref_image = resize_for_sd3(load_image(args.color_ref_path).convert("RGB"))
    else:
        print(">>> No color_ref_path provided. Using original image as reference.")
        ref_image = image

    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(">>> 1. Reference Pass (Extracting Color/Texture Features)...")
    controller.mode = "write"
    controller.clear()
    
    # To extract features, we pass the Reference Image through the VAE and then the Transformer.
    # We force a single forward pass with the prompt.
    with torch.no_grad():
        # Encode Reference to Latents
        ref_tensor = pipe.image_processor.preprocess(ref_image).to(device, dtype=dtype)
        ref_latents = pipe.vae.encode(ref_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
        
        # Encode Prompt
        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
            prompt=args.prompt,
            prompt_2=args.prompt,
            prompt_3=args.prompt, device=device, do_classifier_free_guidance=False
        )

        # Dummy Timestep (Low noise -> t=0 or t=100?)
        # For style extraction, t=0 (clean) or small t is mostly used in implementations like weak control.
        # But SD3 is Flow Matching. t=0 is data, t=1 is noise.
        # We pass t=0.0 (Data) to see clean features.
        timestep = torch.tensor([0.0], device=device, dtype=dtype)
        
        # Forward Pass (Transformer only)
        # We ignore ControlNet for the reference pass (it's color only).
        pipe.transformer(
            hidden_states=ref_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False
        )
    
    print(f"    Cached Reference Keys: {len(controller.reference_key_states)} layers.")
    
    print(">>> 2. Generation Pass (Injecting Features + Canny Control)...")
    controller.mode = "read"
    
    output = pipe(
        prompt=args.prompt,
        image=image,
        mask_image=mask,
        control_image=sketch,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.control_strength,
        generator=generator
    ).images[0]

    output.save(args.output_path)
    print(f"Result saved to {args.output_path}")

if __name__ == "__main__":
    main()
