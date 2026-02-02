
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

# Helper to resize for SD3 (Nearest 64)
def resize_for_sd3(image, base=64):
    w, h = image.size
    new_w = (w // base) * base
    new_h = (h // base) * base
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def main():
    parser = argparse.ArgumentParser(description="SD3.5 Sketch-to-Hair Inference with Ref-Attention")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--sketch_path", type=str, required=True, help="Path to sketch input (input_1)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to original image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to mask image (matte)")
    parser.add_argument("--color_ref_path", type=str, required=True, help="Path to color reference (input_2)")
    parser.add_argument("--output_path", type=str, default="output_sd35_ref.png", help="Path to save result")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control_strength", type=float, default=0.7, help="ControlNet strength")
    
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

    # --- HOOK REGISTRATION ---
    # We initialize the controller
    controller = ReferenceAttentionControl(pipe, mode="write")
    register_reference_attention(pipe, controller)
    
    # --- LOAD INPUTS ---
    sketch = resize_for_sd3(load_image(args.sketch_path).convert("RGB"))
    image = resize_for_sd3(load_image(args.image_path).convert("RGB"))
    mask = resize_for_sd3(load_image(args.mask_path).convert("RGB"))
    ref_image = resize_for_sd3(load_image(args.color_ref_path).convert("RGB"))

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
            prompt=args.prompt, device=device, do_classifier_free_guidance=False
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
        num_inference_steps=28,
        guidance_scale=7.0,
        controlnet_conditioning_scale=args.control_strength,
        generator=generator
    ).images[0]

    output.save(args.output_path)
    print(f"Result saved to {args.output_path}")

if __name__ == "__main__":
    main()
