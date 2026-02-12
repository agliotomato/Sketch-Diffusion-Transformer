import torch
from diffusers import SD3Transformer2DModel

model_name = "stabilityai/stable-diffusion-3.5-large"
print(f"Loading {model_name} structure...")
try:
    transformer = SD3Transformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch.float16)
    print("Model loaded successfully.")
    print("Structure of SD3Transformer2DModel:")
    print(transformer)
    
    # Check for likely candidates
    if hasattr(transformer, 'x_embedder'):
        print("\nFound 'x_embedder'!")
    elif hasattr(transformer, 'pos_embed'):
        print("\nFound 'pos_embed' (might be related)")
    else:
        print("\n'x_embedder' not found. Listing top-level attributes:")
        for name, module in transformer.named_children():
            print(f"- {name}: {type(module)}")

except Exception as e:
    print(f"Error loading model: {e}")
