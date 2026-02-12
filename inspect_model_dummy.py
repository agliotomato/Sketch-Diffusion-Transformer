from diffusers import SD3Transformer2DModel, SD3Transformer2DModel
import torch

print("Creating dummy SD3Transformer2DModel...")
# params for a small valid model to inspect structure
model = SD3Transformer2DModel(
    sample_size=64,
    patch_size=2,
    in_channels=16,
    num_layers=2,
    attention_head_dim=8,
    num_attention_heads=4,
    caption_projection_dim=32,
    joint_attention_dim=32,
    pooled_projection_dim=32,
    out_channels=16
)

print("\nTop level children:")
for name, module in model.named_children():
    print(f"- {name}: {type(module).__name__}")

print("\nLooking for Linear layers (potential LoRA targets):")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # Print only the last part of names to spot patterns like 'to_k', 'q_proj' etc.
        if "attn" in name or "attention" in name or "block" in name:
             print(f"  {name} ({module.in_features} -> {module.out_features})")
             # Print just the first few to avoid spam
             if "block.1" in name: break

