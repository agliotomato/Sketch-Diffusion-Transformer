from diffusers import SD3Transformer2DModel
import torch
from peft import LoraConfig, get_peft_model

print("Creating dummy SD3Transformer2DModel...")
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

def test_config(name, target_modules, layers_to_transform=None):
    print(f"\n--- Testing Config: {name} ---")
    print(f"Targets: {target_modules}")
    print(f"Layers to transform: {layers_to_transform}")
    
    config = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        lora_dropout=0.0
    )
    
    try:
        peft_model = get_peft_model(model, config)
        print("SUCCESS!")
        # Check if parameters are actually trainable
        trainable_params = 0
        all_params = 0
        for _, param in peft_model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable params: {trainable_params} / {all_params}")
        if trainable_params == 0:
            print("WARNING: 0 trainable params. Config didn't match anything.")
        return True
    except ValueError as e:
        print(f"FAILED: {e}")
        return False

# Test Cases
# 1. Standard names, no layers_to_transform (suffix matching)
test_config("Standard Suffix", ["to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_v_proj", "add_q_proj", "to_add_out"])

# 2. Standard names WITH layers_to_transform (The original attempt)
test_config("Standard + Layers", ["to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_v_proj", "add_q_proj", "to_add_out"], layers_to_transform=[0, 1])

# 3. Nested names WITH layers_to_transform (The second attempt)
test_config("Nested + Layers", ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.add_k_proj", "attn.add_v_proj", "attn.add_q_proj", "attn.to_add_out"], layers_to_transform=[0, 1])

# 4. Correct names if diffusers uses 'transformer_blocks'
# Maybe 'transformer_blocks' is the key? But we usually don't need that.
