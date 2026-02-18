
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from typing import Optional

class ReferenceAttentionControl:
    def __init__(self, pipeline, mode="write", do_classifier_free_guidance=False, attention_auto_machine_weight=1.0, style_fidelity=1.0):
        self.pipeline = pipeline
        self.mode = mode # "write" (cache ref) or "read" (inject ref)
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.attention_auto_machine_weight = attention_auto_machine_weight
        self.style_fidelity = style_fidelity
        self.reference_key_states = {}
        self.reference_value_states = {}
        
    def clear(self):
        self.reference_key_states = {}
        self.reference_value_states = {}

    def update(self, writer, reader):
        # Patch the model with our custom processors
        pass

class SD3ReferenceAttnProcessor(nn.Module):
    """
    Custom Attention Processor for Stable Diffusion 3 (JointAttnProcessor2_0).
    Intercepts the attention calculation to inject reference keys/values.
    """
    def __init__(self, original_processor, controller, layer_name):
        super().__init__()
        self.original_processor = original_processor
        self.controller = controller
        self.layer_name = layer_name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # SD3 Joint Attention handles both text (encoder_hidden_states) and image (hidden_states)
        # in a concatenated manner or separate streams depending on the block.
        # But for 'JointAttnProcessor2_0', hidden_states is usually the only input (it contains both if joined).
        
        # We need to rely on the original processor's logic for Q, K, V projection
        # because SD3 implementation is complex (RMSNorm, distinct QKV heads for text/image).
        
        # NOTE: This implementation assumes we can hijack the internal QKV or
        # reuse the original processor's __call__ but we need access to intermediate K, V.
        # Since diffusers doesn't expose K, V easily in __call__, we have to replicate some logic
        # OR use Hooks.
        
        # Simplified Strategy: 
        # 1. Calculate Q, K, V using attn module methods.
        # 2. If mode == "write": save K, V.
        # 3. If mode == "read": concat saved K, V with current K, V.
        # 4. Perform Attention.
        
        # Since SD3 is complex, let's try to stick to the standard logic structure.
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Get Query, Key, Value
        # Current Diffusers SD3 JointAttnProcessor logic:
        # query = attn.to_q(hidden_states)
        # key = attn.to_k(hidden_states)
        # value = attn.to_v(hidden_states)
        # (ignoring encoder_hidden_states separately for now as SD3 usually joins them before attn in MMDiT)
        
        query = attn.to_q(hidden_states)
        # FIX: We want to cache the IMAGE features (hidden_states), not the text features.
        # Even if encoder_hidden_states is present (JointBlock), the 'context' for visual reference is hidden_states.
        # SD3 'add_k_proj' handles the text part. 'to_k' handles the image part.
        # We focus on the image part for Reference Attention.
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # --- REFERENCE ATTENTION LOGIC ---
        if self.controller.mode == "write":
            # Cache the "Style" (Reference) keys/values
            self.controller.reference_key_states[self.layer_name] = key.detach().clone()
            self.controller.reference_value_states[self.layer_name] = value.detach().clone()
        
        elif self.controller.mode == "read":
            # Inject the Reference keys/values
            if self.layer_name in self.controller.reference_key_states:
                ref_k = self.controller.reference_key_states[self.layer_name]
                ref_v = self.controller.reference_value_states[self.layer_name]
                
                # Handle CFG (Classifier Free Guidance)
                # If generation batch_size is 2 (uncond, cond) but ref was 1, duplicate ref
                if key.shape[0] != ref_k.shape[0]:
                    ref_k = ref_k.repeat(key.shape[0], 1, 1, 1)
                    ref_v = ref_v.repeat(value.shape[0], 1, 1, 1)
                
                # Concatenate: [Current Key, Reference Key]
                # We need to concat along the sequence dimension (dim=2)
                key = torch.cat([key, ref_k], dim=2)
                value = torch.cat([value, ref_v], dim=2)
                
                # Note: We do NOT expand attention mask here because SD3 usually uses packed causal masks 
                # but for reference attention (style transfer), we largely ignore strict masking of reference.
        
        # --- ATTENTION CALCULATION ---
        # Scaled Dot-Product Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states

def register_reference_attention(pipeline, controller):
    """
    Traverses the pipeline's transformer and replaces attention processors
    with our custom SD3ReferenceAttnProcessor.
    """
    print("Registering Reference Attention Processors...")
    attn_procs = {}
    
    # SD3 Transformer has 'transformer_blocks'
    # Each block has 'attn' (JointAttn)
    
    # We iterate over named modules to find Attention layers
    for name, module in pipeline.transformer.named_modules():
        if isinstance(module, Attention):
            # We assume it's a self-attention layer we want to hook
            # SD3 BasicTransformerBlock -> attn
            if "attn" in name and "to_q" in dir(module): 
                # Create wrapper
                original_proc = module.processor
                new_proc = SD3ReferenceAttnProcessor(original_proc, controller, layer_name=name)
                module.set_processor(new_proc)

    print("Reference Attention Processors Registered.")
