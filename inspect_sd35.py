"""
SD3.5 Medium 아키텍처 조회 스크립트
controlnet.md에서 요구하는 표를 출력한다:

  Selected DiT block | Relative position | Hidden token grid size | Control level
"""

import torch
from diffusers import StableDiffusion3Pipeline

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
IMAGE_SIZE = 1024  # 학습 해상도


def main():
    print("Loading pipeline config (no weights)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )

    transformer = pipe.transformer
    vae = pipe.vae

    # ── 1. DiT block 수 ──────────────────────────────────────────
    # SD3.5 Medium은 MM-DiT blocks만 존재 (Single blocks는 Large 전용)
    n_mm = len(transformer.transformer_blocks)
    n_single = len(transformer.single_transformer_blocks) if hasattr(transformer, "single_transformer_blocks") else 0
    N = n_mm + n_single
    print(f"\n[DiT blocks]")
    print(f"  MM-DiT blocks       : {n_mm}")
    print(f"  Single-DiT blocks   : {n_single}")
    print(f"  Total N             : {N}")

    # ── 2. hidden dim ──────────────────────────────────────────
    cfg = transformer.config
    # image token hidden_dim = num_heads * head_dim (NOT joint_attention_dim which is for text)
    n_heads   = cfg.num_attention_heads
    head_dim  = cfg.attention_head_dim
    img_hidden_dim = n_heads * head_dim
    print(f"  num_attention_heads : {n_heads}")
    print(f"  attention_head_dim  : {head_dim}")
    print(f"  image hidden_dim    : {img_hidden_dim}  (= {n_heads} x {head_dim})")
    print(f"  joint_attention_dim : {cfg.joint_attention_dim}  (text context, NOT injection target)")
    print(f"  caption_projection_dim: {cfg.caption_projection_dim}")
    # verify against actual layer weight
    block0 = transformer.transformer_blocks[0]
    actual_q_dim = block0.attn.to_q.weight.shape[0]
    print(f"  [verify] block[0].attn.to_q output dim: {actual_q_dim}")

    # ── 3. VAE latent spatial size ────────────────────────────
    vae_scale = vae.config.scaling_factor
    vae_spatial_compression = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_h = IMAGE_SIZE // vae_spatial_compression
    latent_w = IMAGE_SIZE // vae_spatial_compression
    latent_c = vae.config.latent_channels
    print(f"\n[VAE]")
    print(f"  spatial compression : {vae_spatial_compression}x")
    print(f"  latent shape (1024 input): ({latent_c}, {latent_h}, {latent_w})")
    print(f"  scaling_factor      : {vae_scale}")

    # ── 4. patch_size & token grid ────────────────────────────
    patch_size = transformer.config.patch_size
    token_h = latent_h // patch_size
    token_w = latent_w // patch_size
    n_tokens = token_h * token_w
    print(f"\n[Transformer patchify]")
    print(f"  patch_size          : {patch_size}")
    print(f"  token grid          : {token_h} x {token_w}  ({n_tokens} tokens)")

    # ── 5. Selected block 표 ──────────────────────────────────
    # relative positions: ~25%, ~40%, ~55% (MM-DiT 기준)
    positions = [0.25, 0.40, 0.55]
    selected = [round(p * N) for p in positions]
    # clamp to valid range
    selected = [max(0, min(s, N - 1)) for s in selected]

    control_levels = ["F4 (coarse, matte/global shape)", "F3", "F3 or F2 (sketch structure)"]
    alphas         = [1.0, 0.7, 0.4]

    print(f"\n[Selected DiT blocks  (N={N})]")
    print(f"{'Block idx':<12} {'Rel pos':>10} {'Token grid':>14} {'Control level':<32} {'alpha'}")
    print("-" * 80)
    for i, (s, pos, lvl, alpha) in enumerate(zip(selected, positions, control_levels, alphas)):
        is_mm = s < n_mm
        block_type = "MM-DiT" if is_mm else "Single"
        print(f"block[{s:>2}]    {pos*100:>8.0f}%   {token_h}x{token_w}={n_tokens:>5}   {lvl:<32}  {alpha}")
        print(f"           (type={block_type}, local_idx={s if is_mm else s - n_mm})")

    # ── 6. Control encoder pyramid vs token grid 대응 ─────────
    print(f"\n[Control encoder pyramid]")
    pyramid = {
        "F1": (latent_h // 2,  latent_w // 2),
        "F2": (latent_h // 4,  latent_w // 4),
        "F3": (latent_h // 8,  latent_w // 8),
        "F4": (latent_h // 16, latent_w // 16),
    }
    for name, (h, w) in pyramid.items():
        print(f"  {name}: ({h}, {w})  →  needs projection to ({token_h}, {token_w}) token grid")

    print(f"\n  token grid size = {token_h}x{token_w} = {n_tokens} tokens  (hidden_dim={inner_dim})")
    print("\nDone. Copy the table above into controlnet.md before implementing V1.")


if __name__ == "__main__":
    main()
