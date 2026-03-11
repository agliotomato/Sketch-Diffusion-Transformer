"""
V1 ControlNet training for SD3.5 Medium.

Policy (controlnet.md):
  - Control encoder + projections만 학습
  - SD3.5 transformer, VAE, text encoders freeze
  - Injection: block[6](F4,α=1.0), block[10](F3,α=0.7), block[13](F3,α=0.4)
  - Loss: diffusion (flow matching) loss + optional L1 + LPIPS auxiliary
  - Input: 4ch = sketch_rgb(3ch) + soft_matte(1ch)
  - Target: hair_target = image × matte (hair patch only)
  - Stage 1: --category unbraid  (pretraining on hair texture)
  - Stage 2: --category braid    (fine-tuning on braid topology)
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from controlnet_encoder import ControlEncoder, ControlProjection
from dataset import HairInpaintingDataset

try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


# ─────────────────────────────── constants ────────────────────────────────────
MODEL_ID      = "stabilityai/stable-diffusion-3.5-medium"
IMG_HIDDEN    = 1536        # image token hidden dim (24 heads × 64)
# 512 input: 512/8(VAE)/2(patch) = 32×32 token grid
TOKEN_H       = 32
TOKEN_W       = 32
INJECT_BLOCKS = [6, 10, 13] # ~25%, ~40%, ~55% of N=24
ALPHAS        = [1.0, 0.7, 0.4]
# F4(block6), F3(block10), F3(block13)
CTRL_CHANNELS = [256, 256, 256]


# ───────────────────────────── helpers ────────────────────────────────────────

def get_sigmas(timesteps, scheduler, n_dim: int, dtype):
    """
    Flow matching: get sigma values for given integer timestep indices.
    Returns tensor broadcastable to (B, 1, 1, 1) when n_dim=4.
    """
    sigmas = scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    # scheduler.sigmas has shape (num_train_timesteps+1,)
    # timesteps are in [0, num_train_timesteps-1]
    s = sigmas[timesteps]
    for _ in range(n_dim - 1):
        s = s.unsqueeze(-1)
    return s


def encode_null_prompt(pipe, device, dtype, batch_size: int = 1):
    """Encode empty prompt once; reuse for all training steps."""
    with torch.no_grad():
        (prompt_embeds, neg_embeds,
         pooled_embeds, neg_pooled) = pipe.encode_prompt(
            prompt="",
            prompt_2="",
            prompt_3="",
            device=device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=False,
        )
    return prompt_embeds.to(dtype), pooled_embeds.to(dtype)


def forward_with_control(
    transformer,
    projections,        # list of ControlProjection
    alphas,
    inject_blocks,
    ctrl_raw_feats,     # list of (B, C, h, w) raw CNN features
    hidden_states,      # (B, 16, 128, 128) latent
    encoder_hidden_states,
    pooled_projections,
    timestep,
):
    """
    Hook-based control injection.
    Projects each feature to token space, registers additive residual hooks,
    runs transformer forward, then removes hooks.
    """
    # Project raw CNN features → (B, 4096, 1536)
    ctrl_feats = [proj(feat) for proj, feat in zip(projections, ctrl_raw_feats)]

    hooks = []
    for block_idx, feat, alpha in zip(inject_blocks, ctrl_feats, alphas):
        def _make_hook(f, a):
            def _hook(module, input, output):
                # MM-DiT block returns (hidden_states, encoder_hidden_states)
                hs, enc_hs = output
                return (hs + a * f, enc_hs)
            return _hook
        h = transformer.transformer_blocks[block_idx].register_forward_hook(
            _make_hook(feat, alpha)
        )
        hooks.append(h)

    result = transformer(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        timestep=timestep,
        return_dict=True,
    )

    for h in hooks:
        h.remove()

    return result.sample  # (B, 16, 128, 128) predicted velocity


# ─────────────────────────────── main ────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    type=str, required=True)
    p.add_argument("--output_dir",   type=str, default="checkpoints/controlnet_v1")
    p.add_argument("--image_size",   type=int, default=1024)
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--grad_accum",   type=int, default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--max_steps",    type=int, default=10000)
    p.add_argument("--save_every",   type=int, default=500)
    p.add_argument("--log_every",    type=int, default=50)
    p.add_argument("--dtype",        type=str, default="fp16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--num_workers",  type=int, default=4)
    # 2-stage training
    p.add_argument("--category",     type=str, default=None,
                   choices=["unbraid", "braid"],
                   help="Stage 1: unbraid (pretraining), Stage 2: braid (fine-tuning)")
    # auxiliary pixel-space losses (L1 + LPIPS)
    p.add_argument("--lambda_l1",    type=float, default=0.0,
                   help="Weight for L1 pixel loss (decoded). 0 = disabled.")
    p.add_argument("--lambda_lpips", type=float, default=0.0,
                   help="Weight for LPIPS loss (decoded). 0 = disabled.")
    # fine-tuning: load pretrained encoder/projections
    p.add_argument("--resume_encoder",     type=str, default=None,
                   help="Path to encoder.pt to resume/fine-tune from.")
    p.add_argument("--resume_projections", type=str, default=None,
                   help="Path to projections.pt to resume/fine-tune from.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # ── Load pipeline ──────────────────────────────────────────
    print("Loading SD3.5 pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    )
    pipe.to(device)

    transformer = pipe.transformer
    vae         = pipe.vae
    scheduler   = pipe.scheduler

    # ── Freeze everything in the base model ───────────────────
    for p in transformer.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder_2.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder_3.parameters():
        p.requires_grad_(False)

    transformer.eval()
    vae.eval()

    # ── Build trainable modules ────────────────────────────────
    encoder = ControlEncoder(in_channels=4).to(device=device, dtype=dtype)

    # F4(256ch)→block6, F3(256ch)→block10, F3(256ch)→block13
    projections = torch.nn.ModuleList([
        ControlProjection(CTRL_CHANNELS[i], IMG_HIDDEN, TOKEN_H, TOKEN_W)
        for i in range(len(INJECT_BLOCKS))
    ]).to(device=device, dtype=dtype)

    # ── Resume / fine-tune ─────────────────────────────────────
    if args.resume_encoder:
        encoder.load_state_dict(torch.load(args.resume_encoder, map_location=device))
        print(f"Resumed encoder from {args.resume_encoder}")
    if args.resume_projections:
        projections.load_state_dict(torch.load(args.resume_projections, map_location=device))
        print(f"Resumed projections from {args.resume_projections}")

    trainable_params = list(encoder.parameters()) + list(projections.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)

    # ── Auxiliary losses (L1 + LPIPS) ─────────────────────────
    use_aux = (args.lambda_l1 > 0) or (args.lambda_lpips > 0)
    lpips_net = None
    if args.lambda_lpips > 0:
        if not _LPIPS_AVAILABLE:
            raise ImportError("pip install lpips to use --lambda_lpips")
        lpips_net = lpips_lib.LPIPS(net="vgg").eval().to(device)
        for p in lpips_net.parameters():
            p.requires_grad_(False)
        print("LPIPS VGG model loaded.")

    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen    = sum(p.numel() for p in transformer.parameters())
    print(f"Trainable params : {n_trainable / 1e6:.1f}M")
    print(f"Frozen (DiT)     : {n_frozen / 1e6:.1f}M")

    # ── Dataset ────────────────────────────────────────────────
    dataset = HairInpaintingDataset(
        data_root=args.data_root,
        size=args.image_size,
        mode="train",
        category=args.category,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    loader_iter = iter(loader)

    # ── Null text embeddings (fixed for V1) ───────────────────
    print("Encoding null prompt...")
    null_enc_hs, null_pooled = encode_null_prompt(pipe, device, dtype, args.batch_size)

    # ── Scheduler setup ───────────────────────────────────────
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)

    # ── Training loop ─────────────────────────────────────────
    encoder.train()
    projections.train()

    global_step  = 0
    accum_loss   = 0.0
    optimizer.zero_grad()

    print(f"Starting V1 training for {args.max_steps} steps...")
    while global_step < args.max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)   # (B,3,H,W)
        sketch       = batch["sketch"].to(device=device, dtype=dtype)          # (B,3,H,W)
        matte        = batch["matte"].to(device=device, dtype=dtype)           # (B,1,H,W)

        B = pixel_values.shape[0]

        # ── VAE encode ────────────────────────────────────────
        with torch.no_grad():
            z = vae.encode(pixel_values).latent_dist.sample()
            z = z * vae.config.scaling_factor  # (B, 16, 128, 128)

        # ── Flow matching noise ───────────────────────────────
        noise = torch.randn_like(z)
        t_idx = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (B,), device=device
        )
        sigmas = get_sigmas(t_idx, scheduler, n_dim=z.ndim, dtype=dtype)
        z_t    = (1.0 - sigmas) * z + sigmas * noise
        target = noise - z  # flow velocity

        # SD3 transformer expects timestep as the scheduler's actual timestep values
        # scheduler.timesteps is ordered high→low (inference order), so we reverse-index:
        # t_idx=0 → sigma=1.0 → highest noise → timestep near 1000
        # Use scheduler.timesteps directly indexed by t_idx
        timestep = scheduler.timesteps[t_idx]  # float tensor, values in scheduler range

        # ── Control encoder ───────────────────────────────────
        control_input = torch.cat([sketch, matte], dim=1)  # (B,4,H,W)
        f1, f2, f3, f4 = encoder(control_input)
        # block6←F4, block10←F3, block13←F3
        ctrl_raw = [f4, f3, f3]

        # ── Forward with control injection ────────────────────
        # Expand null embeddings to batch size
        enc_hs = null_enc_hs.expand(B, -1, -1)
        pooled = null_pooled.expand(B, -1)

        pred = forward_with_control(
            transformer=transformer,
            projections=projections,
            alphas=ALPHAS,
            inject_blocks=INJECT_BLOCKS,
            ctrl_raw_feats=ctrl_raw,
            hidden_states=z_t,
            encoder_hidden_states=enc_hs,
            pooled_projections=pooled,
            timestep=timestep,
        )

        # ── Loss ──────────────────────────────────────────────
        loss = F.mse_loss(pred.float(), target.float())

        # Auxiliary pixel-space losses (L1 + LPIPS)
        if use_aux:
            with torch.cuda.amp.autocast(enabled=False):
                sigmas_aux = get_sigmas(t_idx, scheduler, n_dim=z.ndim, dtype=torch.float32)
                z0_pred = (z_t.float() - sigmas_aux * pred.float())
                pixel_pred = vae.decode(
                    z0_pred / vae.config.scaling_factor, return_dict=False
                )[0].clamp(-1, 1)
                pixel_gt = vae.decode(
                    z.float() / vae.config.scaling_factor, return_dict=False
                )[0].clamp(-1, 1)

            if args.lambda_l1 > 0:
                loss = loss + args.lambda_l1 * F.l1_loss(pixel_pred, pixel_gt)
            if args.lambda_lpips > 0 and lpips_net is not None:
                loss = loss + args.lambda_lpips * lpips_net(pixel_pred, pixel_gt).mean()

        loss = loss / args.grad_accum
        loss.backward()
        accum_loss += loss.item()

        if (global_step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if global_step % args.log_every == 0:
                print(f"step {global_step:>6}  loss={accum_loss:.4f}")
            accum_loss = 0.0

        global_step += 1

        # ── Save checkpoint ───────────────────────────────────
        if global_step % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"step_{global_step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(encoder.state_dict(),
                       os.path.join(ckpt_dir, "encoder.pt"))
            torch.save(projections.state_dict(),
                       os.path.join(ckpt_dir, "projections.pt"))
            print(f"Saved checkpoint → {ckpt_dir}")

    # Final save
    torch.save(encoder.state_dict(),
               os.path.join(args.output_dir, "encoder_final.pt"))
    torch.save(projections.state_dict(),
               os.path.join(args.output_dir, "projections_final.pt"))
    print("Training done.")


if __name__ == "__main__":
    main()
