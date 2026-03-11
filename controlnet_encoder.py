"""
V1 Control Encoder for SD3.5 ControlNet.

Input : (B, 4, H, W)  = concat(sketch_rgb 3ch, soft_matte 1ch)
Output pyramid:
  F1: (B, 64,  H/2,  W/2)
  F2: (B, 128, H/4,  W/4)
  F3: (B, 256, H/8,  W/8)
  F4: (B, 256, H/16, W/16)

ControlProjection:
  (B, C, h, w) → upsample to token grid → Linear(C→hidden_dim) zero-init
  → (B, N_tokens, hidden_dim)  e.g. (B, 4096, 1536)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = min(32, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act   = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class ControlEncoder(nn.Module):
    """
    Single-branch CNN encoder.
    sketch + matte를 매번 resize해서 각 블록에 넣지 않고
    encoder pyramid로 multi-scale feature를 만든다 (controlnet.md 원칙).
    """
    def __init__(self, in_channels: int = 4):
        super().__init__()

        # Stem: (B,4,H,W) → (B,32,H,W)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )

        # Down1: stride-2, 32→64  → F1 (H/2, W/2)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            ResBlock(64),
        )

        # Down2: stride-2, 64→128 → F2 (H/4, W/4)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            ResBlock(128),
        )

        # Down3: stride-2, 128→256 → F3 (H/8, W/8)
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            ResBlock(256),
        )

        # Down4: stride-2, 256→256 → F4 (H/16, W/16)
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            ResBlock(256),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, 4, H, W)
        returns: (f1, f2, f3, f4)
        """
        x  = self.stem(x)
        f1 = self.down1(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        f4 = self.down4(f3)
        return f1, f2, f3, f4


class ControlProjection(nn.Module):
    """
    CNN feature map → DiT image token sequence.

    1. Bilinear interpolate to (token_h, token_w)
    2. Flatten spatial → (B, N_tokens, in_channels)
    3. Linear(in_channels → hidden_dim), zero-init
    """
    def __init__(self, in_channels: int, hidden_dim: int = 1536,
                 token_h: int = 64, token_w: int = 64):
        super().__init__()
        self.token_h = token_h
        self.token_w = token_w
        self.proj = nn.Linear(in_channels, hidden_dim)
        # zero-init: 학습 시작 시 control이 0이 되어 pretrained 동작 보존
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, h, w)
        returns: (B, token_h*token_w, hidden_dim)
        """
        x = F.interpolate(x, size=(self.token_h, self.token_w),
                          mode="bilinear", align_corners=False)
        # (B, C, token_h, token_w) → (B, N, C)
        x = x.flatten(2).permute(0, 2, 1)
        return self.proj(x)
