"""Shared encoder: maps an aligned 128x128 (or 256x256) face crop to a
compact latent code, then back up to a small shared feature map that both
the scene-decoder and identity-decoder branches take as input.

This is the classic "deepfakes"/faceswap-style architecture: one encoder is
shared across both branches, forcing it to learn a pose/expression
representation that is identity-agnostic; only the decoders are
branch-specific (see `decoder.py`, `autoencoder.py`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _downscale_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(0.1, inplace=True),
    )


def _upscale_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(2),
        nn.LeakyReLU(0.1, inplace=True),
    )


class Encoder(nn.Module):
    """Input: (B, 3, R, R) with R in {128, 256}. Output: (B, 512, R/16, R/16)."""

    def __init__(self, input_resolution: int = 128, latent_dim: int = 512):
        super().__init__()
        if input_resolution % 16 != 0:
            raise ValueError("input_resolution must be a multiple of 16")

        self.input_resolution = input_resolution
        self.bottleneck_spatial = input_resolution // 16  # 4 downscale stages

        self.downscale = nn.Sequential(
            _downscale_block(3, 128),
            _downscale_block(128, 256),
            _downscale_block(256, 512),
            _downscale_block(512, 1024),
        )

        flat_dim = 1024 * self.bottleneck_spatial * self.bottleneck_spatial
        self.to_latent = nn.Linear(flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 4 * 4 * 1024)
        self.upscale = _upscale_block(1024, 512)  # 4x4x1024 -> 8x8x512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downscale(x)
        x = torch.flatten(x, start_dim=1)
        x = self.to_latent(x)
        x = self.from_latent(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.upscale(x)
        return x
