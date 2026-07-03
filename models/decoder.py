"""Per-identity decoder: reconstructs a full-resolution face from the shared
encoder's 8x8x512 feature map. Each branch (scene / one specific identity)
gets its own Decoder instance with independent weights -- only the encoder
is shared (see `autoencoder.py`).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .encoder import _upscale_block

_CHANNELS = [512, 256, 128, 64, 32, 16]  # enough stages for up to 256x output


class Decoder(nn.Module):
    """Input: (B, 512, 8, 8). Output: (B, 3, output_resolution, output_resolution)."""

    def __init__(self, output_resolution: int = 128):
        super().__init__()
        if output_resolution % 8 != 0:
            raise ValueError("output_resolution must be a multiple of 8")

        num_stages = int(math.log2(output_resolution // 8))
        if 8 * (2**num_stages) != output_resolution:
            raise ValueError("output_resolution must be 8 * a power of 2 (e.g. 128, 256)")
        if num_stages + 1 > len(_CHANNELS):
            raise ValueError(f"output_resolution {output_resolution} too large for defined channel schedule")

        blocks = []
        for i in range(num_stages):
            blocks.append(_upscale_block(_CHANNELS[i], _CHANNELS[i + 1]))
        self.upscale = nn.Sequential(*blocks)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(_CHANNELS[num_stages], 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upscale(x)
        return self.to_rgb(x)
