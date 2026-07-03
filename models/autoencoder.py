"""FaceSwapAutoencoder: one shared Encoder + per-branch Decoders.

Branches are named strings -- in this v1 tool there are exactly two:
"scene" (reconstructs scene/driving faces) and "identity" (reconstructs the
consented identity's face). Training alternates reconstruction loss between
the two branches through the *same* encoder; inference always does
encoder(scene_face) -> identity_decoder(...) to perform the actual swap.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder

SCENE_BRANCH = "scene"
IDENTITY_BRANCH = "identity"


class FaceSwapAutoencoder(nn.Module):
    def __init__(self, resolution: int = 128, latent_dim: int = 512):
        super().__init__()
        self.resolution = resolution
        self.encoder = Encoder(input_resolution=resolution, latent_dim=latent_dim)
        self.decoders = nn.ModuleDict(
            {
                SCENE_BRANCH: Decoder(output_resolution=resolution),
                IDENTITY_BRANCH: Decoder(output_resolution=resolution),
            }
        )

    def forward(self, x: torch.Tensor, branch: str) -> torch.Tensor:
        if branch not in self.decoders:
            raise ValueError(f"Unknown branch '{branch}'. Options: {list(self.decoders)}")
        features = self.encoder(x)
        return self.decoders[branch](features)

    @torch.no_grad()
    def swap(self, scene_face: torch.Tensor) -> torch.Tensor:
        """Encode a scene face and decode it through the IDENTITY branch --
        this is the actual face-swap operation used at inference time."""
        self.eval()
        features = self.encoder(scene_face)
        return self.decoders[IDENTITY_BRANCH](features)
