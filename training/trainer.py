"""Training loop for the shared-encoder / dual-decoder face-swap autoencoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.identity_dataset import IdentityConsentManifest
from models.autoencoder import IDENTITY_BRANCH, SCENE_BRANCH, FaceSwapAutoencoder
from models.losses import reconstruction_loss
from utils.checkpoint import save_checkpoint
from utils.device import get_device

from .dataset_adapters import IdentityFaceSet, SceneFaceSet

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    resolution: int = 128
    latent_dim: int = 512
    batch_size: int = 16
    epochs: int = 200
    lr: float = 2e-4
    log_every: int = 50
    checkpoint_every: int = 1000
    preview_every: int = 200
    checkpoint_dir: str = "data/checkpoints"
    prefer_cuda: bool = True


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(self, config: TrainerConfig, identity_name: str):
        self.config = config
        self.identity_name = identity_name
        self.device = get_device(config.prefer_cuda)
        self.model = FaceSwapAutoencoder(resolution=config.resolution, latent_dim=config.latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.global_step = 0

    def fit(
        self,
        identity_faces: IdentityFaceSet,
        scene_faces: SceneFaceSet,
        consent: IdentityConsentManifest,
    ) -> None:
        if not isinstance(identity_faces, IdentityFaceSet):
            raise TypeError(f"fit() requires an IdentityFaceSet, got {type(identity_faces).__name__}")
        if not isinstance(scene_faces, SceneFaceSet):
            raise TypeError(f"fit() requires a SceneFaceSet, got {type(scene_faces).__name__}")
        if not isinstance(consent, IdentityConsentManifest):
            raise TypeError(f"fit() requires an IdentityConsentManifest, got {type(consent).__name__}")

        logger.info(
            "Training identity '%s' (consented subject: %s, consent date: %s) on device %s",
            self.identity_name, consent.subject_name, consent.date, self.device,
        )

        cfg = self.config
        identity_loader = DataLoader(identity_faces, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        scene_loader = DataLoader(scene_faces, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        identity_iter = _cycle(identity_loader)
        scene_iter = _cycle(scene_loader)

        steps_per_epoch = max(len(identity_loader), len(scene_loader))
        self.model.train()

        for epoch in range(cfg.epochs):
            for _ in range(steps_per_epoch):
                id_warped, id_target = next(identity_iter)
                sc_warped, sc_target = next(scene_iter)
                id_warped, id_target = id_warped.to(self.device), id_target.to(self.device)
                sc_warped, sc_target = sc_warped.to(self.device), sc_target.to(self.device)

                self.optimizer.zero_grad()
                id_pred = self.model(id_warped, IDENTITY_BRANCH)
                sc_pred = self.model(sc_warped, SCENE_BRANCH)
                loss = reconstruction_loss(id_pred, id_target) + reconstruction_loss(sc_pred, sc_target)
                loss.backward()
                self.optimizer.step()
                self.global_step += 1

                if self.global_step % cfg.log_every == 0:
                    logger.info("step=%d epoch=%d loss=%.4f", self.global_step, epoch, loss.item())
                if self.global_step % cfg.preview_every == 0:
                    self._save_preview(id_pred, id_target, sc_pred, sc_target)
                if self.global_step % cfg.checkpoint_every == 0:
                    self.save_checkpoint()

        self.save_checkpoint(tag="final")

    def _save_preview(self, id_pred, id_target, sc_pred, sc_target) -> None:
        preview_dir = Path(self.config.checkpoint_dir) / self.identity_name / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        def to_bgr(t: torch.Tensor) -> np.ndarray:
            arr = t[0].detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
            return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        row = np.concatenate(
            [to_bgr(id_target), to_bgr(id_pred), to_bgr(sc_target), to_bgr(sc_pred)], axis=1
        )
        cv2.imwrite(str(preview_dir / f"step_{self.global_step:07d}.png"), row)

    def save_checkpoint(self, tag: str | None = None) -> Path:
        checkpoint_dir = Path(self.config.checkpoint_dir) / self.identity_name
        name = f"step_{self.global_step:07d}.pt" if tag is None else f"{tag}.pt"
        path = checkpoint_dir / name
        save_checkpoint(
            path,
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "resolution": self.config.resolution,
                "latent_dim": self.config.latent_dim,
                "identity_name": self.identity_name,
            },
        )
        return path
