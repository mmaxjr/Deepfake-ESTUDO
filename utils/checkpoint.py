"""Checkpoint save/load helpers for the face-swap autoencoder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Saved checkpoint: %s", path)


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict[str, Any]:
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None
