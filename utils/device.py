"""Device selection: auto-detect CUDA, fall back to CPU."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_device(prefer_cuda: bool = True) -> "torch.device":
    import torch

    if prefer_cuda and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("Using CUDA device: %s", name)
        return torch.device("cuda")

    logger.info("CUDA not available or disabled; using CPU")
    return torch.device("cpu")
