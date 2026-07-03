"""CLI for the face-swap research tool.

Commands: dataset-info, align, train, infer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from utils.logging import setup_logging

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


@app.command("dataset-info")
def dataset_info_cmd(name: str = typer.Argument(..., help="faceforensics | celebdf | dfdc")):
    """Print official registration/download instructions for a research dataset.
    This tool never downloads or scrapes these datasets itself."""
    from datasets.dataset_info import print_dataset_info

    print_dataset_info(name)


@app.command("align")
def align_cmd(
    identity: Optional[str] = typer.Option(None, help="Identity folder name under data/identities/"),
    scene: Optional[str] = typer.Option(None, help="Path to a scene video/folder, OR a dataset name (with --dataset-root)"),
    dataset: Optional[str] = typer.Option(None, help="Scene dataset name: custom | faceforensics | celebdf | dfdc"),
    dataset_root: Optional[str] = typer.Option(None, help="Local root folder of the chosen research dataset"),
    resolution: int = typer.Option(128, help="Aligned crop resolution"),
    sanity_check: bool = typer.Option(False, "--sanity-check", help="Also dump a landmark-overlay preview"),
    data_root: str = typer.Option("data", help="Project data root"),
):
    """Run face detection + alignment, caching crops to disk for training."""
    setup_logging()
    from align.detector import FaceDetector
    from align.face_cache import build_from_identity, build_from_scene
    from align.visualize import dump_sanity_check
    from datasets.common import list_images, read_image
    from datasets.identity_dataset import IdentityDataset
    from datasets.scene_dataset import build_scene_dataset

    detector = FaceDetector()

    if identity:
        identity_folder = Path(data_root) / "identities" / identity
        identity_ds = IdentityDataset(identity_folder)
        saved = build_from_identity(identity_ds, detector, resolution)
        typer.echo(f"Aligned {len(saved)} identity face crop(s) for '{identity}'.")
        if sanity_check:
            sample_paths = list_images(identity_folder)[:8]
            images = [read_image(p) for p in sample_paths]
            preview_dir = identity_folder / "_sanity_check"
            dump_sanity_check(images, detector, preview_dir)
            typer.echo(f"Sanity-check previews written to {preview_dir}")

    if scene:
        scene_name = dataset or "custom"
        scene_root = dataset_root or scene
        scene_ds = build_scene_dataset(scene_name, scene_root)
        cache_dir = Path(data_root) / "scenes" / "_aligned_cache"
        saved = build_from_scene(scene_ds, detector, resolution, cache_dir)
        typer.echo(f"Aligned {len(saved)} scene face crop(s) into {cache_dir}.")

    if not identity and not scene:
        typer.echo("Nothing to do: pass --identity and/or --scene.")


@app.command("train")
def train_cmd(
    identity: str = typer.Option(..., help="Identity folder name under data/identities/"),
    scene_cache: str = typer.Option(..., help="Path to aligned scene face-crop cache (from `align --scene`)"),
    resolution: int = typer.Option(128),
    epochs: int = typer.Option(200),
    batch_size: int = typer.Option(16),
    lr: float = typer.Option(2e-4),
    data_root: str = typer.Option("data", help="Project data root"),
):
    """Train the shared-encoder / dual-decoder autoencoder for one identity."""
    setup_logging()
    from datasets.identity_dataset import IdentityDataset
    from training.consent import require_identity_consent
    from training.dataset_adapters import IdentityFaceSet, SceneFaceSet
    from training.trainer import Trainer, TrainerConfig

    identity_folder = Path(data_root) / "identities" / identity
    identity_ds = IdentityDataset(identity_folder)  # raises ConsentNotConfirmedError if unset
    consent = require_identity_consent(identity_ds)

    identity_faces = IdentityFaceSet(identity_folder / "_aligned_cache")
    scene_faces = SceneFaceSet(scene_cache)

    config = TrainerConfig(
        resolution=resolution,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        checkpoint_dir=str(Path(data_root) / "checkpoints"),
    )
    trainer = Trainer(config, identity_name=identity)
    trainer.fit(identity_faces, scene_faces, consent)
    typer.echo(f"Training complete. Checkpoints under {config.checkpoint_dir}/{identity}/")


@app.command("infer")
def infer_cmd(
    identity: str = typer.Option(..., help="Identity name (must have a trained checkpoint)"),
    scene: str = typer.Option(..., help="Path to the scene video to process"),
    output: str = typer.Option(..., help="Output video path"),
    checkpoint: Optional[str] = typer.Option(None, help="Explicit checkpoint path (defaults to latest)"),
    resolution: int = typer.Option(128),
    blend: str = typer.Option("feather", help="'feather' or 'poisson'"),
    keep_audio: bool = typer.Option(True),
    data_root: str = typer.Option("data", help="Project data root"),
):
    """Run the trained model over a scene video, producing a swapped output video."""
    setup_logging()
    from align.detector import FaceDetector
    from inference.swapper import FaceSwapper
    from models.autoencoder import FaceSwapAutoencoder
    from utils.checkpoint import latest_checkpoint, load_checkpoint
    from utils.device import get_device

    checkpoint_dir = Path(data_root) / "checkpoints" / identity
    checkpoint_path = Path(checkpoint) if checkpoint else latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise typer.BadParameter(f"No checkpoint found under {checkpoint_dir}. Train the identity first.")

    device = get_device(prefer_cuda=True)
    state = load_checkpoint(checkpoint_path, map_location=str(device))

    model = FaceSwapAutoencoder(resolution=state.get("resolution", resolution), latent_dim=state.get("latent_dim", 512))
    model.load_state_dict(state["model_state"])

    detector = FaceDetector()
    swapper = FaceSwapper(model, detector, resolution=state.get("resolution", resolution), device=device, blend=blend)
    result_path = swapper.process_video(scene, output, keep_audio=keep_audio)
    typer.echo(f"Wrote {result_path}")


if __name__ == "__main__":
    app()
