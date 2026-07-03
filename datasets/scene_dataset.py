"""Scene datasets: driving video only, never an identity source.

IMPORTANT (hard constraint, see README): every class here exposes only
`path / dataset_name / split` for each video. None of them read or surface
any actor/identity/label field, even when the underlying dataset ships one
(e.g. DFDC's `metadata.json` has a `label`/`original` column listing which
real actor a fake video was derived from -- we deliberately never touch it).
This is what keeps "scene" data structurally incapable of being wired in as
a synthesized identity: there is no identity attribute here to wire in.

None of these loaders download or scrape data. They only enumerate videos
already present in a user-supplied local folder. See `dataset_info.py` for
the official registration/download instructions per dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .common import list_videos


@dataclass(frozen=True)
class SceneVideoRecord:
    """A single scene/driving video. Deliberately has no identity field."""

    path: Path
    dataset_name: str
    split: str = "all"


class SceneDataset(ABC):
    """Base class for any source of driving/scene video footage."""

    dataset_name: str = "custom"

    @abstractmethod
    def iterate_videos(self) -> Iterator[SceneVideoRecord]:
        """Yield every available scene video record."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[SceneVideoRecord]:
        return self.iterate_videos()

    def count(self) -> int:
        return sum(1 for _ in self.iterate_videos())


class CustomVideoScene(SceneDataset):
    """A single arbitrary video file, or a folder of videos, supplied by the user."""

    dataset_name = "custom"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Scene path does not exist: {self.path}")

    def iterate_videos(self) -> Iterator[SceneVideoRecord]:
        if self.path.is_file():
            yield SceneVideoRecord(path=self.path, dataset_name=self.dataset_name)
            return
        for video_path in list_videos(self.path):
            yield SceneVideoRecord(path=video_path, dataset_name=self.dataset_name)


class FaceForensicsScene(SceneDataset):
    """FaceForensics++ layout:

        <root>/original_sequences/youtube/c23/videos/*.mp4
        <root>/manipulated_sequences/<method>/c23/videos/*.mp4

    Registration required: see `dataset_info.py` / `cli.py dataset-info faceforensics`.
    Only enumerates videos already downloaded locally under `root`.
    """

    dataset_name = "faceforensics++"

    def __init__(self, root: str | Path, compression: str = "c23"):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"FaceForensics++ root not found: {self.root}")
        self.compression = compression

    def iterate_videos(self) -> Iterator[SceneVideoRecord]:
        candidates = [
            (self.root / "original_sequences" / "youtube" / self.compression / "videos", "original"),
        ]
        manipulated_root = self.root / "manipulated_sequences"
        if manipulated_root.exists():
            for method_dir in sorted(manipulated_root.iterdir()):
                if method_dir.is_dir():
                    candidates.append((method_dir / self.compression / "videos", method_dir.name))

        for videos_dir, split in candidates:
            if not videos_dir.exists():
                continue
            for video_path in list_videos(videos_dir):
                yield SceneVideoRecord(path=video_path, dataset_name=self.dataset_name, split=split)


class CelebDFScene(SceneDataset):
    """Celeb-DF (v2) layout:

        <root>/Celeb-real/*.mp4
        <root>/Celeb-synthesis/*.mp4
        <root>/YouTube-real/*.mp4

    Registration required: see `dataset_info.py` / `cli.py dataset-info celebdf`.
    """

    dataset_name = "celeb-df"
    _SUBFOLDERS = ("Celeb-real", "Celeb-synthesis", "YouTube-real")

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Celeb-DF root not found: {self.root}")

    def iterate_videos(self) -> Iterator[SceneVideoRecord]:
        for split in self._SUBFOLDERS:
            split_dir = self.root / split
            if not split_dir.exists():
                continue
            for video_path in list_videos(split_dir):
                yield SceneVideoRecord(path=video_path, dataset_name=self.dataset_name, split=split)


class DFDCScene(SceneDataset):
    """DFDC layout (Kaggle download, full or sample set):

        <root>/dfdc_train_part_NN/*.mp4   (or <root>/train_sample_videos/*.mp4)

    Registration required: see `dataset_info.py` / `cli.py dataset-info dfdc`.

    Note: DFDC ships a `metadata.json` per part with a `label`/`original`
    column identifying the source actor of each fake video. This loader
    intentionally never reads that file -- only file paths are enumerated.
    """

    dataset_name = "dfdc"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"DFDC root not found: {self.root}")

    def iterate_videos(self) -> Iterator[SceneVideoRecord]:
        part_dirs = [p for p in self.root.iterdir() if p.is_dir()] or [self.root]
        for part_dir in sorted(part_dirs):
            for video_path in list_videos(part_dir):
                yield SceneVideoRecord(path=video_path, dataset_name=self.dataset_name, split=part_dir.name)


_SCENE_BUILDERS = {
    "custom": CustomVideoScene,
    "faceforensics": FaceForensicsScene,
    "faceforensics++": FaceForensicsScene,
    "celebdf": CelebDFScene,
    "celeb-df": CelebDFScene,
    "dfdc": DFDCScene,
}


def build_scene_dataset(name: str, root: str | Path, **kwargs) -> SceneDataset:
    """Factory: build a SceneDataset by short name (see `_SCENE_BUILDERS`)."""
    key = name.strip().lower()
    if key not in _SCENE_BUILDERS:
        raise ValueError(f"Unknown scene dataset '{name}'. Options: {sorted(set(_SCENE_BUILDERS))}")
    return _SCENE_BUILDERS[key](root, **kwargs)
