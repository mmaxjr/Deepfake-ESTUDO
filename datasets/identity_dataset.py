"""IdentityDataset: the ONLY source of the face identity to be synthesized.

Structurally distinct from `SceneDataset` (does not subclass it, shares no
base class implying "this is a place identity data can come from"). Always
wraps a local, user-supplied folder and unconditionally requires a valid
`consent.yaml` manifest before any data is exposed. See
`configs/example_identity_consent.yaml` for the template and the README's
"Regra de uso" section for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .common import list_images, list_videos


class ConsentNotConfirmedError(RuntimeError):
    """Raised when an identity folder lacks a valid, confirmed consent manifest."""


@dataclass(frozen=True)
class IdentityConsentManifest:
    subject_name: str
    date: str
    statement: str
    source_path: Path


def load_consent_manifest(identity_folder: str | Path) -> IdentityConsentManifest:
    """Load and validate `consent.yaml` inside `identity_folder`.

    Raises ConsentNotConfirmedError if the file is missing, malformed, or
    `consent_confirmed` is not explicitly `true`, or required fields are blank.
    """
    identity_folder = Path(identity_folder)
    manifest_path = identity_folder / "consent.yaml"

    if not manifest_path.exists():
        raise ConsentNotConfirmedError(
            f"No consent.yaml found in {identity_folder}. Copy "
            "configs/example_identity_consent.yaml there, fill it out, and "
            "set consent_confirmed: true before training on this identity."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if raw.get("consent_confirmed") is not True:
        raise ConsentNotConfirmedError(
            f"consent.yaml in {identity_folder} does not have "
            "consent_confirmed: true. Refusing to use this folder as an "
            "identity source."
        )

    subject_name = str(raw.get("subject_name") or "").strip()
    date = str(raw.get("date") or "").strip()
    statement = str(raw.get("statement") or "").strip()

    missing = [
        field
        for field, value in (("subject_name", subject_name), ("date", date), ("statement", statement))
        if not value
    ]
    if missing:
        raise ConsentNotConfirmedError(
            f"consent.yaml in {identity_folder} is missing required field(s): "
            f"{', '.join(missing)}."
        )

    return IdentityConsentManifest(
        subject_name=subject_name, date=date, statement=statement, source_path=manifest_path
    )


class IdentityDataset:
    """A local, consented folder of photos/video for one face identity.

    Deliberately does NOT inherit from SceneDataset and exposes a different
    method name (`iterate_media`, not `iterate_videos`) to avoid accidental
    duck-typed interchange with scene datasets.
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"Identity folder does not exist: {self.folder}")
        # Consent is checked eagerly, at construction time -- an IdentityDataset
        # object cannot exist in memory without a validated manifest.
        self.consent = load_consent_manifest(self.folder)

    @property
    def name(self) -> str:
        return self.folder.name

    def iterate_media(self) -> list[Path]:
        return list_images(self.folder) + list_videos(self.folder)

    def __repr__(self) -> str:
        return f"IdentityDataset(name={self.name!r}, subject={self.consent.subject_name!r})"
