"""Tests for the identity consent gate -- no ML dependencies required."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from datasets.identity_dataset import (
    ConsentNotConfirmedError,
    IdentityDataset,
    load_consent_manifest,
)


def _write_consent(folder: Path, **overrides) -> None:
    data = {
        "consent_confirmed": True,
        "subject_name": "Test Subject",
        "date": "2026-01-01",
        "statement": "I consent to my own likeness being used.",
    }
    data.update(overrides)
    with open(folder / "consent.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def test_missing_consent_file_raises(tmp_path: Path):
    identity_folder = tmp_path / "alice"
    identity_folder.mkdir()
    with pytest.raises(ConsentNotConfirmedError):
        load_consent_manifest(identity_folder)


def test_consent_not_confirmed_raises(tmp_path: Path):
    identity_folder = tmp_path / "bob"
    identity_folder.mkdir()
    _write_consent(identity_folder, consent_confirmed=False)
    with pytest.raises(ConsentNotConfirmedError):
        load_consent_manifest(identity_folder)


def test_consent_missing_fields_raises(tmp_path: Path):
    identity_folder = tmp_path / "carol"
    identity_folder.mkdir()
    _write_consent(identity_folder, subject_name="")
    with pytest.raises(ConsentNotConfirmedError):
        load_consent_manifest(identity_folder)


def test_valid_consent_loads(tmp_path: Path):
    identity_folder = tmp_path / "dana"
    identity_folder.mkdir()
    _write_consent(identity_folder)
    manifest = load_consent_manifest(identity_folder)
    assert manifest.subject_name == "Test Subject"


def test_identity_dataset_requires_consent_at_construction(tmp_path: Path):
    identity_folder = tmp_path / "erin"
    identity_folder.mkdir()
    with pytest.raises(ConsentNotConfirmedError):
        IdentityDataset(identity_folder)

    _write_consent(identity_folder)
    ds = IdentityDataset(identity_folder)
    assert ds.name == "erin"
    assert ds.consent.subject_name == "Test Subject"


def test_scene_dataset_is_not_an_identity_dataset():
    from datasets.scene_dataset import CustomVideoScene, SceneDataset

    assert not issubclass(CustomVideoScene, IdentityDataset)
    assert not issubclass(IdentityDataset, SceneDataset)
