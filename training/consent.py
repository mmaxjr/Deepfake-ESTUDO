"""Second line of defense: re-verify identity consent right before training starts.

`IdentityDataset.__init__` already refuses to construct without a valid
`consent.yaml` (see `datasets/identity_dataset.py`), but `Trainer.fit`
re-checks here so that a consent file edited/revoked *after* the
`IdentityDataset` object was created (e.g. a long-running notebook session)
still gets caught before a training run starts.
"""

from __future__ import annotations

from datasets.identity_dataset import (
    ConsentNotConfirmedError,
    IdentityConsentManifest,
    IdentityDataset,
    load_consent_manifest,
)

__all__ = ["require_identity_consent", "ConsentNotConfirmedError"]


def require_identity_consent(identity: IdentityDataset) -> IdentityConsentManifest:
    if not isinstance(identity, IdentityDataset):
        raise TypeError(f"Expected IdentityDataset, got {type(identity).__name__}")
    return load_consent_manifest(identity.folder)
