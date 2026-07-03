"""Prints official access/registration instructions for research datasets.

This module never downloads, scrapes, or mirrors any dataset content -- all
three datasets below require the requester to register and agree to
academic-use terms directly with the dataset owners.
"""

from __future__ import annotations

_INFO = {
    "faceforensics": {
        "display_name": "FaceForensics++",
        "how_to_get_access": (
            "Fill out the request form linked from the official repo "
            "(github.com/ondyari/FaceForensics) to receive download-script "
            "credentials. Access requires agreeing to the dataset's academic "
            "terms of use."
        ),
        "expected_layout": (
            "<root>/original_sequences/youtube/c23/videos/*.mp4\n"
            "<root>/manipulated_sequences/<Method>/c23/videos/*.mp4"
        ),
        "loader": "datasets.scene_dataset.FaceForensicsScene",
    },
    "celebdf": {
        "display_name": "Celeb-DF (v2)",
        "how_to_get_access": (
            "Request access via the official project page "
            "(cited in the Celeb-DF paper / its GitHub repo), which requires "
            "submitting a request form agreeing to research-only use."
        ),
        "expected_layout": (
            "<root>/Celeb-real/*.mp4\n<root>/Celeb-synthesis/*.mp4\n<root>/YouTube-real/*.mp4"
        ),
        "loader": "datasets.scene_dataset.CelebDFScene",
    },
    "dfdc": {
        "display_name": "DFDC (Deepfake Detection Challenge)",
        "how_to_get_access": (
            "Download via Kaggle (kaggle.com/c/deepfake-detection-challenge), "
            "which requires a free Kaggle account and accepting the "
            "competition rules/terms."
        ),
        "expected_layout": (
            "<root>/dfdc_train_part_NN/*.mp4  (or <root>/train_sample_videos/*.mp4)"
        ),
        "loader": "datasets.scene_dataset.DFDCScene",
    },
}

_ALIASES = {
    "faceforensics++": "faceforensics",
    "ff++": "faceforensics",
    "ff": "faceforensics",
    "celeb-df": "celebdf",
}


def get_dataset_info(name: str) -> dict:
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _INFO:
        raise ValueError(f"Unknown dataset '{name}'. Options: {sorted(_INFO)}")
    return _INFO[key]


def print_dataset_info(name: str) -> None:
    info = get_dataset_info(name)
    print(f"== {info['display_name']} ==")
    print(f"Access: {info['how_to_get_access']}")
    print("Expected local folder layout once downloaded:")
    print(info["expected_layout"])
    print(f"Loader: {info['loader']}")
    print(
        "\nThis tool does not download or scrape this dataset. Once you have "
        "it locally (after registering with the dataset owners), pass its "
        "root folder via --dataset-root."
    )
