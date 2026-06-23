"""Runtime model path helpers for the image classifier service."""

from __future__ import annotations

import os
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent


def classifier_model_dir() -> Path:
    candidates = []
    configured_dir = str(os.getenv("CLASSIFIER_MODEL_DIR") or "").strip()
    if configured_dir:
        candidates.append(Path(configured_dir).expanduser())

    app_models_dir = str(os.getenv("APP_MODELS_DIR") or "").strip()
    if app_models_dir:
        candidates.append(Path(app_models_dir).expanduser() / "image_classifier")

    candidates.extend(
        [
            Path("/app/runtime_assets/models/image_classifier"),
            MODULE_DIR / "weights",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def classifier_weight_path(filename: str) -> Path:
    return classifier_model_dir() / filename
