"""
Path resolution across the three environments this project runs in.

The same code runs on a laptop, in a Kaggle notebook and in CI, and each puts
data and writable storage somewhere different. Resolving that here — once —
avoids the alternative, which is an ``if IS_KAGGLE`` scattered through every
training script.

    environment   input data                       writable output
    ------------  -------------------------------  ---------------------------
    local         ./data/                          ./results, ./checkpoints
    Kaggle        /kaggle/input/<dataset-slug>/    /kaggle/working/
    CI            tests use fixtures, no data       tmp_path
"""

from __future__ import annotations

import os
from pathlib import Path


def is_kaggle() -> bool:
    """True when running inside a Kaggle kernel."""
    return Path("/kaggle/input").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def repo_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[2]


def data_dir(dataset_slug: str | None = None) -> Path:
    """Directory holding input data.

    On Kaggle, datasets are mounted read-only at ``/kaggle/input/<slug>``; the
    slug is the last path component of the dataset's URL and is recorded in
    ``configs/perception.yaml`` under ``data.kaggle_slugs``.
    """
    if is_kaggle():
        base = Path("/kaggle/input")
        return base / dataset_slug.split("/")[-1] if dataset_slug else base
    return repo_root() / "data" / (dataset_slug.split("/")[-1] if dataset_slug else "")


def output_dir(*parts: str) -> Path:
    """Writable output directory, created if missing.

    On Kaggle only ``/kaggle/working`` survives to the notebook's output; writing
    anywhere else means the artefact is lost when the session ends.
    """
    base = Path("/kaggle/working") if is_kaggle() else repo_root()
    out = base.joinpath(*parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def results_dir() -> Path:
    return output_dir("results")


def figures_dir() -> Path:
    return output_dir("figures")


def checkpoints_dir(*parts: str) -> Path:
    return output_dir("checkpoints", *parts)
