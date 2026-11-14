"""
Path resolution across the three environments this project runs in.

The same code runs on a laptop, in a Kaggle notebook and in CI, and each puts
data and writable storage somewhere different. Resolving that here — once —
avoids the alternative, which is an ``if IS_KAGGLE`` scattered through every
training script.

    environment   input data                       writable output
    ------------  -------------------------------  ---------------------------
    local         ./data/                          <local_work_dir>/results, ...
    Kaggle        /kaggle/input/<dataset-slug>/    /kaggle/working/

``local_work_dir`` is set in ``configs/artifacts.yaml``. Locally, stages never write
into the repository's own ``results/`` and ``figures/``: those hold the promoted
outputs of one run and only the promotion step writes them.
    CI            tests use fixtures, no data       tmp_path

Files that cross from one stage to another never travel through these paths: they
go through the artefact store (scoutfield.utils.artifacts), so no stage depends
on where Kaggle mounted another stage's output.
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

    A dataset arrives in one of two layouts, and both occur in practice:

    * **Attached as a notebook input** — Kaggle mounts it read-only at
      ``/kaggle/input/<name>``, where ``<name>`` is the slug's last component.
    * **Downloaded with kagglehub** — the owner is kept, giving
      ``<cache>/datasets/<owner>/<name>``.

    Both are checked, in that order, and the first that exists wins. When neither
    does, the attached-input path is returned so the caller's error message names
    the canonical location rather than the fallback.

    Slugs are recorded in ``configs/perception.yaml`` under ``data.kaggle_slugs``.
    """
    base = Path("/kaggle/input") if is_kaggle() else repo_root() / "data"
    if not dataset_slug:
        return base
    owner, _, name = dataset_slug.rpartition("/")
    candidates = [base / name]
    if owner:
        candidates.append(base / "datasets" / owner / name)
    return next((c for c in candidates if c.exists()), candidates[0])


def local_work_root() -> Path:
    """``local_work_dir`` from ``configs/artifacts.yaml``, resolved against the repo."""
    from scoutfield.config import load_config

    settings = load_config(repo_root() / "configs" / "artifacts.yaml")
    if "local_work_dir" not in settings.data:
        raise KeyError("configs/artifacts.yaml has no local_work_dir; local stage runs "
                       "need it so they never write into the promoted results/")
    root = Path(settings["local_work_dir"])
    return root if root.is_absolute() else repo_root() / root


def output_dir(*parts: str) -> Path:
    """Writable output directory, created if missing.

    On Kaggle only ``/kaggle/working`` survives to the notebook's output; writing
    anywhere else means the artefact is lost when the session ends. Locally the
    base is ``local_work_dir`` from ``configs/artifacts.yaml``.
    """
    base = Path("/kaggle/working") if is_kaggle() else local_work_root()
    out = base.joinpath(*parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def results_dir() -> Path:
    return output_dir("results")


def figures_dir() -> Path:
    return output_dir("figures")


def checkpoints_dir(*parts: str) -> Path:
    return output_dir("checkpoints", *parts)
