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


def find_checkpoint(*parts: str) -> Path:
    """Locate an existing checkpoint, which an *earlier notebook* may have written.

    Writing and reading a checkpoint are not symmetric on Kaggle, and conflating
    them is why notebook 02 cannot find what notebook 01 produced. A run writes to
    ``/kaggle/working/checkpoints`` — the only writable place — but that directory
    belongs to one session and is gone when it ends. A later notebook receives the
    same file mounted read-only under ``/kaggle/input``, attached either as that
    notebook's committed output (``kernel_sources``) or as a published dataset.

    So both are searched, writable location first, and the first hit wins. Locally
    there is only one location and the extra search costs nothing.

    Raises rather than returning a non-existent path: a missing checkpoint means the
    upstream notebook was never committed or never attached, and that is worth
    saying in as many words at the point of failure.
    """
    tail = Path(*parts)
    candidates = [output_dir("checkpoints") / tail]
    if is_kaggle():
        # Mount depth is not fixed. An attached kernel output sits directly under
        # /kaggle/input/<slug>, while anything fetched with kagglehub keeps its owner
        # and lands at /kaggle/input/<kind>/<owner>/<slug>. Walking a bounded number
        # of levels covers both without guessing which one is in play; three is enough
        # for the deepest layout and stops well above the class directories, so the
        # scan stays cheap even beside a 50,000-image dataset.
        candidates.extend(
            root / prefix / tail
            for root in _shallow_dirs(Path("/kaggle/input"), max_depth=3)
            for prefix in (Path("checkpoints"), Path("."))
        )
    found = next((c for c in candidates if c.is_file()), None)
    if found is None:
        shown = [str(c) for c in candidates[:12]]
        if len(candidates) > len(shown):
            shown.append(f"... and {len(candidates) - len(shown)} more")
        raise FileNotFoundError(
            f"checkpoint '{tail}' not found. Searched: " + ", ".join(shown)
            + ". On Kaggle it is produced by an earlier notebook: commit that notebook "
              "(Save Version -> Save & Run All) and attach it here as a kernel source, "
              "or publish its output as a dataset and attach that."
        )
    return found


def _shallow_dirs(base: Path, max_depth: int) -> list[Path]:
    """``base`` and its subdirectories, breadth-first, no deeper than ``max_depth``.

    Depth-limited on purpose. A plain ``rglob`` under /kaggle/input would descend into
    every attached dataset and stat tens of thousands of image files to find one
    checkpoint.
    """
    if not base.exists():
        return []
    found, frontier = [base], [base]
    for _ in range(max_depth):
        nxt = []
        for directory in frontier:
            try:
                nxt.extend(p for p in sorted(directory.iterdir()) if p.is_dir())
            except OSError:
                continue  # unreadable mount; not worth failing the search over
        found.extend(nxt)
        frontier = nxt
    return found
