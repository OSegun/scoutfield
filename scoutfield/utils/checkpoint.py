"""
Resumable execution.

Kaggle enforces a nine-hour session ceiling and disconnects idle sessions long
before that. The pilot solved the same problem with ``run_jobs.py`` checkpointing
completed jobs to ``results/_done.json``; this module generalises that so both
the sweep driver and the training loops share one mechanism.

The rule this enforces: a job that has completed is never re-run, and a job that
was interrupted leaves no partial row in the results file. Half-written CSV rows
that silently mix into an analysis are worse than a crash.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class DoneRegistry:
    """Tracks which jobs have completed, persisted as JSON.

    Write-through on every completion rather than at exit, because "at exit" is
    exactly the code path a killed session does not reach.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                self._done = set(json.load(fh))

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._done

    def mark(self, job_id: str) -> None:
        self._done.add(job_id)
        self._flush()

    def pending(self, job_ids: Iterable[str]) -> list[str]:
        return [j for j in job_ids if j not in self._done]

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(sorted(self._done), fh, indent=0)
        tmp.replace(self.path)  # atomic; a kill mid-write cannot corrupt the registry

    def reset(self) -> None:
        """Clear the registry. Also delete the results CSV, or a restarted sweep
        will append new rows to old ones and nobody will notice the mixture."""
        self._done.clear()
        if self.path.exists():
            self.path.unlink()


def rng_state() -> dict[str, Any]:
    """Snapshot every RNG the training path touches.

    Omitting these means a resumed run is not a continuation of the run it claims
    to continue: the data order and dropout masks would restart from the seed
    rather than carrying on.
    """
    import random

    import numpy as np

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch
    except ImportError:
        return state
    state["torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Inverse of ``rng_state``. Missing keys are skipped, not guessed at."""
    if not state:
        return

    import random

    import numpy as np

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch
    except ImportError:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["torch_cuda"]])


def save_training_state(path: str | Path, state: dict[str, Any]) -> None:
    """Persist a training checkpoint atomically.

    Writes to a sibling ``.tmp`` and renames, matching the atomicity of
    ``DoneRegistry._flush``: a session killed mid-write leaves the previous
    checkpoint intact rather than a truncated file that fails to load.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_training_state(path: str | Path) -> dict[str, Any] | None:
    """Load a training checkpoint, or None if there is nothing to resume from."""
    import torch

    path = Path(path)
    if not path.exists():
        return None
    # weights_only=False: the checkpoint carries RNG states and scheduler config,
    # not just tensors. It is written by this project, so it is trusted input.
    return torch.load(path, map_location="cpu", weights_only=False)
