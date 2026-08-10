"""
Push and pull Kaggle notebooks so git stays the source of truth.

    python scripts/kaggle_sync.py push          # all notebooks
    python scripts/kaggle_sync.py push 01       # one, by prefix
    python scripts/kaggle_sync.py pull          # bring executed outputs back
    python scripts/kaggle_sync.py status        # last run status of each kernel

Requires the kaggle CLI and credentials at ~/.kaggle/kaggle.json
(Kaggle -> Settings -> API -> Create New Token).

Why sync rather than just editing on Kaggle: a notebook that only exists in the
Kaggle UI has no history, no review and no diff. Pushing from git means the
version that produced a result can always be recovered.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def kernel_dirs(prefix: str | None = None) -> list[Path]:
    dirs = sorted(d for d in NOTEBOOKS.iterdir()
                  if d.is_dir() and (d / "kernel-metadata.json").exists())
    if prefix:
        dirs = [d for d in dirs if d.name.startswith(prefix)]
    if not dirs:
        sys.exit(f"no notebook directories matched prefix={prefix!r} under {NOTEBOOKS}")
    return dirs


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def push(prefix: str | None) -> None:
    """Upload each notebook. Kaggle queues the run; it does not execute inline."""
    for d in kernel_dirs(prefix):
        _run(["kaggle", "kernels", "push", "-p", str(d)])


def pull(prefix: str | None) -> None:
    """Download the latest version of each kernel, executed outputs included."""
    for d in kernel_dirs(prefix):
        meta = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))
        _run(["kaggle", "kernels", "pull", meta["id"], "-p", str(d), "-m"])


def status(prefix: str | None) -> None:
    for d in kernel_dirs(prefix):
        meta = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))
        _run(["kaggle", "kernels", "status", meta["id"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["push", "pull", "status"])
    parser.add_argument("prefix", nargs="?", default=None,
                        help="notebook directory prefix, e.g. 01")
    args = parser.parse_args()
    {"push": push, "pull": pull, "status": status}[args.action](args.prefix)


if __name__ == "__main__":
    main()
