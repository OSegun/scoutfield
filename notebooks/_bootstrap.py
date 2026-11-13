"""
The first cell of every Kaggle notebook. Kept here so there is one copy to fix.

The cell is rendered from ``configs/kaggle.yaml`` and ``configs/artifacts.yaml`` by
``scripts/make_notebooks.py``, so changing the fork, the branch or the store means
editing that YAML and re-running:

    python scripts/make_notebooks.py --force

Two checks run before anything is cloned or installed
------------------------------------------------------
1. **The Hugging Face token.** Every stage reads its inputs from, and writes its
   outputs to, the artefact store, and writing needs ``HF_TOKEN``. It comes from
   Kaggle Secrets and is attached per notebook; without it the cell stops at once.
2. **The GPU**, in the GPU notebooks. Kaggle still offers the P100 (compute
   capability 6.0), which the shipped PyTorch cannot run; the cell stops if no GPU
   is visible or its capability is below 7.0.

Both fail within seconds, so a misconfigured run costs almost no quota.

Why the repository is cloned rather than only pip-installed
-----------------------------------------------------------
``pyproject.toml`` packages ``scoutfield*`` and nothing else, deliberately:
``experiments/`` holds runnable drivers and is not part of the importable surface. A
pip install alone would leave ``experiments/`` and ``configs/`` absent from the
Kaggle session. So the notebook clones the repo, installs *from the clone*, and works
inside it.
"""

from __future__ import annotations

CHECKS_TEMPLATE = r'''
# ------------------------------------------------------------------ checks
# Run before cloning or installing anything, so a misconfigured notebook stops in
# seconds and spends almost no quota.
import os

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    raise RuntimeError("Attach the HF_TOKEN secret: Add-ons → Secrets") from exc
if not os.environ.get("HF_TOKEN"):
    raise RuntimeError("Attach the HF_TOKEN secret: Add-ons → Secrets")
'''

GPU_CHECK_TEMPLATE = r'''
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 0):
    raise RuntimeError("Set the accelerator to GPU T4 x2")
'''

BOOTSTRAP_TEMPLATE = r'''
# ---------------------------------------------------------------- bootstrap
# Requires: Internet ON, the HF_TOKEN secret attached, and the accelerator in the
# notebook's settings ({accelerator}).
#
# The repo is CLONED, not merely pip-installed. `pyproject.toml` ships the
# `scoutfield*` packages only, so `experiments/` and `configs/` would otherwise be
# missing and every run cell below would fail with "can't open file".
import shutil
import subprocess
import sys

REPO = "https://github.com/{repo}.git"
BRANCH = "{branch}"
SRC = "/kaggle/working/scoutfield"
os.environ["SCOUTFIELD_KAGGLE_KERNEL"] = "{kernel}"

# Cloned if absent, reset to the branch tip if present: a clone left over in
# /kaggle/working would otherwise pin the session to an old commit.
if not os.path.isdir(os.path.join(SRC, ".git")):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, SRC],
                   check=True)
else:
    subprocess.run(["git", "-C", SRC, "fetch", "--depth", "1", "origin", BRANCH],
                   check=True)
    subprocess.run(["git", "-C", SRC, "reset", "--hard", f"origin/{{BRANCH}}"], check=True)
    subprocess.run(["git", "-C", SRC, "clean", "-fd"], check=True)
os.chdir(SRC)

# Dependencies come from the CLONED file, including huggingface_hub for the
# artefact store. Kaggle already ships torch, torchvision, numpy, pandas,
# scikit-learn and pillow, so they are not reinstalled.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                "requirements-kaggle.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"],
               check=True)

# `utils.paths` writes to /kaggle/working, while the configs name repo-relative
# paths such as results/_done.json. Linking the repo's results/, checkpoints/ and
# figures/ to /kaggle/working makes both name the same directory.
for _d in ("results", "checkpoints", "figures"):
    _target = f"/kaggle/working/{{_d}}"
    os.makedirs(_target, exist_ok=True)
    if os.path.islink(_d):
        continue
    if os.path.isdir(_d):
        shutil.rmtree(_d)          # a fresh clone puts only .gitkeep in here
    os.symlink(_target, _d)

import scoutfield
from scoutfield.utils.artifacts import load_settings

COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
SETTINGS = load_settings()
print("repo commit:", COMMIT, "| scoutfield", scoutfield.__version__,
      "| pilot pinned at", scoutfield.PILOT_TAG)
print("run_id:", SETTINGS["run_id"])
print("artefact store:", SETTINGS["hub_repo"])
print("HF_TOKEN:", "present" if os.environ.get("HF_TOKEN") else "MISSING")
'''


def render_bootstrap(repo: str = "OSegun/scoutfield", branch: str = "main",
                     gpu: bool = True, kernel: str = "") -> str:
    """Fill the template. Called by scripts/make_notebooks.py with the configs."""
    cell = CHECKS_TEMPLATE
    if gpu:
        cell += GPU_CHECK_TEMPLATE
    return cell + BOOTSTRAP_TEMPLATE.format(
        repo=repo, branch=branch, kernel=kernel,
        accelerator="GPU T4 x2" if gpu else "None")


#: Default rendering, so the cell can be imported and read without a config file.
BOOTSTRAP_CELL = render_bootstrap()
