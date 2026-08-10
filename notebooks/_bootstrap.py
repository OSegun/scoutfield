"""
The first cell of every Kaggle notebook. Kept here so there is one copy to fix.

The cell is rendered from ``configs/kaggle.yaml`` by ``scripts/make_notebooks.py``, so
changing the fork or the branch means editing that YAML and re-running:

    python scripts/make_notebooks.py --force

Why the repository is cloned rather than only pip-installed
-----------------------------------------------------------
``pyproject.toml`` packages ``scoutfield*`` and nothing else, deliberately:
``experiments/`` holds runnable drivers and is not part of the importable surface. But
that means a pip install alone leaves ``experiments/`` and ``configs/`` absent from the
Kaggle session, and every run cell dies with "can't open file". So the notebook clones
the repo, installs *from the clone*, and works inside it.

Installing from the clone has a second benefit: the requirements file is then a local
path, so nothing depends on ``raw.githubusercontent.com`` being reachable or on the
branch being public at the moment the cell runs.
"""

from __future__ import annotations

BOOTSTRAP_TEMPLATE = r'''
# ---------------------------------------------------------------- bootstrap
# Requires: Internet ON, Accelerator GPU (None for the evaluation sweep),
# Persistence "Variables and Files".
#
# The repo is CLONED, not merely pip-installed. `pyproject.toml` ships the
# `scoutfield*` packages only, so `experiments/` and `configs/` would otherwise be
# missing and every run cell below would fail with "can't open file".
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/{repo}.git"
BRANCH = "{branch}"
SRC = "/kaggle/working/scoutfield"

if not os.path.isdir(os.path.join(SRC, ".git")):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, SRC],
                   check=True)
os.chdir(SRC)

# Printed, not assumed: `main` moves, so the commit is the only honest record of what
# this run actually executed. Quote it beside any number this notebook produces.
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()

# Dependencies come from the CLONED file, so nothing here depends on
# raw.githubusercontent.com. Kaggle already ships torch, torchvision, numpy, pandas,
# scikit-learn and pillow — reinstalling them burns GPU quota and risks a CUDA
# mismatch against the driver on the machine, so they are not in that file.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                "requirements-kaggle.txt"], check=True)
# Editable, --no-deps: the line above already installed the pilot and the rest.
# `python experiments/01_....py` puts `experiments/` on sys.path, not the repo root,
# so `import scoutfield` needs a real install rather than the working directory.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"],
               check=True)

# Only /kaggle/working survives the session, and `utils.paths` writes there. The repo
# also carries results/, checkpoints/ and figures/ (each holding just a .gitkeep), so
# repo-relative paths and absolute ones would otherwise point at different places and
# the sweep would write its CSV somewhere its own summary step cannot find it.
for _d in ("results", "checkpoints", "figures"):
    _target = f"/kaggle/working/{{_d}}"
    os.makedirs(_target, exist_ok=True)
    if os.path.islink(_d):
        continue
    if os.path.isdir(_d):
        shutil.rmtree(_d)          # a fresh clone puts only .gitkeep in here
    os.symlink(_target, _d)

import scoutfield
from scoutfield.utils.paths import is_kaggle, output_dir
from scoutfield.utils.seeding import seed_everything

print("scoutfield", scoutfield.__version__, "@", COMMIT,
      "| pilot pinned at", scoutfield.PILOT_TAG,
      "| kaggle:", is_kaggle())

rng = seed_everything(0)

OUT = output_dir("results")
print("writing results to", OUT)
'''


def render_bootstrap(repo: str = "OSegun/scoutfield", branch: str = "main") -> str:
    """Fill the template. Called by scripts/make_notebooks.py with configs/kaggle.yaml."""
    return BOOTSTRAP_TEMPLATE.format(repo=repo, branch=branch)


#: Default rendering, so the cell can be imported and read without a config file.
BOOTSTRAP_CELL = render_bootstrap()


def write_bootstrap_cell(path: str, repo: str = "OSegun/scoutfield",
                         branch: str = "main") -> None:
    """Write the rendered cell into a .py file for eyeballing outside a notebook."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_bootstrap(repo, branch))
