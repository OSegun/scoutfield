"""
The first cell of every Kaggle notebook. Kept here so there is one copy to fix.

Paste as the first cell, or in a notebook simply run:

    !pip install -q -r https://raw.githubusercontent.com/OSegun/scoutfield/main/requirements-kaggle.txt

Requires Internet: On in the notebook settings panel.
"""

BOOTSTRAP_CELL = r'''
# ---------------------------------------------------------------- bootstrap
# Requires: Internet ON, Accelerator GPU, Persistence "Variables and files".
# Installs the pinned pilot and this repo. Does NOT reinstall torch/numpy/etc —
# Kaggle already ships them, and reinstalling burns quota and risks a CUDA
# mismatch against the driver on the machine.
!pip install -q -r https://raw.githubusercontent.com/OSegun/scoutfield/main/requirements-kaggle.txt

import scoutfield
from scoutfield.utils.paths import is_kaggle, output_dir
from scoutfield.utils.seeding import seed_everything

print("scoutfield", scoutfield.__version__,
      "| pilot pinned at", scoutfield.PILOT_TAG,
      "| kaggle:", is_kaggle())

rng = seed_everything(0)

# Only /kaggle/working survives the session. Everything else is discarded.
OUT = output_dir("results")
print("writing results to", OUT)
'''


def write_bootstrap_cell(path: str) -> None:
    """Write BOOTSTRAP_CELL into a .py file for eyeballing outside a notebook."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(BOOTSTRAP_CELL)
