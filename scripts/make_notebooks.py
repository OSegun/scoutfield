"""
Generate the notebooks and their Kaggle metadata.

    python scripts/make_notebooks.py          # only creates missing notebooks
    python scripts/make_notebooks.py --force  # rebuild every notebook too
    python scripts/make_notebooks.py --user someone-else   # one-off username override

Each notebook gets the shared bootstrap cell and a call into an ``experiments/`` script.
Logic lives in the package; a notebook that contains a training loop is a bug.

One source, four files
----------------------
The Kaggle API requires **one ``kernel-metadata.json`` per kernel directory** — each
declares its own ``id``, ``code_file``, accelerator and attached datasets, and
``kaggle kernels push -p <dir>`` reads exactly one of them. Four kernels therefore mean
four files, and that is not negotiable.

What *is* negotiable is whether they are maintained in four places. They are not: all
four are generated here, from ``configs/kaggle.yaml`` (who is running this) and
``configs/perception.yaml`` (which datasets, shared with the training code so a slug is
never written twice). Metadata is rewritten on every run, without ``--force``, precisely
so that "change it in one place" stays true — a hand-edit in a generated file would be
silently reverted, which is the intended behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from _bootstrap import render_bootstrap  # noqa: E402


def load_settings(user_override: str | None = None) -> dict:
    """Read the two config files that between them describe every generated value."""
    kaggle = yaml.safe_load((ROOT / "configs" / "kaggle.yaml").read_text(encoding="utf-8"))
    perception = yaml.safe_load(
        (ROOT / "configs" / "perception.yaml").read_text(encoding="utf-8")
    )
    return {
        "user": user_override or kaggle["username"],
        "repo": kaggle["repo"],
        "branch": kaggle["branch"],
        # Resolved from the training config, so the slugs cannot drift apart.
        "slugs": perception["data"]["kaggle_slugs"],
    }


NOTEBOOKS = [
    {
        "dir": "01_perception_finetune",
        "title": "scoutfield 01 Perception Finetune",
        "gpu": True,
        # Keys into `data.kaggle_slugs` in configs/perception.yaml, not slugs.
        "datasets": ["plantvillage"],
        "purpose": (
            "Fine-tune EfficientNet-B0 on PlantVillage.\n\n"
            "Roadmap item 1. Produces a checkpoint and `results/perception_summary.json`, "
            "whose measured accuracy becomes the BASE_ACC of this phase, replacing the "
            "pilot's cited 0.816.\n\n"
            "Checkpoints every epoch: a session killed at epoch 11 of 12 must resume, "
            "not restart."
        ),
        "calls": [
            "!python experiments/01_finetune_perception.py --config configs/perception.yaml",
            "",
            "import json",
            "print(json.load(open('results/perception_summary.json')))",
        ],
    },
    {
        "dir": "02_calibration_shift",
        "title": "scoutfield 02 Calibration And Shift",
        "gpu": True,
        "datasets": ["plantvillage", "plantdoc"],
        "purpose": (
            "Fit the temperature on validation, sweep it, and evaluate the distribution "
            "shift on PlantDoc.\n\n"
            "Roadmap items 2, 3 and 7. The critical assertion lives here: accuracy must "
            "be invariant across the temperature sweep to four decimal places. If it is "
            "not, the instrument is broken and nothing downstream is interpretable."
        ),
        "calls": [
            "!python experiments/02_calibrate.py --config configs/perception.yaml",
            "",
            "import json",
            "sweep = json.load(open('results/calibration_sweep.json'))",
            "accs = {round(r['accuracy'], 4) for r in sweep.values()}",
            "assert len(accs) == 1, f'instrument broken: {accs}'",
            "print('accuracy invariant at', accs.pop())",
        ],
    },
    {
        "dir": "03_ppo_training",
        "title": "scoutfield 03 PPO Training",
        "gpu": True,
        "datasets": [],
        "purpose": (
            "Train PPO on the 32x32 field with the global belief map.\n\n"
            "Roadmap items 4 and 5. The script verifies the coverage ceiling before "
            "training and refuses to run outside the intended regime — the pilot's "
            "budget permitted near-total coverage, which is why adaptive planning had "
            "nothing to buy.\n\n"
            "Train at least three seeds. A single learning curve is not evidence of "
            "convergence."
        ),
        "calls": [
            "for seed in range(3):",
            "    !python experiments/03_train_ppo.py --config configs/ppo_field32.yaml --seed {seed}",
        ],
    },
    {
        "dir": "04_evaluation_sweep",
        "title": "scoutfield 04 Evaluation Sweep",
        "gpu": False,
        "datasets": [],
        "purpose": (
            "Full evaluation sweep, tau sensitivity, and figures.\n\n"
            "Roadmap items 6 and 8. CPU is enough: evaluation is thousands of short "
            "episodes where per-step transfer latency dominates.\n\n"
            "Run in bounded chunks so a killed session costs one chunk, not the whole "
            "sweep. To restart cleanly, delete BOTH results/_done.json and the results "
            "CSV — deleting one leaves the driver mixing old and new results with no "
            "warning."
        ),
        "calls": [
            "!while JOB_BUDGET=60 python experiments/04_sweep.py; do :; done",
            "!python experiments/05_tau_sensitivity.py --config configs/sweep.yaml",
            "!python experiments/make_figures.py",
        ],
    },
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_notebook(spec: dict, settings: dict) -> dict:
    cells = [
        md(f"# {spec['title']}\n\n{spec['purpose']}\n\n"
           "> Logic lives in `scoutfield/` and `experiments/`. This notebook orchestrates: "
           "install, configure, call, display.\n"),
        code(render_bootstrap(settings["repo"], settings["branch"]).strip()),
        md("## Run\n"),
        code("\n".join(spec["calls"])),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_metadata(spec: dict, settings: dict) -> dict:
    slug = "scoutfield-" + spec["dir"].replace("_", "-")
    slugs = settings["slugs"]
    unknown = [k for k in spec["datasets"] if k not in slugs]
    if unknown:
        raise KeyError(
            f"notebook '{spec['dir']}' asks for dataset key(s) {unknown}, which are not in "
            f"configs/perception.yaml under data.kaggle_slugs (have: {sorted(slugs)})"
        )
    return {
        "id": f"{settings['user']}/{slug}",
        "title": spec["title"],
        "code_file": f"{spec['dir']}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
        "enable_gpu": "true" if spec["gpu"] else "false",
        "enable_tpu": "false",
        # Internet must be on: the clone and pip install both fail without it.
        "enable_internet": "true",
        "dataset_sources": [slugs[k] for k in spec["datasets"]],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing notebooks")
    parser.add_argument("--user", default=None,
                        help="override configs/kaggle.yaml username for this run")
    args = parser.parse_args()

    settings = load_settings(args.user)
    print(f"user={settings['user']} repo={settings['repo']}@{settings['branch']}")

    for spec in NOTEBOOKS:
        d = ROOT / "notebooks" / spec["dir"]
        d.mkdir(parents=True, exist_ok=True)

        nb_path = d / f"{spec['dir']}.ipynb"
        if args.force or not nb_path.exists():
            nb_path.write_text(json.dumps(build_notebook(spec, settings), indent=1),
                               encoding="utf-8")
            print("wrote", nb_path.relative_to(ROOT))

        # Always rewritten, --force or not. Metadata is wholly derived from the two
        # config files, so regenerating it unconditionally is what makes "change the
        # username in one place" actually true. A hand-edit here is meant to be lost.
        meta_path = d / "kernel-metadata.json"
        meta_path.write_text(json.dumps(build_metadata(spec, settings), indent=2) + "\n",
                             encoding="utf-8")
        print("wrote", meta_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
