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
import re
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
        # Keys into `data.kaggle_slugs` in configs/perception.yaml, not slugs. The
        # leakage check hashes PlantDoc as well, so both are attached.
        "datasets": ["plantvillage", "plantdoc"],
        "purpose": (
            "Leakage check, then fine-tune EfficientNet-B0 on PlantVillage.\n\n"
            "Roadmap item 1. Writes `leakage_report.json`, `best.pt`, `last.pt` and "
            "`perception_summary.json` to stage `01_perception` of the artefact store. "
            "`last.pt` is pushed every epoch.\n\n"
            "If the session dies part-way through training, set `RESUME = True` in the "
            "run cell and run again: it continues from the last finished epoch."
        ),
        "calls": [
            "RESUME = False   # True continues this run from 01_perception/last.pt",
            "",
            "!python experiments/00_check_leakage.py --config configs/perception.yaml",
            "assert _exit_code == 0, 'leakage check failed; see the output above'",
            "",
            "RESUME_FLAG = '--resume' if RESUME else ''",
            ("!python experiments/01_finetune_perception.py --config configs/perception.yaml"
             " $RESUME_FLAG"),
            "assert _exit_code == 0, 'fine-tuning failed; see the output above'",
        ],
    },
    {
        "dir": "02_calibration_shift",
        # Title must slugify to the id — see `slugify` and the check in build_metadata.
        # "Calibration And Shift" would give ...-calibration-and-shift and not resolve.
        "title": "scoutfield 02 Calibration Shift",
        "gpu": True,
        "datasets": ["plantvillage", "plantdoc"],
        "purpose": (
            "Fit the lab and shift temperatures, sweep both, build the logit pools.\n\n"
            "Roadmap items 2 and 3. Reads `01_perception/best.pt` from the artefact "
            "store and writes `calibration_sweep.json` and three logit pools to stage "
            "`02_calibration`. It stops if accuracy moves with temperature, or if the "
            "signed calibration error on `shift_eval` does not change sign at T = 1."
        ),
        "calls": [
            "!python experiments/02_calibrate.py --config configs/perception.yaml",
            "assert _exit_code == 0, 'calibration failed; see the output above'",
        ],
    },
    {
        "dir": "03_ppo_training",
        "title": "scoutfield 03 PPO Training",
        "gpu": True,
        # The pools normally come from the store. Both datasets are attached so a
        # pool can be rebuilt if its recorded split does not match.
        "datasets": ["plantvillage", "plantdoc"],
        "purpose": (
            "Train PPO on the 32x32 field, one policy per training seed.\n\n"
            "Roadmap items 4 and 5. Trains on the `shift_fit` pool at the shift-fitted "
            "temperature. Resume state and the learning curve are pushed to stage "
            "`03_ppo` periodically, so if the session ends, run the notebook again: "
            "finished seeds are skipped and an unfinished one resumes."
        ),
        "calls": [
            "import yaml",
            "SEEDS = yaml.safe_load(open('configs/sweep.yaml'))['ppo_train_seeds']",
            "for seed in SEEDS:",
            ("    !python experiments/03_train_ppo.py --config configs/ppo_field32.yaml"
             " --seed {seed}"),
            "    assert _exit_code == 0, f'PPO seed {seed} failed; see the output above'",
        ],
    },
    {
        "dir": "04_evaluation_sweep",
        "title": "scoutfield 04 Evaluation Sweep",
        "gpu": False,
        "datasets": ["plantvillage", "plantdoc"],
        "purpose": (
            "Evaluation sweep, tau sweep and figures.\n\n"
            "Roadmap items 6 and 8. CPU is enough. Progress is pushed to the artefact "
            "store every 25 jobs, so if the session ends, run the notebook again: it "
            "resumes from the store and skips finished jobs."
        ),
        "calls": [
            "# The sweep exits 0 while jobs remain, 1 when every job is done and 2 if it",
            "# aborted. The first non-zero code is recorded inside the loop, because a",
            "# `while` loop whose condition fails reports 0, not the condition's code.",
            ("!bash -c 'while true; do JOB_BUDGET=60 python experiments/04_sweep.py "
             "|| { echo $? > /tmp/sweep_exit; break; }; done'"),
            "SWEEP_EXIT = int(open('/tmp/sweep_exit').read())",
            "assert SWEEP_EXIT == 1, (f'sweep aborted (exit {SWEEP_EXIT}); see the output '",
            "                         'above. Running again resumes from the store.')",
            "",
            "!python experiments/05_tau_sensitivity.py --config configs/sweep.yaml",
            "assert _exit_code == 0, 'tau sweep failed; see the output above'",
            "",
            "!python experiments/make_figures.py",
            "assert _exit_code == 0, 'figures failed; see the output above'",
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
        code(render_bootstrap(settings["repo"], settings["branch"], gpu=spec["gpu"],
                              kernel=kernel_id(spec, settings)).strip()),
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


def slugify(title: str) -> str:
    """Kaggle's rule for turning a kernel title into a URL slug.

    Lowercase, non-alphanumerics collapsed to single hyphens, ends trimmed.
    """
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def kernel_id(spec: dict, settings: dict) -> str:
    return f"{settings['user']}/scoutfield-" + spec["dir"].replace("_", "-")


def build_metadata(spec: dict, settings: dict) -> dict:
    slug = "scoutfield-" + spec["dir"].replace("_", "-")
    # Kaggle derives a slug from the title and expects it to match the id. When it
    # does not, `kaggle kernels push` warns and then fails the save with a 409
    # Conflict — the push is rejected outright, so this is not cosmetic. Caught here
    # because the title is free text and one stray word ("And") breaks it.
    if slugify(spec["title"]) != slug:
        raise ValueError(
            f"notebook '{spec['dir']}': title {spec['title']!r} slugifies to "
            f"{slugify(spec['title'])!r}, which does not match the kernel id "
            f"{slug!r}. Kaggle rejects the push. Reword the title so it resolves."
        )
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
        # No kernel_sources: no notebook reads another's output through a Kaggle
        # mount. Every stage reads its inputs from the artefact store
        # (configs/artifacts.yaml), so where Kaggle would have mounted them is moot.
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
