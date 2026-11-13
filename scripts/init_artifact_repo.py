"""
Write the dataset card (README.md) of the artefact store on Hugging Face.

    python scripts/init_artifact_repo.py --dry-run   # print the card, upload nothing
    python scripts/init_artifact_repo.py             # upload it (needs a write token)

Run once, before the first pipeline run. Re-running replaces the card.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoutfield.utils.artifacts import STAGES, HubBackend, load_settings

CARD = """---
license: mit
pretty_name: scoutfield artefacts
tags:
  - calibration
  - reinforcement-learning
  - informative-path-planning
  - plant-disease
---

# scoutfield artefacts

Every output of every stage of the [scoutfield](https://github.com/{code_repo})
pipeline: model checkpoints, logit pools, calibration sweeps, planner sweeps, the
tau sensitivity analysis and the figures. The pipeline's notebooks read their inputs
from this repository and write their outputs to it, so each result here is the exact
file the next stage consumed.

## Layout

```
runs/<run_id>/manifest.json
runs/<run_id>/<stage>/<file>
```

One `run_id` per complete run of the pipeline. Stages, in order:

{stage_list}

`manifest.json` records, for every stage: each file's SHA-256 and size, the commit of
the code repository that produced it, the commit of this repository it was uploaded
in, the Kaggle kernel when there was one, a UTC timestamp, and the upstream files the
stage read, each with its SHA-256 and upload commit. A file uploaded but not listed
in the manifest belongs to an interrupted stage and should not be used.

## Reading it

Any single file opens in a browser, no login needed:

```
https://huggingface.co/datasets/{repo}/resolve/main/runs/<run_id>/<stage>/<file>
```

A whole run, checked against the manifest, from a clone of the code repository:

```
python scripts/pull_artifacts.py --run <run_id>
```

Or one file from Python:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("{repo}", "runs/<run_id>/04_sweep/summary.json",
                       repo_type="dataset")
```

## Licence

The files in this repository are released under the MIT licence, as is the code that
produced them. The image datasets they were computed from are not redistributed here
and remain under their own terms.

## Data sources

The perception model is fine-tuned on PlantVillage and evaluated under distribution
shift on PlantDoc:

- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant
  health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*.
- Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc:
  A dataset for visual plant disease detection. *Proceedings of the 7th ACM IKDD CoDS
  and 25th COMAD*, 249–253. https://doi.org/10.1145/3371158.3371196
"""


def render(settings: dict) -> str:
    from scoutfield.config import load_config
    from scoutfield.utils.paths import repo_root

    code_repo = load_config(repo_root() / "configs" / "kaggle.yaml")["repo"]
    stage_list = "\n".join(f"- `{s}`" for s in STAGES)
    return CARD.format(repo=settings["hub_repo"], code_repo=code_repo, stage_list=stage_list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the card only")
    args = parser.parse_args()

    settings = load_settings()
    card = render(settings)
    if args.dry_run:
        print(card)
        return
    backend = HubBackend(settings["hub_repo"], settings.get("repo_type", "dataset"))
    commit = backend.upload({"README.md": card.encode("utf-8")}, "Dataset card")
    print(f"wrote the dataset card to {backend.description} (commit {commit[:10]})")


if __name__ == "__main__":
    main()
