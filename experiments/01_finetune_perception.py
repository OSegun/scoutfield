"""
Roadmap item 1: fine-tune EfficientNet-B0 on PlantVillage.

    python experiments/01_finetune_perception.py --config configs/perception.yaml

Writes ``best.pt``, ``last.pt`` and ``results/perception_summary.json`` and pushes
them to stage ``01_perception`` of the artefact store; ``last.pt`` is pushed every
epoch. Pass ``--resume`` to continue a run whose session died. The ``shift_eval``
accuracy in the summary is this phase's BASE_ACC, replacing the pilot's cited
0.816: it is measured here, not assumed.

Constraints that are not stylistic
----------------------------------
**Checkpoint every epoch.** Kaggle kills sessions. A twelve-epoch run restarting from
scratch because the session dropped at epoch eleven wastes a real fraction of a weekly
GPU quota. Resume must restore optimizer, scheduler *and* RNG states — a resumed run that
reseeds is not a continuation of the run it claims to continue.

**Label smoothing stays at zero.** It is a standard accuracy trick that directly alters
confidence calibration, which is this study's dependent variable. Enabling it would
confound the thing being measured.

**No early stopping on the validation split.** That split fits the temperature; using it
for both leaks the calibration fit into model selection.
"""

from __future__ import annotations

import argparse

from scoutfield.config import load_config
from scoutfield.perception.train import ResumeRefused, train
from scoutfield.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/perception.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="continue from 01_perception/last.pt in the artefact store")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    # Without --resume, existing resume state refuses the run instead of being
    # overwritten; see `train`.
    try:
        summary = train(cfg, resume=args.resume)
    except ResumeRefused as exc:
        raise SystemExit(str(exc)) from exc
    print({k: v for k, v in summary.items() if k not in ("history", "splits")})


if __name__ == "__main__":
    main()
