"""
Roadmap item 1: fine-tune EfficientNet-B0 on PlantVillage.

    python experiments/01_finetune_perception.py --config configs/perception.yaml

Writes a checkpoint and ``results/perception_summary.json``. The accuracy in that
summary becomes BASE_ACC for this phase, replacing the pilot's cited 0.816 — it is
measured here, not assumed.

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
from scoutfield.perception.train import train
from scoutfield.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/perception.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the last epoch checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    summary = train(cfg)
    print(summary)


if __name__ == "__main__":
    main()
