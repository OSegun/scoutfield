"""
Roadmap items 2, 3 and 7: calibration.

    python experiments/02_calibrate.py --config configs/perception.yaml

Reads the stage 01 checkpoint from the artefact store, then:

1. **Fits two temperatures** by minimising NLL with the network frozen (Guo et al.,
   2017): the lab temperature on PlantVillage validation and the shift temperature
   on the PlantDoc ``shift_fit`` half.
2. **Sweeps T** relative to them: ``val`` and ``test`` relative to the lab
   temperature, ``shift_eval`` relative to the shift temperature (the condition
   the planner sweep uses) and relative to the lab temperature (the lab-calibrated
   condition).
3. **Builds the logit pools** for ``shift_fit``, ``shift_eval`` and ``test``, each
   carrying the membership hash of its split.

Everything is pushed to stage ``02_calibration``.

The invariant, restated because it is the whole experiment
-----------------------------------------------------------
Temperature scaling divides logits by a scalar. The link is strictly monotone and the
decision boundary is at logit 0, so accuracy cannot move while confidence does. The
run asserts it on every sweep and exits non-zero if it fails. It also asserts that on
``shift_eval`` the signed calibration error is positive below T = 1 and negative
above it; if not, the sweep is not centred on the calibrated point and the run stops.
"""

from __future__ import annotations

import argparse

from scoutfield.config import load_config
from scoutfield.perception.calibrate import run
from scoutfield.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/perception.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="a local checkpoint to use instead of the stage 01 one "
                             "in the store")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    # `run` asserts accuracy invariance and the sign change, and raises if either
    # fails, so a broken instrument exits non-zero rather than pushing a file.
    out = run(cfg, args.checkpoint)

    for name, entry in out["sweeps"].items():
        accs = {round(r["accuracy"], 6) for r in entry["sweep"].values()}
        print(f"{name:>15} (relative to {entry['relative_to']}): accuracy {accs}")
    for split, pool in out["pools"].items():
        print(f"pool {split:>10}: {pool['n_by_class']}")


if __name__ == "__main__":
    main()
