"""
Roadmap items 2, 3 and 7: calibration.

    python experiments/02_calibrate.py --config configs/perception.yaml

Three stages, in order:

1. **Fit T on the validation split** by minimising NLL with the network frozen
   (Guo et al., 2017). This is the model's actual calibration state and the T = 1
   reference point of the sweep.
2. **Sweep T** to reproduce the pilot's instrument on a real classifier. This is the step
   that tests whether the pilot's effect was an artefact of the Gaussian surrogate.
3. **MC-dropout and deep ensembles** as alternative uncertainty estimates, evaluated on
   the same axis so the three are comparable.

The invariant, restated because it is the whole experiment
-----------------------------------------------------------
Temperature scaling divides logits by a scalar. The link is strictly monotone and the
decision boundary is at logit 0, so accuracy cannot move while confidence does. This
script asserts it and exits non-zero if it fails. That is not a tolerance issue — the
invariance is exact, to floating-point equality of the arg-max. A failure means the
temperature is being applied after a softmax, or the hard prediction is being recomputed
from scaled values.
"""

from __future__ import annotations

import argparse

from scoutfield.config import load_config
from scoutfield.perception.adapter import build_logit_pool
from scoutfield.perception.calibrate import run
from scoutfield.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/perception.yaml")
    # No default path. A literal one is wrong on Kaggle, where notebook 01's
    # checkpoint arrives read-only under /kaggle/input rather than in this session's
    # working directory, and hard-coding it here silently bypasses the search in
    # `utils.paths.find_checkpoint`. None means "find it"; pass the flag to override.
    parser.add_argument("--checkpoint", default=None,
                        help="explicit checkpoint path; searched for when omitted")
    parser.add_argument("--skip-pool", action="store_true",
                        help="skip caching the logit pool the planner sweep samples from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    # `run` asserts accuracy invariance internally and raises if it fails, so a
    # broken instrument exits non-zero here rather than producing a results file.
    out = run(cfg, args.checkpoint)

    for split, summary in out["splits"].items():
        accs = {round(r["accuracy"], 4) for r in summary["sweep"].values()}
        print(f"{split:>6}: accuracy across the sweep = {accs}")

    if not args.skip_pool:
        # `out["checkpoint"]` is the path `run` actually resolved and loaded. Passing
        # `args.checkpoint` here would be None on the default path, and would risk
        # pooling logits from a different checkpoint than the one just calibrated.
        pool = build_logit_pool(out["checkpoint"], split="test", config=cfg)
        print(f"cached logit pool: { {k: len(v) for k, v in pool.items()} }")


if __name__ == "__main__":
    main()
