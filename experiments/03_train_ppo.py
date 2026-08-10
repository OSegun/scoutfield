"""
Roadmap items 4 and 5: PPO on the 32x32 field.

    python experiments/03_train_ppo.py --config configs/ppo_field32.yaml --seed 0

Why PPO rather than the pilot's REINFORCE
------------------------------------------
REINFORCE was chosen in the pilot for one reason: it needs no deep learning framework, so
it runs on a bare NumPy kernel. That bought reproducibility at the cost of sample
efficiency, and it is one of three named causes of the learned agent losing to a fixed
sweep at every temperature. PPO's clipped objective and value baseline cut gradient
variance, which is what a reward dominated by rare detection events needs.

The other two causes are addressed by the *environment*, not the algorithm — a global
belief map in the observation, and a field scale where selectivity is worth something.
Worth being explicit about: attributing an improvement to PPO when it came from the
observation change would be wrong, which is why `env.observation.global_belief_map` is an
ablation switch rather than a hard-coded choice.

Verify the coverage ceiling first
----------------------------------
This script prints the ceiling before training. If it is not near 0.40 the budget in the
config is wrong and the run will reproduce the pilot's problem — a field so cheap to cover
that a lawnmower sweep is near-optimal by construction.

Honesty requirement: if PPO also loses, that gets reported the same way the pilot reported
its loss, with a stated reason. The comparison is only worth anything if the answer was
allowed to be negative.
"""

from __future__ import annotations

import argparse

from scoutfield.config import load_config
from scoutfield.envs.field_env import coverage_ceiling
from scoutfield.planners.ppo import train_ppo
from scoutfield.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo_field32.yaml")
    parser.add_argument("--seed", type=int, default=None, help="overrides the config seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    seed_everything(seed)

    env = cfg.section("env")
    ceiling = coverage_ceiling(env["grid"], env["budget"],
                               env["energy"]["hover"], env["energy"]["translate"])
    print(f"coverage ceiling (optimistic bound): {ceiling:.3f}")
    if not 0.30 <= ceiling <= 0.55:
        raise SystemExit(
            f"ceiling {ceiling:.3f} is outside the intended regime — retune env.budget "
            "in the config before training, or this run repeats the pilot's problem"
        )

    ckpt = train_ppo(cfg, seed=seed)
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
