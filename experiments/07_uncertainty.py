"""
Roadmap item 7: temperature scaling, MC-dropout and a deep ensemble on one axis.

    python experiments/07_uncertainty.py train --member 1    # then 2, 3, 4 (GPU)
    python experiments/07_uncertainty.py evaluate            # GPU; needs all members

``train --member k`` fine-tunes ensemble member k exactly as stage 01 fine-tuned
``best.pt`` (member 0): same code, hyperparameters and PlantVillage split, with the
seed ``uncertainty.member_seeds[k - 1]`` from ``configs/perception.yaml``. Its files
go to stage ``07_uncertainty``: ``member<k>_best.pt``, ``member<k>_last.pt`` (pushed
every epoch) and ``member<k>_summary.json``. Run it again after a Kaggle session
dies: a finished member is skipped and an unfinished one resumes from its last
epoch.

``evaluate`` scores the three methods on PlantDoc ``shift_eval`` and PlantVillage
``test``, each raw and with its own fitted temperature, checks that temperature
scaling reproduces stage 02, and pushes ``uncertainty_comparison.json`` to stage
``07_uncertainty``. Figure 16 is then drawn by ``experiments/make_figures.py``.
The method is in ``docs/METHOD.md``, section "2. Perception", subsection
"Uncertainty methods".
"""

from __future__ import annotations

import argparse

from scoutfield.config import load_config


def train_member(config, member: int, store=None) -> str:
    """Train, resume or skip one member, from what the store already holds."""
    from scoutfield.perception.train import run_target, train
    from scoutfield.utils.artifacts import get_store

    store = store or get_store()
    target = run_target(config, member)
    if store.has(target["stage"], target["summary"]):
        print(f"member {member} is already trained ({target['stage']}/{target['summary']} "
              f"in run {store.run_id}); skipped")
        return "skipped"
    resume = store.has(target["stage"], target["last"])
    print(f"member {member}, seed {target['seed']}: "
          + ("resuming from its last epoch" if resume else "training from the start"))
    train(config, resume=resume, store=store, member=member)
    return "resumed" if resume else "trained"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--member", type=int, default=None,
                        help="ensemble member to train (1 to 4); member 0 is stage 01")
    parser.add_argument("--config", default="configs/perception.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.action == "train":
        if args.member is None:
            parser.error("train needs --member <k>")
        train_member(config, args.member)
    else:
        from scoutfield.perception.uncertainty import run
        from scoutfield.utils.seeding import seed_everything

        seed_everything(int(config["seed"]))
        run(config)


if __name__ == "__main__":
    main()
