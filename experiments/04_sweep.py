"""
Resumable evaluation sweep. The counterpart of the pilot's run_jobs.py.

    python experiments/04_sweep.py --config configs/sweep.yaml

    # keep going until every job is done, in bounded chunks
    while JOB_BUDGET=60 python experiments/04_sweep.py; do :; done

Why resumable rather than one long process
-------------------------------------------
Kaggle kills sessions and laptops get closed. The pilot learned this expensively and
solved it by checkpointing completed jobs to ``results/_done.json``. Same mechanism here,
via ``scoutfield.utils.checkpoint.DoneRegistry``.

The trap, carried over from the pilot: to restart cleanly, delete *both* the
done-registry and the results CSV. Deleting one means the driver resumes, appends fresh
rows to stale ones, and nothing warns you that the analysis now mixes two code versions.

Cost note
---------
PPO evaluation is roughly thirty times slower per run than the hand-written baselines and
dominates sweep time. While iterating on the environment, run the baselines alone — a
broken environment is much cheaper to discover in seconds than in an hour.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

from scoutfield.config import load_config
from scoutfield.utils.checkpoint import DoneRegistry
from scoutfield.utils.paths import results_dir
from scoutfield.utils.seeding import run_id, seed_everything

INFO_KEYS = ("recall", "precision", "detections_per_joule",
             "false_alarms", "coverage", "time_to_first_detection")

# The experimental condition, in one place so the reported ECE and the observed
# logits cannot come from different splits.
#
# `shift` — PlantDoc field imagery: accuracy 0.8036, ECE 0.1376, overconfident in
# every reliability bin. `test` is in-distribution and, at 0.9998 accuracy with ECE
# 0.0004, leaves a planner nothing to consume: precision sits at exactly 1.000 with
# zero false alarms, so the precision-for-recall trade this study measures cannot
# occur. See docs/RESULTS.md, section "3. Planners" -> "PPO training".
CALIBRATION_SPLIT = "shift"


def build_job_list(cfg) -> list[dict]:
    """Cartesian product of agents x temperatures x seeds x cluster sigmas.

    Each job carries its ``run_id``, which is what the done-registry keys on. If a
    parameter is added that changes results, it must go into the run id too — otherwise a
    resumed sweep treats two different runs as the same completed job and skips one.
    """
    env_cfg = load_config("configs/ppo_field32.yaml").section("env")
    tau = float(env_cfg["detect_threshold"])
    # Which trained PPO policies to evaluate. Listing them rather than defaulting to
    # the config's single `seed` is deliberate: evaluating one policy across five
    # evaluation seeds measures that policy, not the method. Three independently
    # trained policies is what separates "PPO reached this" from "PPO reaches this".
    ppo_seeds = [int(s) for s in cfg.get("ppo_train_seeds", [cfg["seed"]])]

    jobs = []
    for agent, temperature, seed, sigma in product(
        cfg["agents"], cfg["temperatures"], cfg["seeds"], cfg["cluster_sigmas"]
    ):
        # A learned planner has one more axis than a hand-written one: which trained
        # policy it is. That axis must reach the run id, or the done-registry treats
        # three distinct policies as one completed job and silently skips two.
        for ppo_seed in (ppo_seeds if agent == "ppo" else [None]):
            suffix = "" if ppo_seed is None else f"-p{ppo_seed}"
            jobs.append({
                "run_id": run_id(agent, seed, temperature, tau, sigma) + suffix,
                "agent": agent,
                "seed": int(seed),
                "temperature": float(temperature),
                "tau": tau,
                "sigma": float(sigma),
                "ppo_train_seed": ppo_seed,
            })
    return jobs


@functools.lru_cache(maxsize=1)
def _calibration_table() -> dict:
    """Measured ECE per temperature, read rather than recomputed.

    Reading it means perception and planning cannot disagree about what a given
    temperature meant. Returns an empty table when calibration has not been run,
    so a baselines-only sweep is still possible during development — the missing
    columns are then visibly blank rather than quietly invented.
    """
    path = results_dir() / "calibration_sweep.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("splits", {}).get(CALIBRATION_SPLIT, {}).get("sweep", {})


def _make_agent(name: str, seed: int, cfg, ppo_train_seed: int | None = None):
    """Construct one planner by name. All go through the same evaluation path."""
    from agents import GreedyEntropyAgent, LawnmowerAgent, RandomAgent, ReinforceAgent

    from scoutfield.planners.oracle import OraclePlanner
    from scoutfield.planners.ppo import PPOPlanner

    rng = np.random.default_rng(seed)
    if name == "lawnmower":
        return LawnmowerAgent(rng)
    if name == "random":
        return RandomAgent(rng)
    if name == "greedy_entropy":
        return GreedyEntropyAgent(rng)
    if name == "reinforce":
        return ReinforceAgent(rng)
    if name == "oracle":
        return OraclePlanner()
    if name == "ppo":
        # Resolved, not composed from the writable directory: on Kaggle these
        # policies arrive read-only from notebook 03's output under /kaggle/input.
        from scoutfield.utils.paths import find_checkpoint

        policy_seed = cfg["seed"] if ppo_train_seed is None else ppo_train_seed
        return PPOPlanner(find_checkpoint("ppo", f"ppo_seed{policy_seed}.zip"))
    raise KeyError(f"unknown agent '{name}'")


def run_job(job: dict, cfg, classifier=None) -> dict:
    """Execute one evaluation run and return a flat row for the CSV.

    Carries every key the pilot's ``info`` dict has, plus the measured ECE and
    signed calibration error at this temperature.
    """
    from scoutfield.envs.field_env import make_field_env

    env_config = load_config("configs/ppo_field32.yaml")
    episodes = int(cfg["eval_episodes"])

    agent = _make_agent(job["agent"], job["seed"], cfg, job.get("ppo_train_seed"))
    calibration = _calibration_table().get(f"{job['temperature']:g}", {})

    if classifier is None:
        # Built once per job rather than once per episode: the logit pool is
        # loaded from disk, and reloading it 20 times per job would dominate the
        # runtime of every baseline.
        from scoutfield.envs.field_env import _reference_temperature
        from scoutfield.perception.adapter import CNNClassifier
        from scoutfield.utils.paths import find_checkpoint

        # The pool split must match the split `_calibration_table` reads its ECEs
        # from. Pairing shift ECEs with in-distribution logits produces a table
        # that looks entirely reasonable and relates two different conditions.
        classifier = CNNClassifier(
            checkpoint=find_checkpoint("perception", "best.pt"),
            temperature=job["temperature"],
            reference_temperature=_reference_temperature(),
            rng=np.random.default_rng(job["seed"]),
            pool_split=CALIBRATION_SPLIT,
        )

    rows = []
    for episode in range(episodes):
        # Distinct field per episode, identical across agents at the same seed:
        # every planner meets the same set of fields or the comparison is noise.
        env = make_field_env(
            env_config,
            seed=job["seed"] * 10_000 + episode,
            temperature=job["temperature"],
            tau=job["tau"],
            sigma=job["sigma"],
            classifier=classifier,
        )
        obs = env.reset()
        if hasattr(agent, "reset"):
            agent.reset(env)
        done, info = False, {}
        while not done:
            obs, _, done, info = env.step(int(agent.act(obs, env)))
        rows.append(info)

    row = {**job, "episodes": episodes}
    for key in INFO_KEYS:
        row[key] = float(np.mean([r[key] for r in rows]))
    row["ece"] = calibration.get("ece")
    row["signed_calibration_error"] = calibration.get("signed_calibration_error")
    row["accuracy"] = calibration.get("accuracy")
    # Carried per row, not only in the run metadata: a CSV gets copied into a
    # notebook, merged with another, or read a year later, and a row that does not
    # say which condition produced it is a row that will eventually be compared
    # against one from the other condition.
    row["pool_split"] = getattr(classifier, "pool_split", CALIBRATION_SPLIT)
    return row


def write_run_metadata(cfg, jobs: list[dict], path: Path) -> dict:
    """Record how this sweep was produced, beside the results it produced.

    A number is only reproducible if the conditions that made it are recoverable.
    Several of these are invisible in the results file itself — which split the
    agent observed through, which trained policies were evaluated, which commit
    was running — and each has already been, at some point, the difference between
    a correct result and a plausible-looking wrong one.

    Written every invocation. The sweep is resumable and runs in chunks, so the
    file always describes the configuration of the most recent chunk; a mid-sweep
    config change shows up as a changed file rather than as silence.
    """
    from scoutfield.utils.paths import find_checkpoint

    env_cfg = load_config("configs/ppo_field32.yaml").section("env")
    try:
        checkpoint = str(find_checkpoint("perception", "best.pt"))
    except FileNotFoundError:
        checkpoint = None

    meta = {
        "experiment": "04_sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "pool_split": CALIBRATION_SPLIT,
        "perception_checkpoint": checkpoint,
        "ppo_train_seeds": sorted({j["ppo_train_seed"] for j in jobs
                                   if j.get("ppo_train_seed") is not None}),
        "evaluation_seeds": [int(x) for x in cfg["seeds"]],
        "agents": list(cfg["agents"]),
        "temperatures": [float(t) for t in cfg["temperatures"]],
        "cluster_sigmas": [float(x) for x in cfg["cluster_sigmas"]],
        "tau": float(env_cfg["detect_threshold"]),
        "eval_episodes": int(cfg["eval_episodes"]),
        "grid": env_cfg["grid"],
        "budget": env_cfg["budget"],
        "n_jobs": len(jobs),
        "configs": {"sweep": str(cfg.path), "env": "configs/ppo_field32.yaml"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _git_commit() -> str | None:
    """The commit this ran from, or None outside a checkout."""
    import subprocess

    # check=False on purpose: outside a checkout git exits non-zero, and a missing
    # commit id is recorded as None rather than aborting a sweep that is otherwise
    # perfectly runnable. Provenance is worth recording, not worth failing over.
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _append_row(path: Path, row: dict) -> None:
    """Append one row, writing the header only when creating the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sweep.yaml")
    parser.add_argument("--job-budget", type=int, default=None,
                        help="stop after N jobs; overrides config and $JOB_BUDGET")
    parser.add_argument("--reset", action="store_true",
                        help="clear the done-registry AND the results CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    runner = cfg.section("runner")

    budget = args.job_budget or int(os.environ.get("JOB_BUDGET", runner["job_budget"]))
    registry = DoneRegistry(runner["checkpoint_file"])
    csv_path = Path(runner["output_csv"])
    if args.reset:
        registry.reset()
        if csv_path.exists():
            # Deleting both together, so the pair can never fall out of step and
            # silently mix two code versions into one analysis.
            csv_path.unlink()
            print(f"cleared registry and {csv_path}")
        else:
            print("registry cleared")

    jobs = build_job_list(cfg)
    meta = write_run_metadata(cfg, jobs, results_dir() / "sweep_metadata.json")
    print(f"condition: pool_split={meta['pool_split']}, "
          f"ppo_train_seeds={meta['ppo_train_seeds']}, commit={meta['git_commit']}")
    pending = [j for j in jobs if j["run_id"] not in registry]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already done; "
          f"running up to {budget} of the remaining {len(pending)}")

    for i, job in enumerate(pending[:budget], start=1):
        row = run_job(job, cfg)
        # Write the row before marking done: the reverse order would let a kill
        # between the two mark a job complete that produced no data.
        _append_row(csv_path, row)
        registry.mark(job["run_id"])
        print(f"[{i}/{min(budget, len(pending))}] {job['run_id']}  "
              f"det/J {row['detections_per_joule']:.4f}  recall {row['recall']:.3f}")

    remaining = len([j for j in jobs if j["run_id"] not in registry])
    print(f"{remaining} jobs remaining")

    if not remaining and csv_path.exists():
        from scoutfield.analysis.summary import write_summary

        summary = write_summary(csv_path, runner["summary_json"], cfg)
        print(f"wrote {runner['summary_json']} from {summary['n_rows']} rows")
    # Exit 0 while work remains, 1 when finished, so the shell while-loop in the
    # module docstring keeps going and then stops on its own. An unhandled error
    # exits 2 — see the wrapper below for why that distinction is load-bearing.
    if remaining:
        raise SystemExit(0)
    print(f"SWEEP COMPLETE: {len(jobs)} jobs")
    raise SystemExit(1)


if __name__ == "__main__":
    # `while python 04_sweep.py; do :; done` stops on ANY non-zero exit, so a crash
    # is indistinguishable from a finished sweep: the loop exits quietly, the
    # notebook proceeds to the figures, and the only symptom is a summary that was
    # never written. That happened once. It now says so.
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        import traceback

        traceback.print_exc()
        print(f"\nSWEEP ABORTED, jobs remain: {type(exc).__name__}: {exc}\n"
              f"The done-registry is intact, so fixing the cause and re-running "
              f"resumes from here. Do NOT delete results/_done.json — that discards "
              f"every completed job.")
        raise SystemExit(2) from exc
