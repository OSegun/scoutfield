"""
Resumable evaluation sweep. The counterpart of the pilot's run_jobs.py.

    python experiments/04_sweep.py --config configs/sweep.yaml

    # keep going until every job is done, in bounded chunks
    while JOB_BUDGET=60 python experiments/04_sweep.py; do :; done

What runs
---------
Every planner at every temperature of the grid, relative to the shift-fitted
temperature (``reference_condition``), on the PlantDoc ``shift_eval`` pool, plus each
``named_conditions`` entry once at its own T = 1. Every row records its reference
temperature, its condition and the effective divisor it ran at, and the ECE and
signed calibration error of that exact condition, read from stage 02.

Why resumable rather than one long process
-------------------------------------------
Kaggle kills sessions and laptops get closed. The done-registry and the results CSV
live in the artefact store, stage ``04_sweep``: they are pulled on start, pushed every
``push_every_jobs`` jobs and at the end of each invocation, so re-running the same
command on any machine carries on from where the store says the run is. They are
always pushed together, so they cannot fall out of step.

Exit codes: 0 while jobs remain, 1 when every job is done, 2 when the sweep aborted
(for example on a row with no ECE). A shell loop on "exit 0" keeps going until the
sweep finishes or fails, and the two are distinguishable.

Cost note
---------
PPO evaluation is far slower per run than the hand-written baselines and dominates
sweep time. While iterating on the environment, run the baselines alone — a broken
environment is much cheaper to discover in seconds than in an hour.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

from scoutfield.config import load_config
from scoutfield.utils.artifacts import get_store, git_commit, load_settings
from scoutfield.utils.checkpoint import DoneRegistry
from scoutfield.utils.paths import results_dir
from scoutfield.utils.seeding import run_id, seed_everything

INFO_KEYS = ("recall", "precision", "detections_per_joule",
             "false_alarms", "coverage", "time_to_first_detection")

STAGE = "04_sweep"


# ------------------------------------------------------------------- jobs

def build_job_list(cfg) -> list[dict]:
    """Every run the sweep makes, each with the ``run_id`` the registry keys on.

    The grid is agents x temperatures x seeds x cluster sigmas at the reference
    condition, plus agents x seeds x sigmas at T = 1 for each named condition. A
    learned planner has one more axis: which trained policy it is. Every axis that
    changes a result reaches the run id, or a resumed sweep would treat two
    different runs as the same completed job and skip one.
    """
    env_cfg = load_config("configs/ppo_field32.yaml").section("env")
    tau = float(env_cfg["detect_threshold"])
    ppo_seeds = [int(s) for s in cfg.get("ppo_train_seeds", [cfg["seed"]])]
    reference = str(cfg["reference_condition"])

    def jobs_for(condition: str, temperatures) -> list[dict]:
        out = []
        for agent, temperature, seed, sigma in product(
            cfg["agents"], temperatures, cfg["seeds"], cfg["cluster_sigmas"]
        ):
            for ppo_seed in (ppo_seeds if agent == "ppo" else [None]):
                suffix = "" if ppo_seed is None else f"-p{ppo_seed}"
                if condition != reference:
                    suffix += f"__{condition}"
                out.append({
                    "run_id": run_id(agent, seed, temperature, tau, sigma) + suffix,
                    "agent": agent,
                    "seed": int(seed),
                    "temperature": float(temperature),
                    "tau": tau,
                    "sigma": float(sigma),
                    "ppo_train_seed": ppo_seed,
                    "reference_condition": condition,
                })
        return out

    jobs = jobs_for(reference, cfg["temperatures"])
    for condition in cfg.get("named_conditions", []) or []:
        jobs += jobs_for(str(condition), [1.0])
    return jobs


# ------------------------------------------------------------ calibration

_CALIBRATION: dict[str, dict] = {}
_POOLS: dict[tuple[str, str], dict] = {}


def _calibration(store) -> dict:
    """Stage 02's ``calibration_sweep.json``, pulled once per process. Required."""
    if store.run_id not in _CALIBRATION:
        _CALIBRATION[store.run_id] = store.pull_json("02_calibration",
                                                     "calibration_sweep.json")
    return _CALIBRATION[store.run_id]


def _calibration_table(store, pool_split: str, condition: str) -> dict:
    """Measured calibration per temperature for one split and one condition.

    Read rather than recomputed, so perception and planning cannot disagree about
    what a temperature meant. The sweep must exist; a missing one raises.
    """
    from scoutfield.envs.field_env import REFERENCE_CONDITIONS

    if condition not in REFERENCE_CONDITIONS:
        raise KeyError(f"unknown condition {condition!r}")
    cal = _calibration(store)
    for entry in cal.get("sweeps", {}).values():
        if entry.get("split") == pool_split and entry.get("relative_to") == condition:
            return entry["sweep"]
    raise KeyError(f"calibration_sweep.json in run {store.run_id} has no sweep of split "
                   f"{pool_split!r} relative to {condition!r}")


def _pool(store, pool_split: str) -> dict:
    key = (store.run_id, pool_split)
    if key not in _POOLS:
        from scoutfield.perception.adapter import load_or_build_logit_pool

        _POOLS[key] = load_or_build_logit_pool(pool_split, store=store)
    return _POOLS[key]


def _make_agent(name: str, seed: int, cfg, ppo_train_seed: int | None = None, store=None):
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
        policy_seed = cfg["seed"] if ppo_train_seed is None else ppo_train_seed
        return PPOPlanner((store or get_store()).pull("03_ppo", f"ppo_seed{policy_seed}.zip"))
    raise KeyError(f"unknown agent '{name}'")


# -------------------------------------------------------------------- run

def episode_metrics(info: dict) -> dict:
    """The pilot's info keys plus the two secondary metrics.

    Detections per joule counts true positives only, so it favours overconfidence.
    F1 brings precision back in, and false alarms per joule prices the false
    confirmations an overconfident classifier makes. Both per episode, from the
    same ``info`` dict, so they average exactly like the primary metric.
    """
    out = {k: float(info[k]) for k in INFO_KEYS}
    p, r = out["precision"], out["recall"]
    out["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    out["false_alarms_per_joule"] = out["false_alarms"] / max(1e-9, float(info["energy_spent"]))
    return out


def run_job(job: dict, cfg, classifier=None, store=None) -> dict:
    """Execute one evaluation run and return a flat row for the CSV.

    Carries every key the pilot's ``info`` dict has, the secondary metrics, and the
    calibration of the exact condition the run used: reference temperature,
    condition, effective divisor, ECE, signed calibration error and accuracy.
    ``classifier`` is injected by tests; otherwise the pool comes from stage 02.
    """
    from scoutfield.envs.field_env import _reference_temperature, make_field_env
    from scoutfield.perception.adapter import CNNClassifier

    env_config = load_config("configs/ppo_field32.yaml")
    episodes = int(cfg["eval_episodes"])
    pool_split = str(cfg["pool_split"])
    condition = str(job["reference_condition"])

    if classifier is None:
        store = store or get_store()
        reference_temperature = _reference_temperature(condition, store=store)
        calibration = _calibration_table(store, pool_split, condition).get(
            f"{job['temperature']:g}", {})
        # Built once per job: the pool is loaded once per process, and a fresh
        # generator per job keeps each run reproducible from its own seed.
        classifier = CNNClassifier(
            logit_pool=_pool(store, pool_split),
            temperature=job["temperature"],
            reference_temperature=reference_temperature,
            rng=np.random.default_rng(job["seed"]),
            pool_split=pool_split,
        )
    else:
        reference_temperature = float(classifier.reference_temperature)
        calibration = {}

    agent = _make_agent(job["agent"], job["seed"], cfg, job.get("ppo_train_seed"), store)

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
        rows.append(episode_metrics(info))

    row = {**job, "episodes": episodes}
    for key in rows[0]:
        row[key] = float(np.mean([r[key] for r in rows]))
    row["ece"] = calibration.get("ece")
    row["signed_calibration_error"] = calibration.get("signed_calibration_error")
    row["accuracy"] = calibration.get("accuracy")
    # Carried per row, not only in the run metadata: a CSV gets copied, merged and
    # read a year later, and a row that does not say which condition produced it
    # will eventually be compared against one from another condition.
    row["pool_split"] = getattr(classifier, "pool_split", pool_split)
    row["reference_temperature"] = reference_temperature
    row["effective_divisor"] = reference_temperature * float(job["temperature"])
    return row


# ---------------------------------------------------------------- records

def write_run_metadata(cfg, jobs: list[dict], path: Path, store) -> dict:
    """Record how this sweep was produced, beside the results it produced.

    Several conditions are invisible in the results file itself — which split the
    agent observed through, which policies were evaluated, which commit was
    running, which temperatures T = 1 meant — and each has at some point been the
    difference between a correct result and a plausible-looking wrong one.
    """
    from scoutfield.envs.field_env import REFERENCE_CONDITIONS

    env_cfg = load_config("configs/ppo_field32.yaml").section("env")
    cal = _calibration(store)
    conditions = [str(cfg["reference_condition"]), *map(str, cfg.get("named_conditions") or [])]
    meta = {
        "experiment": STAGE,
        "run_id": store.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "pool_split": str(cfg["pool_split"]),
        "reference_condition": str(cfg["reference_condition"]),
        "named_conditions": conditions[1:],
        "reference_temperatures": {c: float(cal[REFERENCE_CONDITIONS[c]]) for c in conditions},
        "perception_checkpoint_sha256": cal.get("checkpoint_sha256"),
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


def _append_row(path: Path, row: dict) -> None:
    """Append one row, writing the header only when creating the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def sync_resume_state(store, stage: str, registry_path: Path, csv_path: Path) -> None:
    """Make the local registry and CSV equal to the store's, or start both afresh.

    The store is the record. When it holds this stage's state, both files are
    pulled over whatever is on disk; when it holds none, any local leftovers belong
    to some other run and are deleted, so rows from two runs can never mix.

    A row is written before its job is marked done, so a crash between the two
    leaves a row for a job the registry does not hold, and that job runs again on
    resume. Rows whose ``run_id`` is not in the registry are therefore dropped here,
    which keeps exactly one row per completed job.
    """
    import shutil

    for local in (registry_path, csv_path):
        if local.exists():
            local.unlink()
    for local in (registry_path, csv_path):
        pulled = store.pull(stage, local.name, required=False)
        if pulled is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pulled, local)

    if csv_path.exists():
        done = set(json.loads(registry_path.read_text(encoding="utf-8"))) \
            if registry_path.exists() else set()
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields, rows = reader.fieldnames, list(reader)
        kept = [r for r in rows if r["run_id"] in done]
        if len(kept) != len(rows):
            print(f"dropped {len(rows) - len(kept)} row(s) of jobs not marked done")
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(kept)


def push_state(store, stage: str, paths: list[Path], message: str) -> None:
    present = [p for p in paths if p.exists()]
    if present:
        store.push(stage, present, message)


# ------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sweep.yaml")
    parser.add_argument("--job-budget", type=int, default=None,
                        help="stop after N jobs; overrides config and $JOB_BUDGET")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    runner = cfg.section("runner")
    push_every = int(load_settings()["push_every_jobs"])
    store = get_store()

    budget = args.job_budget or int(os.environ.get("JOB_BUDGET", runner["job_budget"]))
    registry_path = Path(runner["checkpoint_file"])
    csv_path = Path(runner["output_csv"])
    meta_path = results_dir() / "sweep_metadata.json"
    sync_resume_state(store, STAGE, registry_path, csv_path)
    registry = DoneRegistry(registry_path)

    jobs = build_job_list(cfg)
    meta = write_run_metadata(cfg, jobs, meta_path, store)
    store.record_inputs(STAGE, [("02_calibration", "calibration_sweep.json")])
    print(f"run {store.run_id}: pool_split={meta['pool_split']}, "
          f"reference={meta['reference_condition']} "
          f"T*={meta['reference_temperatures']}, commit={meta['git_commit']}")
    pending = [j for j in jobs if j["run_id"] not in registry]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already done; "
          f"running up to {budget} of the remaining {len(pending)}")

    state = [csv_path, registry_path, meta_path]
    try:
        for i, job in enumerate(pending[:budget], start=1):
            row = run_job(job, cfg, store=store)
            if row["ece"] is None:
                raise RuntimeError(
                    f"{job['run_id']}: no ECE for T={job['temperature']:g} under "
                    f"{job['reference_condition']} on {cfg['pool_split']}. Every row must "
                    f"carry the calibration it ran at; check calibration_sweep.json in "
                    f"stage 02 of run {store.run_id}."
                )
            # Write the row before marking done: the reverse order would let a kill
            # between the two mark a job complete that produced no data.
            _append_row(csv_path, row)
            registry.mark(job["run_id"])
            print(f"[{i}/{min(budget, len(pending))}] {job['run_id']}  "
                  f"det/J {row['detections_per_joule']:.4f}  recall {row['recall']:.3f}")
            if i % push_every == 0:
                done = sum(1 for j in jobs if j["run_id"] in registry)
                push_state(store, STAGE, state, f"{done} jobs done")
    finally:
        # Also on an abort: the rows that did finish are kept.
        push_state(store, STAGE, state, "end of invocation")

    remaining = len([j for j in jobs if j["run_id"] not in registry])
    print(f"{remaining} jobs remaining")
    if remaining:
        raise SystemExit(0)

    from scoutfield.analysis.environment import field_statistics
    from scoutfield.analysis.summary import write_summary

    summary_path = Path(runner["summary_json"])
    summary = write_summary(csv_path, summary_path, cfg)
    env_cfg = load_config("configs/ppo_field32.yaml").section("env")
    summary["environment"] = field_statistics(env_cfg, cfg["cluster_sigmas"],
                                              int(cfg["morisita_seeds"]))
    summary["run_id"] = store.run_id
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    store.push(STAGE, [csv_path, registry_path, meta_path, summary_path], "sweep complete")
    print(f"wrote and pushed {summary_path} from {summary['n_rows']} rows")
    print(f"SWEEP COMPLETE: {len(jobs)} jobs")
    raise SystemExit(1)


if __name__ == "__main__":
    # `while python 04_sweep.py; do :; done` stops on ANY non-zero exit, so a crash
    # must be distinguishable from a finished sweep: 1 = finished, 2 = aborted.
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        import traceback

        traceback.print_exc()
        print(f"\nSWEEP ABORTED, jobs remain: {type(exc).__name__}: {exc}\n"
              f"The finished rows and the done-registry were pushed to the store, so "
              f"fixing the cause and re-running resumes from there.")
        raise SystemExit(2) from exc
