"""
Roadmap item 6: sensitivity of the effect size to the confirmation threshold tau.

    python experiments/05_tau_sensitivity.py --config configs/sweep.yaml

Why this blocks a headline number
----------------------------------
From the pilot write-up, chapter "4.10.3 Threats to validity":

    "The confirmation threshold tau = 0.75 interacts with the temperature manipulation by
    construction. A sensitivity analysis over tau is required before the effect size can
    be quoted as a general magnitude rather than a magnitude at this threshold."

The interaction is structural, not incidental. A detection is confirmed when posterior
belief exceeds tau. Temperature scaling moves reported confidence, so it moves how often
belief crosses any fixed threshold. Raising T pushes probabilities toward 0.5 and makes
crossing tau = 0.75 rarer; lowering T pushes them to the extremes and makes it commoner.
Some portion of the pilot's 13.4x collapse is therefore attributable to the threshold
rather than to miscalibration itself, and nobody currently knows how much.

Until this runs, 13.4x may be quoted only as "at tau = 0.75".

The output is a surface, not a corrected number
------------------------------------------------
Detections-per-joule as a function of (tau, T), per planner. Three outcomes, each meaning
something different:

  * The effect persists at every tau — the finding generalises; report the surface and
    quote the range.
  * The effect vanishes at some tau — the pilot's headline was threshold-specific. That
    is a substantial correction and gets reported as one.
  * There is an optimal tau that varies with T — then tau should adapt to measured
    calibration, which is a practical design recommendation and arguably a more useful
    result than the original.

Cost control
------------
The full grid is |tau| x |T| x |seeds| x |agents|, several times the main sweep. Run
baselines across the whole grid and PPO on a reduced tau set, and say so explicitly in the
write-up rather than leaving the asymmetry for a reader to notice.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scoutfield.analysis.stats import iqm, stratified_bootstrap_ci
from scoutfield.config import load_config
from scoutfield.utils.artifacts import get_store, load_settings
from scoutfield.utils.checkpoint import DoneRegistry
from scoutfield.utils.paths import results_dir
from scoutfield.utils.seeding import run_id, seed_everything

STAGE = "05_tau"


def _sweep_driver():
    """The sibling ``04_sweep.py``, loaded by path: ``experiments/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "sweep_driver", Path(__file__).with_name("04_sweep.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_tau_jobs(cfg, agents=None) -> list[dict]:
    """Every tau-sweep run, at the reference condition.

    Baselines run every cluster scale, so each (agent, tau) cell holds several runs
    per seed and the stratified bootstrap has something to resample. PPO runs the
    single median scale on a reduced tau grid, because its evaluation is far slower;
    its intervals are therefore pooled rather than stratified, and recorded as such.
    """
    tau_cfg = cfg.section("tau_sensitivity")
    agents = agents or list(cfg["agents"])
    all_sigmas = [float(s) for s in cfg["cluster_sigmas"]]
    median_sigma = all_sigmas[len(all_sigmas) // 2]
    condition = str(cfg["reference_condition"])

    jobs = []
    for agent in agents:
        grid = tau_cfg["ppo_tau_grid"] if agent == "ppo" else tau_cfg["tau_grid"]
        sigmas = [median_sigma] if agent == "ppo" else all_sigmas
        for tau in grid:
            for temperature in cfg["temperatures"]:
                for seed in cfg["seeds"]:
                    for sigma in sigmas:
                        jobs.append({
                            "run_id": run_id(agent, seed, temperature, tau, sigma),
                            "agent": agent,
                            "seed": int(seed),
                            "temperature": float(temperature),
                            "tau": float(tau),
                            "sigma": sigma,
                            "ppo_train_seed": None,
                            "reference_condition": condition,
                        })
    return jobs


def run_tau_sweep(cfg, agents=None, job_budget: int | None = None, store=None) -> Path:
    """Sweep tau x temperature and write results/tau_sensitivity.csv.

    Resumable through the artefact store, stage ``05_tau``, exactly like the main
    sweep: state is pulled on start and pushed every ``push_every_jobs`` jobs and
    at the end. A row without an ECE aborts the sweep.
    """
    sweep = _sweep_driver()
    store = store or get_store()
    push_every = int(load_settings()["push_every_jobs"])
    csv_path = results_dir() / "tau_sensitivity.csv"
    registry_path = results_dir() / "_done_tau.json"
    sweep.sync_resume_state(store, STAGE, registry_path, csv_path)
    registry = DoneRegistry(registry_path)
    store.record_inputs(STAGE, [("02_calibration", "calibration_sweep.json")])

    jobs = build_tau_jobs(cfg, agents)
    pending = [j for j in jobs if j["run_id"] not in registry]
    todo = pending[:job_budget or len(pending)]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} tau jobs done; running {len(todo)}")

    try:
        for i, job in enumerate(todo, start=1):
            row = sweep.run_job(job, cfg, store=store)
            if row["ece"] is None:
                raise RuntimeError(f"{job['run_id']}: no ECE for this condition; check "
                                   f"stage 02 of run {store.run_id}")
            sweep._append_row(csv_path, row)
            registry.mark(job["run_id"])
            print(f"  tau={job['tau']:.2f} T={job['temperature']:.2f} "
                  f"{job['agent']:<14} det/J {row['detections_per_joule']:.4f}")
            if i % push_every == 0:
                sweep.push_state(store, STAGE, [csv_path, registry_path], f"{i} tau jobs")
    finally:
        sweep.push_state(store, STAGE, [csv_path, registry_path], "end of invocation")
    return csv_path


def effect_size_by_tau(results, low: float = 1.0, high: float = 4.0) -> dict:
    """Detections-per-joule ratio between T=1 and T=4, per tau and per agent.

    The direct generalisation of the pilot's 13.4x. Reported with the IQM and a
    bootstrap CI at every tau, never as a bare point estimate — the reason this
    analysis exists is that a single number overstated its own generality.

    ``results`` is an iterable of row dicts, as written by the sweep driver.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for row in results:
        key = (row["agent"], float(row["tau"]))
        grouped[key][float(row["temperature"])].append(
            (int(row["seed"]), float(row["detections_per_joule"]))
        )

    out = {}
    for (agent, tau), by_temp in sorted(grouped.items()):
        if low not in by_temp or high not in by_temp:
            continue
        lo_vals = np.array([v for _, v in by_temp[low]], dtype=float)
        hi_vals = np.array([v for _, v in by_temp[high]], dtype=float)
        strata = np.array([s for s, _ in by_temp[low]])

        if lo_vals.size < 4 or hi_vals.size < 4:
            # IQM needs four samples; reporting a ratio from fewer would be a
            # point estimate dressed as a robust one.
            continue

        lo_iqm, hi_iqm = iqm(lo_vals), iqm(hi_vals)
        entry = {
            "agent": agent,
            "tau": tau,
            f"iqm_T{low:g}": lo_iqm,
            f"iqm_T{high:g}": hi_iqm,
            "ratio": (lo_iqm / hi_iqm) if hi_iqm > 0 else float("inf"),
            "n": int(lo_vals.size),
        }
        # Stratify by seed where the design supplies replicates within a seed, and
        # fall back to a pooled bootstrap where it does not — PPO runs one cluster
        # scale, so its strata are singletons. The fallback is recorded per entry
        # rather than applied silently: an interval computed a different way must
        # say so, or the caveat is lost the moment the number is quoted.
        entry["ci_stratified_by"] = "seed"
        try:
            entry["ci_T1"] = stratified_bootstrap_ci(lo_vals, strata, resamples=2000)
        except ValueError as exc:
            if "one member" not in str(exc):
                entry["ci_T1"] = None
                entry["ci_stratified_by"] = None
            else:
                entry["ci_T1"] = stratified_bootstrap_ci(
                    lo_vals, strata, resamples=2000, allow_unstratified=True
                )
                entry["ci_stratified_by"] = "pooled (no within-seed replicates)"
        out[f"{agent}@tau{tau:g}"] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sweep.yaml")
    parser.add_argument("--agents", nargs="*", default=None,
                        help="restrict to these planners; PPO is expensive")
    parser.add_argument("--job-budget", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    store = get_store()
    csv_path = run_tau_sweep(cfg, agents=args.agents, job_budget=args.job_budget,
                             store=store)

    registry = DoneRegistry(results_dir() / "_done_tau.json")
    remaining = [j for j in build_tau_jobs(cfg, args.agents) if j["run_id"] not in registry]
    if remaining:
        print(f"{len(remaining)} tau jobs remain; run again to continue")
        raise SystemExit(0)

    import csv as _csv

    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))

    summary = effect_size_by_tau(rows)
    out = results_dir() / "tau_sensitivity_summary.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    store.push(STAGE, [csv_path, results_dir() / "_done_tau.json", out], "tau sweep complete")

    for key, entry in sorted(summary.items()):
        print(f"{key:<28} ratio T1/T4 = {entry['ratio']:.2f}  (n={entry['n']})")
    print(f"wrote and pushed {out}")


if __name__ == "__main__":
    main()
