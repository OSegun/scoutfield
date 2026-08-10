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
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scoutfield.analysis.stats import iqm, stratified_bootstrap_ci
from scoutfield.config import load_config
from scoutfield.utils.checkpoint import DoneRegistry
from scoutfield.utils.paths import results_dir
from scoutfield.utils.seeding import run_id, seed_everything

TAU_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.85, 0.90]

# PPO evaluation is ~30x slower than the baselines, so it runs on a reduced tau
# set. The asymmetry is deliberate and is stated here so the write-up reports it
# rather than leaving a reader to notice the gap.
PPO_TAU_GRID = [0.60, 0.75, 0.90]


def run_tau_sweep(cfg, agents=None, job_budget: int | None = None) -> Path:
    """Sweep tau x temperature and write results/tau_sensitivity.csv.

    Resumable on the same mechanism as the main sweep: this grid is several times
    larger and will not finish inside one Kaggle session.
    """
    # Imported here rather than at module scope: `experiments/` is not a package,
    # so the sibling driver is loaded by path.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sweep_driver", Path(__file__).with_name("04_sweep.py")
    )
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)

    agents = agents or list(cfg["agents"])
    csv_path = results_dir() / "tau_sensitivity.csv"
    registry = DoneRegistry(results_dir() / "_done_tau.json")

    jobs = []
    for agent in agents:
        grid = PPO_TAU_GRID if agent == "ppo" else TAU_GRID
        for tau in grid:
            for temperature in cfg["temperatures"]:
                for seed in cfg["seeds"]:
                    sigma = float(cfg["cluster_sigmas"][len(cfg["cluster_sigmas"]) // 2])
                    jobs.append({
                        "run_id": run_id(agent, seed, temperature, tau, sigma),
                        "agent": agent,
                        "seed": int(seed),
                        "temperature": float(temperature),
                        "tau": float(tau),
                        "sigma": sigma,
                    })

    pending = [j for j in jobs if j["run_id"] not in registry]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} tau jobs done; "
          f"running {min(job_budget or len(pending), len(pending))}")

    for job in pending[:job_budget or len(pending)]:
        row = sweep.run_job(job, cfg)
        sweep._append_row(csv_path, row)
        registry.mark(job["run_id"])
        print(f"  tau={job['tau']:.2f} T={job['temperature']:.2f} "
              f"{job['agent']:<14} det/J {row['detections_per_joule']:.4f}")

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
        try:
            entry["ci_T1"] = stratified_bootstrap_ci(lo_vals, strata, resamples=2000)
        except ValueError:
            entry["ci_T1"] = None
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

    csv_path = run_tau_sweep(cfg, agents=args.agents, job_budget=args.job_budget)

    import csv as _csv

    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))

    summary = effect_size_by_tau(rows)
    out = results_dir() / "tau_sensitivity_summary.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    for key, entry in sorted(summary.items()):
        print(f"{key:<28} ratio T1/T4 = {entry['ratio']:.2f}  (n={entry['n']})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
