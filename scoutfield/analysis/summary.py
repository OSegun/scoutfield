"""
Aggregating sweep rows into ``results/summary.json``.

Why a summary file exists at all
--------------------------------
The pilot's fifth invariant: "Paper numbers come from ``summary.json``. Never
hard-code a result into ``paper_part*.py``." Every figure and every document
reads from this one generated file, so text, figures and data cannot drift apart.
The failure mode it prevents — a document quoting a number the code no longer
produces — is silent, and it reached the pilot's `.docx` once already.

Estimator choice is not free here. Per Agarwal et al. (2021) and Colas et al.
(2018, 2019), the interquartile mean with a stratified bootstrap interval is used
throughout, never mean ± std over a handful of seeds. Stratification is by seed,
because runs sharing a seed share a field layout and a classifier draw and are
therefore not independent.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scoutfield.analysis.stats import iqm, probability_of_improvement, stratified_bootstrap_ci

METRICS = ("detections_per_joule", "recall", "precision",
           "coverage", "false_alarms", "time_to_first_detection")

#: Sweep columns that are labels, not measurements, and must not be coerced to
#: float. Every other column is required to be numeric — see ``read_rows``.
TEXT_COLUMNS = ("agent", "run_id", "pool_split")


def read_rows(csv_path: str | Path) -> list[dict]:
    """Read a sweep CSV, coercing the numeric columns."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"no sweep results at {path}; run experiments/04_sweep.py")
    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        for key, value in list(row.items()):
            if key in TEXT_COLUMNS:
                continue
            if value in ("", None):
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except ValueError as exc:
                # Adding a label column to the sweep CSV used to fail here, deep in
                # aggregation and long after the sweep had finished — the jobs were
                # intact but the summary was never written. Say what to do instead.
                raise ValueError(
                    f"column '{key}' holds the non-numeric value {value!r}. If it is a "
                    f"label rather than a measurement, add it to TEXT_COLUMNS in "
                    f"scoutfield/analysis/summary.py."
                ) from exc
    return rows


def _aggregate(values: np.ndarray, strata: np.ndarray, resamples: int, ci: float) -> dict:
    """IQM plus a stratified bootstrap interval, or the mean when too few samples."""
    if values.size >= 4:
        point = iqm(values)
        lo, hi = stratified_bootstrap_ci(values, strata, resamples=resamples, ci=ci)
        estimator = "iqm"
    else:
        # Below four samples the IQM is undefined. Report the mean and say so,
        # rather than silently switching estimators inside one table.
        point, lo, hi = float(values.mean()), float("nan"), float("nan")
        estimator = "mean"
    return {"estimator": estimator, "value": point, "ci_low": lo, "ci_high": hi,
            "n": int(values.size)}


def build_summary(csv_path: str | Path, config=None) -> dict[str, Any]:
    """Aggregate a sweep CSV into the structure every figure and document reads."""
    rows = read_rows(csv_path)
    stats_cfg = (config.section("statistics") if config is not None else {})
    resamples = int(stats_cfg.get("bootstrap_resamples", 10_000))
    ci = float(stats_cfg.get("ci", 0.95))

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["agent"], row["temperature"])].append(row)

    cells: dict[str, Any] = {}
    for (agent, temperature), group in sorted(by_cell.items()):
        strata = np.array([r["seed"] for r in group])
        entry = {"agent": agent, "temperature": temperature, "n_runs": len(group)}
        for metric in METRICS:
            values = np.array([r[metric] for r in group if r.get(metric) is not None],
                              dtype=float)
            if values.size:
                entry[metric] = _aggregate(values, strata[:values.size], resamples, ci)
        # Calibration numbers are carried through, never recomputed here, so the
        # planner table and the calibration file cannot disagree.
        for key in ("ece", "signed_calibration_error", "accuracy"):
            vals = [r[key] for r in group if r.get(key) is not None]
            entry[key] = float(np.mean(vals)) if vals else None
        cells[f"{agent}@T{temperature:g}"] = entry

    return {
        "cells": cells,
        "agents": sorted({r["agent"] for r in rows}),
        "temperatures": sorted({r["temperature"] for r in rows}),
        "degradation": _degradation(by_cell),
        "rankings": _rankings(by_cell),
        "n_rows": len(rows),
        "source_csv": str(Path(csv_path)),
        "estimator": "iqm + stratified bootstrap (Agarwal et al. 2021)",
    }


def _degradation(by_cell, low: float = 1.0, high: float = 4.0) -> dict:
    """Per-agent detections-per-joule ratio between the calibrated point and T=4.

    This is the quantity the pilot reported as 13.4x for Lawnmower and 1.84x for
    GreedyEntropy.
    """
    out = {}
    agents = {agent for agent, _ in by_cell}
    for agent in sorted(agents):
        lo_rows = by_cell.get((agent, low))
        hi_rows = by_cell.get((agent, high))
        if not lo_rows or not hi_rows:
            continue
        lo = np.array([r["detections_per_joule"] for r in lo_rows], dtype=float)
        hi = np.array([r["detections_per_joule"] for r in hi_rows], dtype=float)
        lo_p = iqm(lo) if lo.size >= 4 else float(lo.mean())
        hi_p = iqm(hi) if hi.size >= 4 else float(hi.mean())
        out[agent] = {
            f"T{low:g}": lo_p,
            f"T{high:g}": hi_p,
            "ratio": (lo_p / hi_p) if hi_p > 0 else None,
            "probability_of_improvement_T1_over_T4": probability_of_improvement(lo, hi),
        }
    return out


def _rankings(by_cell) -> dict:
    """Agent ordering by detections-per-joule at each temperature.

    The pilot's headline included a *ranking reversal* at T = 4, so the ordering
    is recorded explicitly rather than left to be eyeballed off a chart.
    """
    out = {}
    temperatures = sorted({t for _, t in by_cell})
    for temperature in temperatures:
        scored = []
        for (agent, t), group in by_cell.items():
            if t != temperature:
                continue
            vals = np.array([r["detections_per_joule"] for r in group], dtype=float)
            scored.append((agent, iqm(vals) if vals.size >= 4 else float(vals.mean())))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        out[f"T{temperature:g}"] = [{"agent": a, "detections_per_joule": v}
                                    for a, v in scored]
    return out


def write_summary(csv_path: str | Path, out_path: str | Path, config=None) -> dict:
    """Build the summary and write it, returning what was written."""
    summary = build_summary(csv_path, config)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary
