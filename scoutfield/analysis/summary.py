"""
Aggregating sweep rows into ``results/summary.json``.

Why a summary file exists at all
--------------------------------
A rule inherited from the pilot: every reported number comes from ``summary.json``
and is never hard-coded into a document. Every figure and every document reads
from this one generated file, so text, figures and data cannot drift apart. The
failure mode it prevents — a document quoting a number the code no longer
produces — is silent, and it happened in the pilot once already.

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

from scoutfield.analysis.stats import (
    iqm,
    probability_of_improvement,
    stratified_bootstrap_ci,
    stratified_bootstrap_ratio_ci,
)

#: Detections per joule is the primary metric, as in the pilot. It counts only
#: true positives, so it rewards overconfidence; F1 and false alarms per joule are
#: the secondary metrics that show the cost (docs/METHOD.md, "Metrics").
METRICS = ("detections_per_joule", "recall", "precision", "f1",
           "coverage", "false_alarms", "false_alarms_per_joule",
           "time_to_first_detection")

#: Sweep columns that are labels, not measurements, and must not be coerced to
#: float. Every other column is required to be numeric — see ``read_rows``.
TEXT_COLUMNS = ("agent", "run_id", "pool_split", "reference_condition")

#: The pilot's spans, used when a caller supplies no config. The sweep config's
#: ``degradation_spans`` is the source of truth.
DEFAULT_SPANS = {"full": [0.30, 4.00], "overconfident": [0.30, 1.00],
                 "underconfident": [1.00, 4.00]}


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
                # A label column added to the sweep CSV would otherwise fail deep in
                # aggregation, long after the sweep had finished, with the jobs
                # intact but no summary written. Say what to do instead.
                raise ValueError(
                    f"column '{key}' holds the non-numeric value {value!r}. If it is a "
                    f"label rather than a measurement, add it to TEXT_COLUMNS in "
                    f"scoutfield/analysis/summary.py."
                ) from exc
    return rows


def _aggregate(values: np.ndarray, strata: np.ndarray, resamples: int, ci: float) -> dict:
    """IQM plus a stratified bootstrap interval, or the mean when too few samples."""
    stratified_by = "seed"
    if values.size >= 4:
        point = iqm(values)
        try:
            lo, hi = stratified_bootstrap_ci(values, strata, resamples=resamples, ci=ci)
        except ValueError as exc:
            # A cell holding one run per seed has no within-stratum replicate to
            # draw, so resampling within strata would return a zero-width interval.
            # Pool instead, and record that the interval was computed a different
            # way: an unlabelled interval that means something else is worse than
            # no interval at all.
            if "one member" not in str(exc):
                raise
            lo, hi = stratified_bootstrap_ci(values, strata, resamples=resamples, ci=ci,
                                             allow_unstratified=True)
            stratified_by = "pooled (no within-seed replicates)"
        estimator = "iqm"
    else:
        # Below four samples the IQM is undefined. Report the mean and say so,
        # rather than silently switching estimators inside one table.
        point, lo, hi = float(values.mean()), float("nan"), float("nan")
        estimator, stratified_by = "mean", None
    return {"estimator": estimator, "value": point, "ci_low": lo, "ci_high": hi,
            "n": int(values.size), "ci_stratified_by": stratified_by}


def _strata(group: list[dict]) -> np.ndarray:
    """Evaluation seed AND, for a learned planner, the trained policy.

    A PPO cell holds three independently trained policies across five evaluation
    seeds; treating training-seed variation as evaluation-seed variation resamples
    across it and narrows the interval, which is the direction that manufactures
    significance.
    """
    return np.array([f"{r['seed']}|{r.get('ppo_train_seed')}" for r in group])


def _cell(agent, temperature, condition, group, resamples, ci) -> dict:
    strata = _strata(group)
    entry = {"agent": agent, "temperature": temperature,
             "reference_condition": condition, "n_runs": len(group)}
    for metric in METRICS:
        keep = [i for i, r in enumerate(group) if r.get(metric) is not None]
        if keep:
            values = np.array([group[i][metric] for i in keep], dtype=float)
            entry[metric] = _aggregate(values, strata[keep], resamples, ci)
    # Calibration numbers are carried through, never recomputed here, so the
    # planner table and the calibration file cannot disagree.
    for key in ("ece", "signed_calibration_error", "accuracy",
                "reference_temperature", "effective_divisor"):
        vals = [r[key] for r in group if r.get(key) is not None]
        entry[key] = float(np.mean(vals)) if vals else None
    return entry


def build_summary(csv_path: str | Path, config=None) -> dict[str, Any]:
    """Aggregate a sweep CSV into the structure every figure and document reads.

    Rows from the reference condition (the one the temperature grid is relative
    to) form the ``cells`` keyed ``<agent>@T<t>``, and feed the degradation spans
    and rankings. Rows from any other condition, such as the lab-fitted one,
    form ``conditions[<name>][<agent>]`` and are never pooled with the reference
    rows, even where the two share a temperature label.
    """
    rows = read_rows(csv_path)
    stats_cfg = (config.section("statistics") if config is not None else {})
    resamples = int(stats_cfg.get("bootstrap_resamples", 10_000))
    ci = float(stats_cfg.get("ci", 0.95))
    spans = dict(config.get("degradation_spans", DEFAULT_SPANS)) if config is not None \
        else DEFAULT_SPANS
    reference = (str(config.get("reference_condition")) if config is not None
                 and config.get("reference_condition") else None)
    if reference is None:
        conditions = {r.get("reference_condition") for r in rows}
        reference = conditions.pop() if len(conditions) == 1 else None

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    by_named: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        condition = row.get("reference_condition")
        if reference is None or condition in (None, reference):
            by_cell[(row["agent"], row["temperature"])].append(row)
        else:
            by_named[(condition, row["agent"])].append(row)

    cells = {f"{agent}@T{t:g}": _cell(agent, t, reference, group, resamples, ci)
             for (agent, t), group in sorted(by_cell.items())}
    named: dict[str, dict] = defaultdict(dict)
    for (condition, agent), group in sorted(by_named.items()):
        named[condition][agent] = _cell(agent, group[0]["temperature"], condition,
                                        group, resamples, ci)

    return {
        "cells": cells,
        "conditions": dict(named),
        "reference_condition": reference,
        "agents": sorted({r["agent"] for r in rows}),
        "temperatures": sorted({t for _, t in by_cell}),
        "degradation": _degradation(by_cell, spans, resamples, ci),
        "rankings": _rankings(by_cell),
        "n_rows": len(rows),
        "source_csv": str(Path(csv_path)),
        "estimator": "iqm + stratified bootstrap (Agarwal et al. 2021)",
    }


def _degradation(by_cell, spans: dict, resamples: int = 10_000, ci: float = 0.95) -> dict:
    """Detections-per-joule ratio over each span, per agent, with an interval.

    ``spans`` maps a name to ``[low, high]`` temperatures; the ratio is the value
    at ``low`` over the value at ``high``, so a planner that loses detections per
    joule as the classifier softens has a ratio above 1. The pilot reported its
    Lawnmower collapse over the full span, T = 0.3 to 4 (13.41x), which splits
    into 1.23x on the overconfident half and 10.86x on the underconfident half;
    reporting all three is what makes the comparison like-for-like.

    Each span carries the IQM at both ends, the ratio, a stratified bootstrap
    interval on the ratio, and the probability that a run at ``low`` beats a run
    at ``high``. A span whose temperatures are not in the sweep is skipped.
    """
    out: dict[str, dict] = {}
    agents = sorted({agent for agent, _ in by_cell})
    for name, (low, high) in spans.items():
        low, high = float(low), float(high)
        entry = {"low": low, "high": high, "agents": {}}
        for agent in agents:
            lo_rows = by_cell.get((agent, low))
            hi_rows = by_cell.get((agent, high))
            if not lo_rows or not hi_rows:
                continue
            lo = np.array([r["detections_per_joule"] for r in lo_rows], dtype=float)
            hi = np.array([r["detections_per_joule"] for r in hi_rows], dtype=float)
            lo_p = iqm(lo) if lo.size >= 4 else float(lo.mean())
            hi_p = iqm(hi) if hi.size >= 4 else float(hi.mean())
            result = {
                f"T{low:g}": lo_p,
                f"T{high:g}": hi_p,
                "ratio": (lo_p / hi_p) if hi_p > 0 else None,
                "probability_of_improvement": probability_of_improvement(lo, hi),
                "ci_low": None, "ci_high": None, "ci_stratified_by": None,
            }
            if lo.size >= 4 and hi.size >= 4:
                c_lo, c_hi, by = stratified_bootstrap_ratio_ci(
                    lo, _strata(lo_rows), hi, _strata(hi_rows),
                    resamples=resamples, ci=ci)
                result.update(ci_low=c_lo, ci_high=c_hi, ci_stratified_by=by)
            entry["agents"][agent] = result
        if entry["agents"]:
            out[name] = entry
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
