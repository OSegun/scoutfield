"""
Statistics and summary-aggregation tests.

The estimator choices here are the ones Agarwal et al. (2021) and Colas et al.
(2018, 2019) identify as the difference between RL results that replicate and RL
results that do not, so they are tested rather than assumed.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from scoutfield.analysis.stats import iqm, probability_of_improvement, stratified_bootstrap_ci


def test_iqm_discards_the_tails():
    """The point of the IQM: one lucky seed must not drag the estimate."""
    clean = np.array([1.0, 2.0, 3.0, 4.0])
    assert iqm(clean) == pytest.approx(2.5)

    # An extreme outlier moves the mean a long way and the IQM not at all.
    with_outlier = np.array([1.0, 2.0, 3.0, 1000.0])
    assert iqm(with_outlier) == pytest.approx(2.5)
    assert np.mean(with_outlier) > 200


def test_iqm_refuses_too_few_samples():
    with pytest.raises(ValueError, match="at least 4"):
        iqm(np.array([1.0, 2.0, 3.0]))


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    values = rng.normal(10.0, 1.0, 40)
    strata = np.repeat(np.arange(8), 5)
    lo, hi = stratified_bootstrap_ci(values, strata, resamples=2000, rng=rng)
    assert lo < iqm(values) < hi


def test_bootstrap_ci_is_reproducible_from_its_seed():
    """An unseeded interval is not a result — the pilot's third invariant."""
    values = np.random.default_rng(1).normal(5.0, 2.0, 40)
    strata = np.repeat(np.arange(8), 5)
    a = stratified_bootstrap_ci(values, strata, resamples=500,
                                rng=np.random.default_rng(7))
    b = stratified_bootstrap_ci(values, strata, resamples=500,
                                rng=np.random.default_rng(7))
    assert a == b


def test_bootstrap_ci_rejects_mismatched_strata():
    with pytest.raises(ValueError, match="differ in length"):
        stratified_bootstrap_ci(np.zeros(10), np.zeros(4))


def test_probability_of_improvement_handles_the_obvious_cases():
    assert probability_of_improvement(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert probability_of_improvement(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    # Ties count as half a win: detections-per-joule has a discrete numerator, so
    # exact ties are common and counting them as losses would bias every column.
    assert probability_of_improvement(np.array([1.0]), np.array([1.0])) == 0.5


def _write_sweep_csv(path, agents=("lawnmower", "greedy_entropy"),
                     temperatures=(1.0, 4.0), seeds=(0, 1, 2, 3, 4)):
    """A synthetic sweep in which lawnmower collapses at T=4 and greedy does not."""
    rows = []
    for agent in agents:
        for temperature in temperatures:
            for seed in seeds:
                base = 0.08 if agent == "lawnmower" else 0.04
                value = base if temperature == 1.0 else (base / 10 if
                                                         agent == "lawnmower" else base * 0.9)
                rows.append({
                    "run_id": f"{agent}__s{seed}__T{temperature:g}__tau0.75__sig1.5",
                    "agent": agent, "seed": seed, "temperature": temperature,
                    "tau": 0.75, "sigma": 1.5, "episodes": 3,
                    "recall": 0.3, "precision": 0.5,
                    "detections_per_joule": value + seed * 1e-4,
                    "false_alarms": 2, "coverage": 0.3,
                    "time_to_first_detection": 10, "ece": 0.05,
                    "signed_calibration_error": -0.05, "accuracy": 0.82,
                })
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_summary_reports_degradation_and_ranking(tmp_path):
    """The summary must carry the two things the pilot's headline was made of:
    a per-agent degradation ratio and the ranking at each temperature."""
    from scoutfield.analysis.summary import build_summary

    csv_path = _write_sweep_csv(tmp_path / "sweep.csv")
    summary = build_summary(csv_path)

    assert summary["n_rows"] == 20
    assert summary["agents"] == ["greedy_entropy", "lawnmower"]

    degradation = summary["degradation"]
    # Lawnmower was built to collapse ~10x and greedy to hold roughly flat. The
    # ratio is 9.78 rather than exactly 10 because the fixture adds a per-seed
    # offset, which is deliberate: without spread the IQM would be untested.
    assert degradation["lawnmower"]["ratio"] == pytest.approx(9.78, rel=0.01)
    assert degradation["greedy_entropy"]["ratio"] == pytest.approx(1.11, rel=0.05)
    # Every seed collapses, not just the average.
    assert degradation["lawnmower"]["probability_of_improvement_T1_over_T4"] == 1.0

    # The ranking reverses at T=4, which is the pilot's finding restated.
    assert summary["rankings"]["T1"][0]["agent"] == "lawnmower"
    assert summary["rankings"]["T4"][0]["agent"] == "greedy_entropy"


def test_summary_cells_carry_intervals(tmp_path):
    from scoutfield.analysis.summary import build_summary

    summary = build_summary(_write_sweep_csv(tmp_path / "sweep.csv"))
    cell = summary["cells"]["lawnmower@T1"]
    assert cell["detections_per_joule"]["estimator"] == "iqm"
    assert cell["detections_per_joule"]["n"] == 5
    assert cell["ece"] == pytest.approx(0.05)


def test_summary_fails_loudly_when_there_is_nothing_to_read(tmp_path):
    from scoutfield.analysis.summary import build_summary

    with pytest.raises(FileNotFoundError, match="04_sweep"):
        build_summary(tmp_path / "absent.csv")


def test_figures_refuse_a_stale_summary(tmp_path):
    """A stale summary quietly producing plausible figures is the failure mode
    most likely to reach a document unnoticed."""
    import importlib.util
    import os
    import time
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "figs", Path("experiments/make_figures.py"))
    figs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs)

    summary = tmp_path / "summary.json"
    newer_csv = tmp_path / "sweep_results.csv"
    summary.write_text("{}", encoding="utf-8")
    time.sleep(0.01)
    newer_csv.write_text("x", encoding="utf-8")
    os.utime(newer_csv, (time.time() + 10, time.time() + 10))

    with pytest.raises(SystemExit, match="older than"):
        figs.check_not_stale(summary, [newer_csv])

    with pytest.raises(SystemExit, match="does not exist"):
        figs.check_not_stale(tmp_path / "absent.json", [])
