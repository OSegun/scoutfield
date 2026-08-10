"""
Statistics: interquartile mean with stratified bootstrap confidence intervals.

Why not mean +/- standard deviation over three seeds
-----------------------------------------------------
Because it is, per Agarwal et al. (2021) and Colas et al. (2018, 2019), the single
commonest reason reinforcement learning results fail to replicate. RL returns are
heavy-tailed and multimodal: one lucky seed drags the mean, and a standard
deviation computed over three samples is close to meaningless. The interquartile
mean discards the top and bottom quartiles and is far more stable at the sample
sizes a no-budget project can afford.

The pilot used exactly this and the numbers held. Same method here, so the two
phases are directly comparable — a different estimator would make the comparison
between phases partly an artefact of the statistics.

Prefer importing the pilot's implementation where one exists, for the same reason
metrics.py imports its ECE: two implementations that differ slightly produce
numbers that cannot be compared.
"""

from __future__ import annotations

import numpy as np


def iqm(values: np.ndarray) -> float:
    """Interquartile mean: the mean of the middle 50% of the sample."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n < 4:
        raise ValueError(f"IQM needs at least 4 samples, got {n}")
    lo, hi = n // 4, n - n // 4
    return float(v[lo:hi].mean())


def stratified_bootstrap_ci(
    values: np.ndarray,
    strata: np.ndarray,
    statistic=iqm,
    resamples: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Confidence interval by resampling within strata.

    Stratifying by seed matters: runs from the same seed share a field layout and
    a classifier draw, so they are not independent. Resampling them as if they
    were produces intervals that are too narrow, which is the direction that
    manufactures significance.
    """
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    if values.shape[0] != strata.shape[0]:
        raise ValueError(f"values and strata differ in length: "
                         f"{values.shape[0]} != {strata.shape[0]}")
    rng = rng if rng is not None else np.random.default_rng(0)

    # Index positions per stratum, so each resample preserves stratum sizes.
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]

    stats = np.empty(resamples, dtype=float)
    for i in range(resamples):
        picked = np.concatenate([
            g[rng.integers(0, g.size, size=g.size)] for g in groups
        ])
        stats[i] = statistic(values[picked])

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def probability_of_improvement(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) over paired runs, per Agarwal et al. (2021).

    More informative than a significance test for this project's question:
    "GreedyEntropy beats Lawnmower 7.3x at T = 4" is a ratio of central
    tendencies and says nothing about how often it wins on an individual field.
    A practitioner deciding which planner to fly cares about the latter.

    Ties count as half a win, the standard convention for the Mann-Whitney
    statistic this is. Counting them as losses would bias every comparison on a
    discrete metric — and detections-per-joule is discrete in its numerator.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("probability_of_improvement needs non-empty samples")
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)
