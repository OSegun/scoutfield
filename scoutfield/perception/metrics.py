"""
Calibration metrics.

The pilot already implements ECE in its ``perception.py`` and that implementation
is the reference. Import it rather than reimplementing:

    from perception import expected_calibration_error   # the PILOT's module

Two implementations of ECE with different binning conventions produce different
numbers, and comparing this phase's ECE to the pilot's 0.0037 / 0.1980 only means
something if both were computed the same way.

What this module adds beyond the pilot
--------------------------------------
* MCE, NLL and Brier score. ECE alone hides the shape of the miscalibration, and
  the pilot's refuted hypothesis H2 — that performance is monotone in ECE — is
  precisely a finding about ECE being an insufficient summary. The *direction* of
  miscalibration governed the outcome, not its magnitude.
* Signed calibration error, so over- and underconfidence are distinguishable in
  a single number. This is the quantity H2's refutation says actually matters.
* Reliability diagrams, per condition, for roadmap item 7.
* Per-class breakdowns. An aggregate ECE over an imbalanced binary problem can
  look excellent while one class is badly miscalibrated.
"""

from __future__ import annotations

import itertools

import numpy as np

# Bin edges follow the pilot's convention exactly: equal width over [0, 1] and a
# half-open (lo, hi] membership test. Comparing this phase's ECE against the
# pilot's 0.0037 / 0.1980 is only meaningful if the binning matches.
_EPS = 1e-12


def confidence_and_correctness(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map (P(disease), true label) to the pilot's (confidence, correct) pair.

    The pilot's metrics take confidence in the *predicted* class, not the
    probability of the positive class. Converting here, once, keeps every metric
    in this module on the pilot's convention.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    conf = np.maximum(probs, 1.0 - probs)
    correct = ((probs > 0.5).astype(int) == labels).astype(float)
    return conf, correct


def _bin_stats(conf: np.ndarray, correct: np.ndarray, bins: int):
    """Yield (count, mean confidence, mean accuracy) for each non-empty bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        yield int(m.sum()), float(conf[m].mean()), float(correct[m].mean()), float(lo), float(hi)


def signed_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Confidence minus accuracy, bin-weighted, *without* the absolute value.

    Positive means overconfident, negative means underconfident. ECE takes the
    absolute value and so throws away the sign — which, given that the pilot
    found direction rather than magnitude to be what governs planner performance,
    discards the informative part.
    """
    conf, correct = confidence_and_correctness(probs, labels)
    if conf.size == 0:
        return float("nan")
    n = conf.size
    return float(
        sum(
            (count / n) * (mean_conf - mean_acc)
            for count, mean_conf, mean_acc, _, _ in _bin_stats(conf, correct, bins)
        )
    )


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Largest absolute confidence-accuracy gap over the bins.

    ECE averages the gap away; a single badly wrong bin can matter more than the
    average when the planner's confirmation threshold sits inside that bin.
    """
    conf, correct = confidence_and_correctness(probs, labels)
    if conf.size == 0:
        return float("nan")
    gaps = [abs(c - a) for _, c, a, _, _ in _bin_stats(conf, correct, bins)]
    return float(max(gaps)) if gaps else float("nan")


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean binary NLL of the reported probabilities."""
    probs = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0 - _EPS)
    labels = np.asarray(labels, dtype=float)
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error of the reported probability against the label."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def reliability_data(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> dict:
    """Bin centres, per-bin accuracy, per-bin confidence and per-bin counts.

    Returns data rather than a plot so the same numbers can be written to the
    results file and re-plotted without re-running inference.
    """
    conf, correct = confidence_and_correctness(probs, labels)
    centres, accs, confs, counts = [], [], [], []
    for count, mean_conf, mean_acc, lo, hi in _bin_stats(conf, correct, bins):
        centres.append(0.5 * (lo + hi))
        accs.append(mean_acc)
        confs.append(mean_conf)
        counts.append(count)
    return {
        "bin_centre": np.asarray(centres, dtype=float),
        "accuracy": np.asarray(accs, dtype=float),
        "confidence": np.asarray(confs, dtype=float),
        "count": np.asarray(counts, dtype=int),
        "bins": bins,
        "n": int(conf.size),
    }


def per_class_breakdown(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> dict:
    """Calibration metrics computed separately for each true class.

    An aggregate ECE over an imbalanced binary problem can look excellent while
    one class is badly miscalibrated, so the aggregate is never reported alone.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    out = {}
    for cls in (0, 1):
        m = labels == cls
        if not m.any():
            continue
        out[cls] = {
            "n": int(m.sum()),
            "signed_calibration_error": signed_calibration_error(probs[m], labels[m], bins),
            "maximum_calibration_error": maximum_calibration_error(probs[m], labels[m], bins),
            "nll": negative_log_likelihood(probs[m], labels[m]),
            "brier": brier_score(probs[m], labels[m]),
        }
    return out


# Project visual identity, shared with the pilot's figures.
_FOREST = "#2C5F2D"
_MOSS = "#97BC62"
_TINT = "#EEF3EA"
_INK = "#1F2A1F"
_MUTED = "#6B7A6B"


def plot_reliability(data: dict, ax=None, title: str = ""):
    """Reliability diagram from ``reliability_data`` output.

    Bin counts are annotated: a bin holding four samples looks identical to one
    holding four thousand otherwise, and readers over-interpret the sparse tail.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    centres = data["bin_centre"]
    accs = data["accuracy"]
    confs = data["confidence"]
    counts = data["count"]

    ax.plot([0, 1], [0, 1], linestyle="--", color=_MUTED, linewidth=1.0, zorder=1)
    width = 1.0 / max(int(data.get("bins", 15)), 1)
    ax.bar(centres, accs, width=width * 0.9, color=_MOSS, edgecolor=_FOREST,
           linewidth=0.8, zorder=2, label="accuracy")
    ax.plot(centres, confs, marker="o", markersize=4, color=_FOREST,
            linewidth=1.4, zorder=3, label="confidence")

    total = max(int(data.get("n", counts.sum() if len(counts) else 0)), 1)
    for x, a, c in zip(centres, accs, counts):
        ax.annotate(f"{c}\n{100.0 * c / total:.0f}%", (x, a), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6, color=_INK)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence", color=_INK)
    ax.set_ylabel("accuracy", color=_INK)
    ax.set_facecolor(_TINT)
    if title:
        ax.set_title(title, color=_INK)
    ax.legend(frameon=False, fontsize=8)
    return ax
