"""
Regenerate every figure from results/summary.json.

    python experiments/make_figures.py

Rules inherited from the pilot, and why each exists
----------------------------------------------------
* Every number is read from ``results/summary.json``. Nothing is hard-coded. Never
  regenerate a document before regenerating the figures, or text and data disagree and
  nobody notices.
* Figures are named ``Figure{N}_{Title_Case}.png`` and referenced *by number* in the
  surrounding narrative, not merely captioned.
* Generated files are never hand-edited. A wrong figure means wrong code or wrong data.

Visual identity, shared across every deliverable in this project::

    Forest #2C5F2D   Dark #1B3A1C   Panel #24491F   Moss #97BC62  Leaf #5A8F4A
    Tint   #EEF3EA   Tint2 #E3EEDB  Warm  #F2EEE6   Ink  #1F2A1F  Muted #6B7A6B
    Headings: Cambria    Body: Calibri

No accent stripes, no colour bars. Cards use a background tint and a soft shadow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scoutfield.utils.paths import figures_dir, results_dir

PALETTE = {
    "forest": "#2C5F2D", "dark": "#1B3A1C", "panel": "#24491F",
    "moss": "#97BC62", "leaf": "#5A8F4A", "tint": "#EEF3EA",
    "tint2": "#E3EEDB", "warm": "#F2EEE6", "ink": "#1F2A1F", "muted": "#6B7A6B",
}

# Numbering continues from the pilot write-up, which used Figures 1-8.
FIGURES = {
    9:  "Perception_Calibration_Under_Shift",     # PlantVillage vs PlantDoc reliability
    10: "Temperature_Sweep_Real_Classifier",      # accuracy invariance + ECE curve
    11: "Coverage_Ceiling_Field32",               # verifies the ~40% regime
    12: "PPO_Learning_Curves",                    # >=3 seeds, IQM with CI band
    13: "Detections_Per_Joule_By_Temperature",    # the pilot's headline, re-measured
    14: "Tau_Sensitivity_Surface",                # unblocks the effect size
    15: "Regret_Against_Oracle",
    16: "Uncertainty_Method_Comparison",          # temp scaling vs MC-dropout vs ensemble
}


def _style():
    """Apply the project's visual identity to matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "DejaVu Sans"],
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.edgecolor": PALETTE["muted"],
        "axes.labelcolor": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "figure.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
    return plt


def check_not_stale(summary_path: Path, sources: list[Path]) -> None:
    """Fail loudly if the summary predates any results file it derives from.

    A stale summary quietly producing plausible-looking figures is the failure
    mode most likely to reach a document unnoticed — and it has reached one
    before.
    """
    if not summary_path.exists():
        raise SystemExit(
            f"{summary_path} does not exist. Run experiments/04_sweep.py to completion "
            "first; figures are never drawn from anything but the summary."
        )
    summary_mtime = summary_path.stat().st_mtime
    stale = [s for s in sources if s.exists() and s.stat().st_mtime > summary_mtime]
    if stale:
        raise SystemExit(
            f"{summary_path.name} is older than {[s.name for s in stale]}. "
            "Re-run the sweep so the summary is rebuilt, then regenerate figures — "
            "never the other way round."
        )


def _save(fig, number: int) -> Path:
    path = figures_dir() / f"Figure{number}_{FIGURES[number]}.png"
    fig.savefig(path)
    print(f"  wrote {path.name}")
    return path


def _cells(summary, agent):
    """(temperature, cell) pairs for one agent, ordered by temperature."""
    out = [(c["temperature"], c) for c in summary["cells"].values() if c["agent"] == agent]
    return sorted(out)


def figure_13_detections_per_joule(summary, plt) -> Path:
    """The pilot's headline metric, re-measured on a real classifier."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colours = [PALETTE["forest"], PALETTE["moss"], PALETTE["leaf"],
               PALETTE["panel"], PALETTE["muted"]]

    for colour, agent in zip(colours, summary["agents"]):
        pairs = _cells(summary, agent)
        if not pairs:
            continue
        temps = [t for t, _ in pairs]
        vals = [c["detections_per_joule"]["value"] for _, c in pairs]
        lo = [c["detections_per_joule"]["ci_low"] for _, c in pairs]
        hi = [c["detections_per_joule"]["ci_high"] for _, c in pairs]
        ax.plot(temps, vals, marker="o", markersize=4, color=colour, label=agent)
        if not any(np.isnan(lo)):
            ax.fill_between(temps, lo, hi, color=colour, alpha=0.15, linewidth=0)

    ax.set_xlabel("temperature (relative to the fitted calibrated point)")
    ax.set_ylabel("detections per joule")
    ax.set_title("Detections per joule by temperature")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, 13)


def figure_10_temperature_sweep(plt) -> Path | None:
    """Accuracy invariance and the ECE curve, from the calibration run."""
    path = results_dir() / "calibration_sweep.json"
    if not path.exists():
        print("  skipping Figure 10: results/calibration_sweep.json not found")
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    fig, (ax_acc, ax_ece) = plt.subplots(1, 2, figsize=(9, 3.6))
    for split, colour in (("test", PALETTE["forest"]), ("shift", PALETTE["moss"])):
        sweep = data.get("splits", {}).get(split, {}).get("sweep")
        if not sweep:
            continue
        temps = sorted(float(t) for t in sweep)
        ax_acc.plot(temps, [sweep[f"{t:g}"]["accuracy"] for t in temps],
                    marker="o", markersize=4, color=colour, label=split)
        ax_ece.plot(temps, [sweep[f"{t:g}"]["ece"] for t in temps],
                    marker="o", markersize=4, color=colour, label=split)

    ax_acc.set_title("Accuracy is invariant to temperature")
    ax_acc.set_xlabel("temperature")
    ax_acc.set_ylabel("accuracy")
    ax_ece.set_title("ECE is minimised at the calibrated point")
    ax_ece.set_xlabel("temperature")
    ax_ece.set_ylabel("ECE")
    for ax in (ax_acc, ax_ece):
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, 10)


def figure_11_coverage_ceiling(plt) -> Path:
    """Verifies the ~40% regime the 32x32 budget is supposed to produce."""
    from scoutfield.config import load_config
    from scoutfield.envs.field_env import coverage_ceiling

    env_cfg = load_config("configs/ppo_field32.yaml").section("env")
    budgets = np.linspace(100, 1600, 60)
    ceilings = [coverage_ceiling(env_cfg["grid"], b, env_cfg["energy"]["hover"],
                                 env_cfg["energy"]["translate"]) for b in budgets]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(budgets, ceilings, color=PALETTE["forest"])
    chosen = float(env_cfg["budget"])
    ax.axvline(chosen, color=PALETTE["leaf"], linestyle="--", linewidth=1)
    ax.annotate(f"budget = {chosen:g}\nceiling = "
                f"{coverage_ceiling(env_cfg['grid'], chosen):.3f}",
                (chosen, 0.55), fontsize=8, color=PALETTE["ink"],
                xytext=(8, 0), textcoords="offset points")
    ax.set_xlabel("energy budget")
    ax.set_ylabel("optimistic coverage ceiling")
    ax.set_title(f"Coverage ceiling, {env_cfg['grid']}x{env_cfg['grid']} field")
    return _save(fig, 11)


def figure_12_ppo_curves(plt) -> Path | None:
    """Learning curves over every seed that has been trained."""
    curves = sorted(results_dir().glob("ppo_curve_seed*.json"))
    if not curves:
        print("  skipping Figure 12: no results/ppo_curve_seed*.json found")
        return None

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for path in curves:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        episodes = data.get("episodes", [])
        if not episodes:
            continue
        steps = [e["timestep"] for e in episodes]
        vals = [e.get("detections_per_joule", np.nan) for e in episodes]
        # Rolling mean: per-episode detections-per-joule is dominated by rare
        # events and the raw trace hides whether the curve has flattened.
        window = max(len(vals) // 20, 1)
        smoothed = np.convolve(vals, np.ones(window) / window, mode="valid")
        ax.plot(steps[:len(smoothed)], smoothed, linewidth=1.2,
                label=f"seed {data['seed']}")

    ax.set_xlabel("timestep")
    ax.set_ylabel("detections per joule (rolling mean)")
    ax.set_title("PPO learning curves")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, 12)


def figure_14_tau_surface(plt) -> Path | None:
    """Effect size as a function of tau — the analysis that unblocks the headline."""
    path = results_dir() / "tau_sensitivity_summary.json"
    if not path.exists():
        print("  skipping Figure 14: results/tau_sensitivity_summary.json not found")
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    by_agent: dict[str, list[tuple[float, float]]] = {}
    for entry in data.values():
        by_agent.setdefault(entry["agent"], []).append((entry["tau"], entry["ratio"]))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    colours = [PALETTE["forest"], PALETTE["moss"], PALETTE["leaf"], PALETTE["panel"]]
    for colour, (agent, points) in zip(colours, sorted(by_agent.items())):
        points.sort()
        ax.plot([t for t, _ in points], [r for _, r in points],
                marker="o", markersize=4, color=colour, label=agent)

    ax.axvline(0.75, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.annotate("pilot tau = 0.75", (0.75, ax.get_ylim()[1]), fontsize=8,
                color=PALETTE["muted"], xytext=(4, -12), textcoords="offset points")
    ax.set_xlabel("confirmation threshold tau")
    ax.set_ylabel("detections/joule ratio, T=1 over T=4")
    ax.set_title("Effect size against tau")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, 14)


def make_all(only: list[int] | None = None) -> list[Path]:
    """Regenerate figures from the generated results, never from hand-entered numbers."""
    plt = _style()
    summary_path = results_dir() / "summary.json"
    wanted = set(only) if only else set(FIGURES)
    written: list[Path] = []

    # Figures that do not depend on the sweep summary can be built without it.
    if 11 in wanted:
        written.append(figure_11_coverage_ceiling(plt))
    if 10 in wanted:
        written.append(figure_10_temperature_sweep(plt))
    if 12 in wanted:
        written.append(figure_12_ppo_curves(plt))
    if 14 in wanted:
        written.append(figure_14_tau_surface(plt))

    if wanted & {13}:
        check_not_stale(summary_path, [
            results_dir() / "sweep_results.csv",
            results_dir() / "calibration_sweep.json",
        ])
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        written.append(figure_13_detections_per_joule(summary, plt))

    unimplemented = wanted & {9, 15, 16}
    if unimplemented:
        print(f"  figures {sorted(unimplemented)} need data that does not exist yet "
              "(uncertainty-method comparison and regret); not drawn")

    return [w for w in written if w is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", type=int, default=None,
                        help=f"figure numbers to rebuild; default all of {sorted(FIGURES)}")
    args = parser.parse_args()
    written = make_all(args.only)
    print(f"{len(written)} figure(s) written to {figures_dir()}")


if __name__ == "__main__":
    main()
