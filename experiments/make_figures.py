"""
Regenerate every figure from the run's results in the artefact store.

    python experiments/make_figures.py

Inputs are pulled from stages 02 to 05 of the run named in ``configs/artifacts.yaml``;
Figures 9 to 15 are pushed to stage ``06_figures``. Figure 16 needs the optional
uncertainty-method comparison and is drawn only when that exists.

Rules inherited from the pilot, and why each exists
----------------------------------------------------
* Every number is read from a generated results file. Nothing is hard-coded. Never
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
from pathlib import Path

import numpy as np

from scoutfield.utils.paths import figures_dir

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
    return sorted(out, key=lambda pair: pair[0])


COLOURS = [PALETTE["forest"], PALETTE["moss"], PALETTE["leaf"],
           PALETTE["panel"], PALETTE["muted"]]


def _temperature_axis(ax, temperatures) -> None:
    """Log-scaled temperature axis, labelled with the grid's own values.

    Log scale because the grid is multiplicative around T = 1 (0.3 and 3 are
    equally far from calibrated). Plain labels because the default log labels
    need superscript glyphs the body font may not have.
    """
    from matplotlib.ticker import FixedLocator, NullLocator

    ticks = sorted({float(t) for t in temperatures})
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def figure_9_reliability_under_shift(calibration, plt) -> Path:
    """Reliability on shift_eval, lab-fitted and shift-fitted side by side.

    The lab-fitted panel is a system calibrated on PlantVillage and flown over a
    field; the shift-fitted panel is the calibration point the planner sweep is
    centred on. Bars further from the diagonal are worse calibrated.
    """
    panels = [("shift_eval_lab", "Lab-fitted temperature"),
              ("shift_eval", "Shift-fitted temperature")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (name, title) in zip(axes, panels):
        sweep = calibration["sweeps"][name]
        rel = sweep["reliability_at_fitted"]
        width = 1.0 / rel["bins"]
        ax.bar(rel["bin_centre"], rel["accuracy"], width=width * 0.9,
               color=PALETTE["moss"], edgecolor=PALETTE["forest"], label="accuracy")
        ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color=PALETTE["muted"],
                linewidth=1, label="perfect calibration")
        ece = sweep["at_fitted_temperature"]["ece"]
        ax.set_title(f"{title}\n(T* = {sweep['fitted_temperature']:.3f}, ECE {ece:.4f})")
        ax.set_xlabel("confidence")
        ax.set_xlim(0.5, 1.0)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Calibration under shift, PlantDoc shift_eval", fontsize=11)
    fig.tight_layout()
    return _save(fig, 9)


def figure_10_temperature_sweep(calibration, plt) -> Path:
    """Accuracy invariance and the ECE curve, from the calibration run."""
    fig, (ax_acc, ax_ece) = plt.subplots(1, 2, figsize=(9, 3.6))
    series = [("test", "PlantVillage test (lab-fitted)"),
              ("shift_eval_lab", "PlantDoc shift_eval (lab-fitted)"),
              ("shift_eval", "PlantDoc shift_eval (shift-fitted)")]
    for colour, (name, label) in zip(COLOURS, series):
        sweep = calibration["sweeps"][name]["sweep"]
        temps = sorted(float(t) for t in sweep)
        ax_acc.plot(temps, [sweep[f"{t:g}"]["accuracy"] for t in temps],
                    marker="o", markersize=4, color=colour, label=label)
        ax_ece.plot(temps, [sweep[f"{t:g}"]["ece"] for t in temps],
                    marker="o", markersize=4, color=colour, label=label)

    ax_acc.set_title("Accuracy is invariant to temperature")
    ax_acc.set_ylabel("accuracy")
    ax_ece.set_title("ECE against temperature")
    ax_ece.set_ylabel("ECE")
    grid = [float(t) for t in calibration["sweeps"]["shift_eval"]["sweep"]]
    for ax in (ax_acc, ax_ece):
        _temperature_axis(ax, grid)
        ax.set_xlabel("T, relative to the fitted temperature")
        ax.legend(frameon=False, fontsize=7)
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


def figure_12_ppo_curves(curves: list[dict], plt) -> Path:
    """Learning curves of the final policies, one per training seed."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for colour, data in zip(COLOURS, curves):
        episodes = data.get("episodes", [])
        if not episodes:
            continue
        steps = [e["timestep"] for e in episodes]
        vals = [e.get("detections_per_joule", np.nan) for e in episodes]
        # Rolling mean: per-episode detections-per-joule is dominated by rare
        # events and the raw trace hides whether the curve has flattened.
        window = max(len(vals) // 20, 1)
        smoothed = np.convolve(vals, np.ones(window) / window, mode="valid")
        ax.plot(steps[:len(smoothed)], smoothed, linewidth=1.2, color=colour,
                label=f"seed {data['seed']}")

    ax.set_xlabel("timestep")
    ax.set_ylabel("detections per joule (rolling mean)")
    ax.set_title("PPO learning curves")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, 12)


def figure_13_detections_per_joule(summary, plt) -> Path:
    """The pilot's headline metric, re-measured on a real classifier.

    T = 1 is the calibrated point under shift. Left of it the classifier is
    overconfident, right of it underconfident; the two halves are shaded because
    the pilot's effect lived almost entirely in the right-hand one.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4))
    temps_all = summary["temperatures"]
    lo_t, hi_t = min(temps_all), max(temps_all)
    ax.axvspan(lo_t, 1.0, color=PALETTE["warm"], alpha=0.9, linewidth=0)
    ax.axvspan(1.0, hi_t, color=PALETTE["tint"], alpha=0.9, linewidth=0)
    ax.axvline(1.0, color=PALETTE["muted"], linestyle="--", linewidth=1)

    for colour, agent in zip(COLOURS, summary["agents"]):
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

    _temperature_axis(ax, temps_all)
    ymax = ax.get_ylim()[1]
    ax.text(np.sqrt(lo_t * 1.0), ymax * 0.97, "overconfident", ha="center", va="top",
            fontsize=8, color=PALETTE["muted"])
    ax.text(np.sqrt(1.0 * hi_t), ymax * 0.97, "underconfident", ha="center", va="top",
            fontsize=8, color=PALETTE["muted"])
    ax.set_xlabel("T, relative to the shift-fitted temperature (T = 1 calibrated)")
    ax.set_ylabel("detections per joule")
    ax.set_title("Detections per joule by temperature")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, 13)


def figure_14_tau_surface(tau_summary, plt) -> Path:
    """Effect size as a function of tau — the analysis that unblocks the headline."""
    by_agent: dict[str, list[tuple[float, float]]] = {}
    for entry in tau_summary.values():
        by_agent.setdefault(entry["agent"], []).append((entry["tau"], entry["ratio"]))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    # A planner with no detections at T = 4 has an infinite ratio. Matplotlib drops
    # non-finite points without a word, which would make the line simply stop; those
    # points are drawn as triangles on the top edge and labelled instead.
    unbounded: list[tuple[str, list[float]]] = []
    for colour, (agent, points) in zip(COLOURS, sorted(by_agent.items())):
        points.sort()
        finite = [(t, r) for t, r in points if np.isfinite(r)]
        ax.plot([t for t, _ in finite], [r for _, r in finite],
                marker="o", markersize=4, color=colour, label=agent)
        infinite = [t for t, r in points if not np.isfinite(r)]
        if infinite:
            unbounded.append((colour, infinite))

    if unbounded:
        top = ax.get_ylim()[1]
        ax.set_ylim(top=top)
        for colour, taus in unbounded:
            ax.scatter(taus, [top] * len(taus), marker="^", s=40, color=colour,
                       clip_on=False, zorder=3)
            ax.annotate("no detections at T = 4", (taus[-1], top), fontsize=8,
                        color=colour, xytext=(-6, -14), textcoords="offset points",
                        ha="right")

    ax.axvline(0.75, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.annotate("pilot tau = 0.75", (0.75, ax.get_ylim()[1]), fontsize=8,
                color=PALETTE["muted"], xytext=(4, -12), textcoords="offset points")
    ax.set_xlabel("confirmation threshold tau")
    ax.set_ylabel("detections/joule ratio, T=1 over T=4")
    ax.set_title("Effect size against tau")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, 14)


def figure_15_regret(summary, plt) -> Path:
    """Regret against the oracle on detections per joule, per planner and T.

    The oracle is greedy nearest-diseased-cell, a lower bound on the true
    ceiling, so these regrets understate the real gap.
    """
    from scoutfield.planners.oracle import regret

    oracle = dict(_cells(summary, "oracle"))
    if not oracle:
        raise SystemExit("summary.json has no oracle cells; Figure 15 needs them")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    planners = [a for a in summary["agents"] if a != "oracle"]
    for colour, agent in zip(COLOURS, planners):
        pairs = [(t, c) for t, c in _cells(summary, agent) if t in oracle]
        temps = [t for t, _ in pairs]
        vals = [regret(c["detections_per_joule"]["value"],
                       oracle[t]["detections_per_joule"]["value"]) for t, c in pairs]
        ax.plot(temps, vals, marker="o", markersize=4, color=colour, label=agent)
    ax.axvline(1.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    _temperature_axis(ax, summary["temperatures"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("T, relative to the shift-fitted temperature")
    ax.set_ylabel("regret on detections per joule")
    ax.set_title("Regret against the greedy oracle (a lower bound)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, 15)


def check_same_commit(store, stage: str, names: list[str]) -> None:
    """Refuse to draw from files of one stage that were not pushed together.

    The summary and the CSV it was built from are pushed in one commit when a
    sweep completes. If their manifest entries record different commits, one was
    replaced without the other, and a figure from the pair would mix two states.
    """
    files = store.stage_entry(stage).get("files", {})
    commits = {files[n]["hub_commit"] for n in names if n in files}
    if len(commits) > 1:
        raise SystemExit(f"{', '.join(names)} in stage {stage} come from different "
                         "store commits; re-run that stage to completion first.")


def make_all(only: list[int] | None = None, store=None) -> list[Path]:
    """Pull the run's results, draw every figure, push them to ``06_figures``."""
    from scoutfield.config import load_config
    from scoutfield.utils.artifacts import get_store

    store = store or get_store()
    plt = _style()
    wanted = set(only) if only else set(FIGURES)
    written: list[Path] = []

    calibration = store.pull_json("02_calibration", "calibration_sweep.json")
    if 9 in wanted:
        written.append(figure_9_reliability_under_shift(calibration, plt))
    if 10 in wanted:
        written.append(figure_10_temperature_sweep(calibration, plt))
    if 11 in wanted:
        written.append(figure_11_coverage_ceiling(plt))
    if 12 in wanted:
        seeds = load_config("configs/sweep.yaml")["ppo_train_seeds"]
        curves = [store.pull_json("03_ppo", f"ppo_curve_seed{s}.json") for s in seeds]
        written.append(figure_12_ppo_curves(curves, plt))
    if wanted & {13, 15}:
        check_same_commit(store, "04_sweep", ["summary.json", "sweep_results.csv"])
        summary = store.pull_json("04_sweep", "summary.json")
        if 13 in wanted:
            written.append(figure_13_detections_per_joule(summary, plt))
        if 15 in wanted:
            written.append(figure_15_regret(summary, plt))
    if 14 in wanted:
        written.append(figure_14_tau_surface(
            store.pull_json("05_tau", "tau_sensitivity_summary.json"), plt))
    if 16 in wanted:
        print("  Figure 16 needs the uncertainty-method comparison (stage 07); not drawn")

    if written:
        store.push("06_figures", written, f"figures {sorted(wanted - {16})}")
        store.record_inputs("06_figures")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", type=int, default=None,
                        help=f"figure numbers to rebuild; default all of {sorted(FIGURES)}")
    args = parser.parse_args()
    written = make_all(args.only)
    print(f"{len(written)} figure(s) written to {figures_dir()} and pushed")


if __name__ == "__main__":
    main()
