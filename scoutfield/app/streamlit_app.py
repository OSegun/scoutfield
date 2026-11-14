"""
ScoutPlan demo interface.

Run:
    make app        # streamlit run scoutfield/app/streamlit_app.py

What it should demonstrate
---------------------------
Not "here is a leaf classifier" — that is a solved problem and demonstrating it
proves nothing about this work. The demonstration is the *coupling*: that a
miscalibrated classifier changes where a planner chooses to fly and how much
disease it finds per joule.

Three panels, in this order:

1. **Classify.** Upload or pick a leaf image; show probability, prediction, and
   the raw logit. A temperature slider recalibrates live. The point a viewer
   should take away: the prediction does not move as the slider does, only the
   confidence. That is the study's central mechanism made visible in one control.

2. **Plan.** Run an episode on a simulated field at the chosen temperature; show
   the flown path, the belief map evolving, and energy drawn down. Run two
   planners side by side at the same temperature so the difference is visible
   rather than asserted.

3. **Compare.** The measured detections-per-joule against temperature, read from
   ``results/summary.json`` — never hard-coded, never edited by hand.

Honesty in the interface
-------------------------
If the learned planner still loses to the lawnmower baseline, the demo shows that.
A demo that quietly presents only the favourable configuration is a
misrepresentation, and the pilot study set the opposite precedent by reporting two
refuted hypotheses and an underperforming agent in plain terms.

Visual identity as in ``experiments/figures.py``: Forest #2C5F2D, Moss #97BC62,
Tint #EEF3EA, Ink #1F2A1F; Cambria headings, Calibri body; no accent stripes.
"""

from __future__ import annotations

import json

import numpy as np

FOREST, MOSS, TINT, INK, MUTED = "#2C5F2D", "#97BC62", "#EEF3EA", "#1F2A1F", "#6B7A6B"


def _classify_panel(st, store, settings) -> None:
    """Panel 1: the mechanism, made visible in one slider.

    The takeaway a viewer should leave with is that the *prediction* does not move
    as the temperature slider does, only the confidence. The slider is the
    experiment's T, relative to the fitted temperature of the sweep's reference
    condition, so slider = 1 shows the calibrated probability.
    """
    st.subheader("1. Classify")
    from scoutfield.app.export_onnx import load_deployment_config, locate_classifier
    from scoutfield.app.inference import ScoutPlanInference

    try:
        onnx_path, _ = locate_classifier(load_deployment_config(), store)
    except FileNotFoundError as exc:
        st.info(f"{exc}")
        return

    model = ScoutPlanInference.from_results(onnx_path, condition=settings.reference_condition,
                                            store=store)
    temperature = st.slider("temperature T (relative to the calibrated point)",
                            min(settings.temperatures), max(settings.temperatures), 1.0, 0.1)
    upload = st.file_uploader("leaf image", type=["jpg", "jpeg", "png"])
    if upload is None:
        st.caption("Upload a leaf image to see the probability move while the "
                   "prediction stays put.")
        return

    from PIL import Image

    image = Image.open(upload).convert("RGB")
    st.image(image, width=260)

    result = model.predict(image, relative_temperature=temperature)
    left, middle, right = st.columns(3)
    left.metric("probability of disease", f"{result['probability']:.3f}")
    middle.metric("prediction", "diseased" if result["prediction"] else "healthy")
    right.metric("raw logit", f"{result['logit']:+.3f}")
    st.caption(
        f"Probability = sigmoid(logit / ({result['temperature']:.4f} x {temperature:.1f})). "
        f"{result['temperature']:.4f} is the temperature fitted on field imagery "
        f"({settings.reference_condition}), so T = 1 is the calibrated probability. "
        "The prediction comes from the sign of the raw logit, so it cannot move "
        "with T; only the reported confidence does. That invariance is the study's "
        "instrument."
    )


def _plan_panel(st, store, settings) -> None:
    """Panel 2: two planners on one field, at one temperature, as the sweep ran them.

    Observations are drawn from the stored PlantDoc ``shift_eval`` logit pool and
    scaled relative to the shift-fitted temperature, the pool and the reference the
    evaluation sweep used. No torch is imported.
    """
    st.subheader("2. Plan")
    from scoutfield.app import demo

    temperature = st.slider("planner temperature T (relative to the calibrated point)",
                            min(settings.temperatures), max(settings.temperatures), 1.0,
                            0.1, key="plan_t")
    seed = st.number_input("evaluation seed", 0, 9999, 0, key="plan_seed")

    if not st.button("run both planners"):
        return

    pool = demo.load_pool(settings.pool_split, store=store)
    reference = demo.fitted_temperature(settings.reference_condition, store=store)

    results = {}
    for name, label in (("lawnmower", "Lawnmower"), ("greedy_entropy", "GreedyEntropy")):
        results[label] = demo.run_episode(demo.make_planner(name, int(seed)), settings, pool,
                                          reference, temperature, int(seed))

    st.caption(f"Observations: PlantDoc `{settings.pool_split}` logits, divided by "
               f"{reference:.4f} x {temperature:.1f}; tau = {settings.tau}, "
               f"cluster sigma = {settings.sigma}.")
    columns = st.columns(len(results))
    for column, (name, run) in zip(columns, results.items()):
        info, env = run["info"], run["env"]
        column.markdown(f"**{name}**")
        column.metric("detections / joule", f"{info['detections_per_joule']:.4f}")
        column.metric("recall", f"{info['recall']:.3f}")
        column.metric("coverage", f"{info['coverage']:.3f}")
        column.pyplot(_path_figure(run["path"], env, name))


def _path_figure(path, env, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.imshow(env.belief, cmap="Greens", vmin=0, vmax=1)
    ax.plot(path[:, 1], path[:, 0], color=FOREST, linewidth=0.8, alpha=0.85)
    truth = np.argwhere(np.asarray(env.field.labels) == 1)
    ax.scatter(truth[:, 1], truth[:, 0], s=6, facecolors="none",
               edgecolors="#B03A2E", linewidths=0.6)
    ax.set_title(f"{title}: path over belief", fontsize=8, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def _compare_panel(st) -> None:
    """Panel 3: the measured result, read from the generated summary only."""
    st.subheader("3. Compare")
    from scoutfield.utils.paths import results_dir

    summary_path = results_dir() / "summary.json"
    if not summary_path.exists():
        st.info(f"No `{summary_path.name}` yet. Run `experiments/04_sweep.py` to "
                "completion; this panel never hard-codes a number.")
        return

    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    rows = []
    for cell in summary["cells"].values():
        rows.append({
            "agent": cell["agent"],
            "temperature": cell["temperature"],
            "detections_per_joule": cell["detections_per_joule"]["value"],
            "recall": cell.get("recall", {}).get("value"),
            "ece": cell.get("ece"),
        })
    st.dataframe(sorted(rows, key=lambda r: (r["agent"], r["temperature"])))

    degradation = summary.get("degradation", {})
    if degradation:
        st.markdown("**Degradation, calibrated point against T = 4**")
        for agent, entry in sorted(degradation.items()):
            ratio = entry.get("ratio")
            st.write(f"- `{agent}`: "
                     + ("undefined (zero detections at T=4)" if ratio is None
                        else f"{ratio:.2f}x"))
        st.caption(
            "Reported as measured. If the learned planner loses, this table shows "
            "it losing — the pilot reported an underperforming agent and two "
            "refuted hypotheses in plain terms, and that standard holds here."
        )


def main() -> None:
    import streamlit as st

    from scoutfield.app.demo import load_settings
    from scoutfield.utils.artifacts import get_store

    st.set_page_config(page_title="ScoutPlan", page_icon=None, layout="wide")
    st.title("ScoutPlan")
    st.caption(
        "Does a classifier's miscalibration change where a drone should fly? "
        "The demonstration is the coupling, not the classifier."
    )

    store, settings = get_store(), load_settings()
    _classify_panel(st, store, settings)
    st.divider()
    _plan_panel(st, store, settings)
    st.divider()
    _compare_panel(st)


if __name__ == "__main__":
    main()
