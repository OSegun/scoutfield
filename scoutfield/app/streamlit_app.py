"""
ScoutField demonstration.

Run:
    make app            # files from the public artefact store (hosted tiers)
    make app-offline    # files from the offline folder (laptop, no network)

What it demonstrates
--------------------
Not "here is a leaf classifier": that is a solved problem and demonstrating it
proves nothing about this work. The demonstration is the *coupling*: that a
miscalibrated classifier changes where a planner chooses to fly and how much
disease it finds per joule.

Three panels, in this order:

1. **Classify.** A sample or uploaded leaf image; the probability, the prediction
   and the raw logit. The temperature slider is the experiment's T, relative to the
   temperature fitted on field imagery, so T = 1 is the calibrated probability. The
   prediction does not move as the slider does, only the confidence.

2. **Plan.** Lawnmower, GreedyEntropy and PPO fly the same simulated field at the
   chosen T and τ. Every observation is a real logit from the PlantDoc
   ``shift_eval`` pool, scaled as the evaluation sweep scaled it.

3. **Compare.** The measured detections per joule against T, with intervals, and
   the degradation over the three spans, read from the run's ``summary.json``:
   never hard-coded, never edited by hand.

Honesty in the interface
-------------------------
If the learned planner loses to the lawnmower baseline, the demonstration shows it
losing. It shows what the experiment measured, under the conditions it measured,
and labels what is simulated.

Where files come from is a setting (``demo.source`` in configs/deployment.yaml, or
the ``SCOUTFIELD_DEMO_SOURCE`` environment variable): the public store, or the
offline folder. The code path is the same. Nothing here imports PyTorch.

Visual identity as in ``experiments/make_figures.py``: Forest #2C5F2D, Moss #97BC62,
Tint #EEF3EA, Ink #1F2A1F.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# `streamlit run` puts this file's folder on the import path, not the repository root.
# Hosted copies run the clone without installing it (scoutfield/app/requirements.txt
# lists only third-party packages), so the root is added here to make `scoutfield`
# and the configs under it importable and findable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FOREST, MOSS, TINT, INK, MUTED = "#2C5F2D", "#97BC62", "#EEF3EA", "#1F2A1F", "#6B7A6B"
TRUTH = "#B03A2E"
PLANNER_COLOURS = {"lawnmower": "#B03A2E", "greedy_entropy": "#97BC62", "ppo": "#2C5F2D",
                   "random": "#A9B4A9", "oracle": "#1F2A1F"}
#: The oracle is a reference ceiling (a greedy lower bound on it), not a planner a
#: drone could run, so it is drawn dashed.
PLANNER_STYLES = {"oracle": "--"}


# --------------------------------------------------------------- resources

def _resources(st):
    """Everything read once per server process and shared by every session."""
    from scoutfield.app import demo

    @st.cache_resource(show_spinner=False)
    def config():
        return demo.deployment_config()

    @st.cache_resource(show_spinner="Opening the artefact store")
    def store():
        return demo.open_store(config())

    @st.cache_resource(show_spinner=False)
    def settings():
        return demo.load_settings()

    @st.cache_resource(show_spinner="Loading the classifier")
    def classifier():
        from scoutfield.app.inference import ScoutPlanInference

        cfg = config()
        path = store().pull(str(cfg["stage"]), str(cfg.section("export")["classifier_file"]))
        return ScoutPlanInference.from_results(
            path, condition=settings().reference_condition, store=store())

    @st.cache_resource(show_spinner="Loading the logit pool")
    def pool():
        return demo.load_pool(settings().pool_split, store=store())

    @st.cache_resource(show_spinner=False)
    def reference_temperature():
        return demo.fitted_temperature(settings().reference_condition, store=store())

    @st.cache_resource(show_spinner="Loading the PPO policy")
    def policy():
        return demo.load_policy(store(), config())

    @st.cache_data(show_spinner=False)
    def summary():
        cfg = config().section("demo")["summary"]
        return store().pull_json(str(cfg["stage"]), str(cfg["name"]))

    @st.cache_data(show_spinner=False)
    def samples():
        """Sample images: those filed in the store, then laptop-only ones."""
        out = []
        files = demo.sample_files(store(), config())
        if files:
            stage, listing = files[0]
            for s in store().pull_json(stage, listing)["samples"]:
                out.append({**s, "path": str(store().pull(stage, s["store_name"]))})
        if demo.demo_source(config()) == "offline":
            out += [{**s, "path": str(s["path"])} for s in demo.local_samples(config())]
        return out

    return {"config": config, "store": store, "settings": settings,
            "classifier": classifier, "pool": pool,
            "reference_temperature": reference_temperature, "policy": policy,
            "summary": summary, "samples": samples}


def _check_store(st, res) -> bool:
    """Stop with a clear message, naming where it looked, if the files are unreachable."""
    from scoutfield.app import demo

    try:
        store = res["store"]()
        manifest = store.manifest()
        missing = [f"{s}/{n}" for s, n in demo.required_files(res["config"]())
                   if n not in manifest.get("stages", {}).get(s, {}).get("files", {})]
    except Exception as exc:  # noqa: BLE001 - shown on the page, never swallowed
        from scoutfield.utils.artifacts import load_settings

        url = f"https://huggingface.co/datasets/{load_settings()['hub_repo']}"
        st.error(f"The demonstration's files could not be read. {exc}")
        st.caption(f"Hosted: the public artefact store {url} must be reachable. "
                   f"Laptop: run `{demo.FILL_COMMAND}` once while online, then start "
                   "with SCOUTFIELD_DEMO_SOURCE=offline.")
        return False
    if missing:
        st.error(f"Run {store.run_id} in {store.backend.description} does not list "
                 f"{', '.join(missing)}. File them first (docs/REPRODUCING.md, step 8).")
        return False
    return True


# ------------------------------------------------------------------ panels

def _classify_panel(st, res) -> None:
    """Panel 1: the mechanism, made visible in one slider."""
    st.subheader("1. Classify")
    settings = res["settings"]()
    model = res["classifier"]()
    samples = res["samples"]()

    from PIL import Image

    names = ["Upload your own image"] + [
        f"{s['dataset']}: {s['class_folder']} ({s['true_label']})" for s in samples]
    choice = st.selectbox("leaf image", range(len(names)), format_func=names.__getitem__,
                          index=1 if samples else 0)
    sample = samples[choice - 1] if choice > 0 else None
    if sample is None:
        upload = st.file_uploader("upload a leaf image", type=["jpg", "jpeg", "png"])
        if upload is None:
            st.caption("Upload a leaf image to see the probability move while the "
                       "prediction stays put.")
            return
        image = Image.open(upload).convert("RGB")
    else:
        with Image.open(sample["path"]) as im:
            image = im.convert("RGB")

    temperature = st.slider("temperature T (relative to the calibrated point)",
                            min(settings.temperatures), max(settings.temperatures), 1.0, 0.1)
    result = model.predict(image, relative_temperature=temperature)

    left, right = st.columns([1, 2])
    left.image(image, width=260)
    with right:
        a, b, c = st.columns(3)
        a.metric("probability of disease", f"{result['probability']:.3f}")
        b.metric("prediction", "diseased" if result["prediction"] else "healthy")
        c.metric("raw logit", f"{result['logit']:+.3f}")
        if sample is not None:
            st.caption(f"True class: **{sample['true_label']}** ({sample['class_folder']}). "
                       + ("The classifier gets this one wrong."
                          if not sample["correct"] else ""))
        st.caption(
            f"Probability = sigmoid(logit / ({result['temperature']:.4f} x {temperature:.1f})). "
            f"{result['temperature']:.4f} is the temperature fitted on held-out field "
            f"imagery ({settings.reference_condition}), so T = 1 is the calibrated "
            "probability, T < 1 overconfident and T > 1 underconfident. The prediction "
            "comes from the sign of the raw logit, so it cannot move with T; only the "
            "reported confidence does.")
    if sample is not None:
        st.caption(f"Image: {sample['licence']}")


def _plan_panel(st, res) -> None:
    """Panel 2: three planners on one field, at one T and one τ, as the sweep ran them."""
    from scoutfield.app import demo

    st.subheader("2. Plan")
    settings, cfg = res["settings"](), res["config"]()
    st.info(
        "The field is simulated: a 32 x 32 grid whose diseased cells are drawn from a "
        "clustered random process, and no drone flew. Each time a planner looks at a "
        "cell, the observation is a real logit the trained classifier produced for a "
        "real PlantDoc field photograph of that cell's class (held-out shift_eval set), "
        "scaled by the chosen T exactly as in the evaluation sweep.")

    c1, c2, c3 = st.columns(3)
    temperature = c1.select_slider("temperature T", options=list(settings.temperatures),
                                   value=1.0)
    taus = demo.tau_grid()
    tau = c2.select_slider("confirmation threshold tau", options=list(taus),
                           value=settings.tau)
    seed = c3.number_input("evaluation seed", 0, 9999,
                           int(cfg.section("demo")["default_seed"]), key="plan_seed")

    runs = _plan_runs(st, res, float(temperature), float(tau), int(seed))
    st.caption(
        f"PlantDoc `{settings.pool_split}` logits divided by "
        f"{res['reference_temperature']():.4f} x {temperature:g}; tau = {tau:g}; cluster "
        f"sigma = {settings.sigma:g}; one episode on field seed {seed}. Seed "
        f"{cfg.section('demo')['default_seed']} is the default because its episodes are "
        "closest to the sweep's typical (IQM) result at T = 1 and T = 4 "
        "(experiments/08_choose_demo_seed.py).")
    if tau not in demo.ppo_tau_grid():
        st.caption(f"In the tau sweep PPO was evaluated only at tau = "
                   f"{', '.join(f'{t:g}' for t in demo.ppo_tau_grid())}; at {tau:g} its "
                   "column is this one episode, not a measured result.")

    columns = st.columns(len(runs))
    for column, (name, run) in zip(columns, runs.items()):
        info = run["info"]
        column.markdown(f"**{demo.LABELS[name]}**"
                        + (f" ({cfg.section('policy')['checkpoint_file']})"
                           if name == "ppo" else ""))
        column.metric("detections / joule", f"{info['detections_per_joule']:.4f}")
        column.metric("recall", f"{info['recall']:.3f}")
        column.metric("false alarms", f"{int(info['false_alarms'])}")
        column.metric("coverage", f"{info['coverage']:.3f}")
        fig = _path_figure(run, demo.LABELS[name])
        column.pyplot(fig)
        _close(fig)


def _plan_runs(st, res, temperature: float, tau: float, seed: int) -> dict:
    """The three episodes for one (T, τ, seed), cached so moving back is instant."""
    from scoutfield.app import demo

    @st.cache_data(show_spinner="Flying the three planners", max_entries=64)
    def runs(temperature: float, tau: float, seed: int) -> dict:
        settings, pool = res["settings"](), res["pool"]()
        reference, policy = res["reference_temperature"](), res["policy"]()
        out = {}
        for name in demo.PLANNERS:
            run = demo.run_episode(demo.get_planner(name, seed, policy), settings, pool,
                                   reference, temperature, seed, tau=tau)
            env = run["env"]
            out[name] = {"info": {k: float(v) for k, v in run["info"].items()
                                  if np.isscalar(v)},
                         "path": run["path"], "belief": np.array(env.belief),
                         "truth": np.asarray(env.field.labels)}
        return out

    return runs(temperature, tau, seed)


def _path_figure(run, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.imshow(run["belief"], cmap="Greens", vmin=0, vmax=1)
    path = run["path"]
    ax.plot(path[:, 1], path[:, 0], color=FOREST, linewidth=0.8, alpha=0.85)
    truth = np.argwhere(run["truth"] == 1)
    ax.scatter(truth[:, 1], truth[:, 0], s=6, facecolors="none", edgecolors=TRUTH,
               linewidths=0.6)
    ax.set_title(f"{title}: path over belief", fontsize=8, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def _close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _compare_panel(st, res) -> None:
    """Panel 3: the measured result, read from the run's summary only."""
    from scoutfield.app import demo

    st.subheader("3. Compare")
    summary = res["summary"]()
    cells = summary["cells"]
    agents = [a for a in summary["agents"] if any(c["agent"] == a for c in cells.values())]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for agent in agents:
        rows = sorted((c for c in cells.values() if c["agent"] == agent),
                      key=lambda c: c["temperature"])
        t = [c["temperature"] for c in rows]
        dpj = [c["detections_per_joule"] for c in rows]
        colour = PLANNER_COLOURS.get(agent, MUTED)
        ax.plot(t, [d["value"] for d in dpj], marker="o", ms=3, color=colour,
                linestyle=PLANNER_STYLES.get(agent, "-"),
                label=demo.LABELS.get(agent, agent.title()))
        ax.fill_between(t, [d["ci_low"] for d in dpj], [d["ci_high"] for d in dpj],
                        color=colour, alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sorted({c["temperature"] for c in cells.values()}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axvline(1.0, color=MUTED, linewidth=0.6, linestyle=":")
    ax.set_xlabel("T (relative to the shift-fitted temperature)")
    ax.set_ylabel("detections per joule (IQM)")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    st.pyplot(fig)
    _close(fig)
    sweep = demo._sweep_config()
    n_runs = {c["agent"]: int(c["n_runs"]) for c in cells.values()}
    st.caption(
        f"Interquartile mean per point over {n_runs.get('lawnmower')} runs "
        f"({len(sweep['seeds'])} evaluation seeds x {len(sweep['cluster_sigmas'])} cluster "
        f"scales, {sweep['eval_episodes']} episodes each); PPO over {n_runs.get('ppo')}, "
        f"pooling its {len(sweep['ppo_train_seeds'])} trained policies. 95% stratified "
        f"bootstrap intervals shaded; tau = {res['settings']().tau:g}. The oracle (dashed) "
        "sees the true field and is a greedy lower bound on the best possible route. Both "
        "axes are logarithmic. From the run's summary.json, the same numbers as "
        "docs/RESULTS.md.")

    st.markdown("**Degradation of detections per joule, low T against high T**")
    rows = []
    for span, entry in summary.get("degradation", {}).items():
        for agent, d in sorted(entry["agents"].items()):
            ratio = d.get("ratio")
            rows.append({
                "span": f"{span} (T {entry['low']:g} to {entry['high']:g})",
                "planner": demo.LABELS.get(agent, agent.title()),
                "ratio": ("no detections at the high T" if ratio is None
                          else f"{ratio:.2f}x [{d['ci_low']:.2f}, {d['ci_high']:.2f}]"),
            })
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption("A ratio above 1 means the planner found fewer diseases per joule at the "
               "higher T. Reported as measured: if the learned planner loses, this table "
               "shows it losing.")


def main() -> None:
    import streamlit as st

    from scoutfield.app import demo

    cfg = demo.deployment_config().section("demo")
    st.set_page_config(page_title=str(cfg["page_title"]), page_icon=None, layout="wide")
    st.title(str(cfg["page_title"]))
    st.caption(f"{cfg['project_title']}. Does a classifier's miscalibration change "
               f"where a drone should fly? Method and results: {cfg['repository_url']}")

    res = _resources(st)
    if not _check_store(st, res):
        st.stop()

    _classify_panel(st, res)
    st.divider()
    _plan_panel(st, res)
    st.divider()
    _compare_panel(st, res)


if __name__ == "__main__":
    main()
