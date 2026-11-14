"""
The demonstration's logic, kept apart from the Streamlit page so it can be tested.

The demonstration must show exactly what the experiment measured, so every
condition below is read from the same config files and the same store files the
sweep used, never typed in here:

* the planners observe through the PlantDoc ``shift_eval`` logit pool
  (``pool_split`` in ``configs/sweep.yaml``), read from stage ``02_calibration``;
* the temperature a viewer chooses is *relative to* the fitted temperature of
  ``reference_condition`` in ``configs/sweep.yaml`` (the shift-fitted one), so
  T = 1 is the calibrated point, as in the sweep;
* the field, the energy model and τ come from ``configs/ppo_field32.yaml``.

The classifier is built the way ``experiments/04_sweep.py`` builds it: the pool
from ``load_or_build_logit_pool``, the reference temperature from
``_reference_temperature``, a generator seeded with the evaluation seed, and the
field seeded ``seed * 10_000 + episode``. Nothing here imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Planner names as the sweep spells them (``agents`` in configs/sweep.yaml).
BASELINES = ("lawnmower", "greedy_entropy")

#: Episode index used for the field seed, ``seed * 10_000 + episode``: the first
#: episode of each evaluation seed in the sweep.
EPISODE = 0


@dataclass(frozen=True)
class DemoSettings:
    """The experimental conditions the demonstration reproduces."""

    pool_split: str
    reference_condition: str
    temperatures: tuple[float, ...]
    tau: float
    sigma: float
    env_config: object


def load_settings(repo: Path | None = None) -> DemoSettings:
    """Read the conditions from ``configs/sweep.yaml`` and ``configs/ppo_field32.yaml``."""
    from scoutfield.config import load_config
    from scoutfield.utils.paths import repo_root

    root = repo or repo_root()
    sweep = load_config(root / "configs" / "sweep.yaml")
    env_config = load_config(root / "configs" / "ppo_field32.yaml")
    env = env_config.section("env")
    return DemoSettings(
        pool_split=str(sweep["pool_split"]),
        reference_condition=str(sweep["reference_condition"]),
        temperatures=tuple(float(t) for t in sweep["temperatures"]),
        tau=float(env["detect_threshold"]),
        sigma=float(env["cluster_sigma"]),
        env_config=env_config,
    )


def fitted_temperature(condition: str, store=None) -> float:
    """The fitted temperature of ``condition``, from stage 02 (the sweep's reader)."""
    from scoutfield.envs.field_env import _reference_temperature

    return _reference_temperature(condition, store=store)


def load_pool(pool_split: str, store=None) -> dict[int, np.ndarray]:
    """The stage 02 logit pool, verified against the recorded split (the sweep's reader)."""
    from scoutfield.perception.adapter import load_or_build_logit_pool

    return load_or_build_logit_pool(pool_split, store=store)


def make_planner(name: str, seed: int):
    """A baseline planner from the pilot, seeded as the sweep seeds it."""
    from agents import GreedyEntropyAgent, LawnmowerAgent

    rng = np.random.default_rng(seed)
    if name == "lawnmower":
        return LawnmowerAgent(rng)
    if name == "greedy_entropy":
        return GreedyEntropyAgent(rng)
    raise KeyError(f"unknown planner {name!r}; have {', '.join(BASELINES)}")


def run_episode(planner, settings: DemoSettings, pool: dict[int, np.ndarray],
                reference_temperature: float, temperature: float, seed: int,
                tau: float | None = None) -> dict:
    """One episode, built exactly as one sweep episode is built.

    Returns the final ``info`` dict, the flown path, the environment (for the
    belief map and the true field) and the classifier it observed through.
    """
    from scoutfield.envs.field_env import make_field_env
    from scoutfield.perception.adapter import CNNClassifier

    classifier = CNNClassifier(
        logit_pool=pool,
        temperature=temperature,
        reference_temperature=reference_temperature,
        rng=np.random.default_rng(seed),
        pool_split=settings.pool_split,
    )
    env = make_field_env(
        settings.env_config,
        seed=seed * 10_000 + EPISODE,
        temperature=temperature,
        tau=settings.tau if tau is None else float(tau),
        sigma=settings.sigma,
        classifier=classifier,
    )
    obs = env.reset()
    if hasattr(planner, "reset"):
        planner.reset(env)
    path = [env.pos]
    done, info = False, {}
    while not done:
        obs, _, done, info = env.step(int(planner.act(obs, env)))
        path.append(env.pos)
    return {"info": info, "path": np.asarray(path), "env": env, "classifier": classifier}


# ------------------------------------------------------------ where files live

#: Overrides ``demo.source`` in configs/deployment.yaml.
SOURCE_ENV = "SCOUTFIELD_DEMO_SOURCE"
SOURCES = ("store", "offline")
FILL_COMMAND = "python experiments/08_fill_demo_folder.py"


def deployment_config():
    from scoutfield.app.export_onnx import load_deployment_config

    return load_deployment_config()


def demo_source(config=None) -> str:
    """``store`` or ``offline``: the environment variable, else the config value."""
    import os

    config = config or deployment_config()
    source = os.environ.get(SOURCE_ENV) or str(config.section("demo")["source"])
    if source not in SOURCES:
        raise ValueError(f"{SOURCE_ENV}={source!r}; use one of {', '.join(SOURCES)}")
    return source


def offline_root(config=None) -> Path:
    from scoutfield.utils.paths import repo_root

    config = config or deployment_config()
    root = Path(config.section("demo")["offline_dir"])
    return root if root.is_absolute() else repo_root() / root


def offline_store(config=None, run_id: str | None = None):
    """The offline folder as an artefact store: same paths and manifest checks."""
    from scoutfield.utils.artifacts import ArtifactStore, LocalBackend, load_settings

    root = offline_root(config)
    run = run_id or load_settings()["run_id"]
    manifest = root / "store" / "runs" / run / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"the offline folder has no files for run {run}: looked for {manifest}. "
            f"Fill it once while online with `{FILL_COMMAND}`."
        )
    return ArtifactStore(LocalBackend(root / "store"), run, root / "cache")


def open_store(config=None):
    """The store the app reads from, chosen by the ``demo.source`` setting.

    One code path for every tier: the hosted tiers read the public store with no
    token; the laptop tier reads the offline folder, which has the same layout and
    is checked against the same manifest entries.
    """
    from scoutfield.utils.artifacts import get_store

    config = config or deployment_config()
    return offline_store(config) if demo_source(config) == "offline" else get_store()


def required_files(config=None, settings: DemoSettings | None = None
                   ) -> list[tuple[str, str]]:
    """Every ``(stage, name)`` the app reads from the store, apart from samples."""
    from scoutfield.perception.adapter import logit_pool_filename

    config = config or deployment_config()
    settings = settings or load_settings()
    stage = str(config["stage"])
    summary = config.section("demo")["summary"]
    return [
        ("02_calibration", "calibration_sweep.json"),
        ("02_calibration", logit_pool_filename(settings.pool_split)),
        (str(summary["stage"]), str(summary["name"])),
        (stage, str(config.section("export")["classifier_file"])),
        (stage, str(config.section("policy")["policy_file"])),
    ]


def sample_files(store, config=None) -> list[tuple[str, str]]:
    """The sample images filed in the store, with their list, if any were filed."""
    config = config or deployment_config()
    stage, name = str(config["stage"]), str(config.section("demo")["samples"]["file"])
    if not store.has(stage, name):
        return []
    listed = store.pull_json(stage, name)
    return [(stage, name)] + [(stage, s["store_name"]) for s in listed["samples"]]


def fill_offline(source_store, config=None, local_samples: Path | None = None) -> dict:
    """Copy every file the app reads from ``source_store`` into the offline folder.

    Each file is pulled through the store (so its SHA-256 is checked against the
    run's manifest) and written at the same path under ``<offline_dir>/store``,
    with a manifest holding the same entries, so the offline app verifies every
    read exactly as the online one does. ``local_samples`` is a folder of
    laptop-only samples (with their list), copied to ``<offline_dir>/local_samples``.
    """
    import json
    import shutil

    config = config or deployment_config()
    root = offline_root(config)
    target = root / "store" / "runs" / source_store.run_id
    if target.exists():
        shutil.rmtree(target)
    wanted = required_files(config) + sample_files(source_store, config)
    full = source_store.manifest()
    subset = {"run_id": source_store.run_id, "filled_from": source_store.backend.description,
              "stages": {}}
    for stage, name in wanted:
        local = source_store.pull(stage, name)
        dest = target / stage / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
        entry = full["stages"][stage]["files"][name]
        subset["stages"].setdefault(stage, {"files": {}})["files"][name] = entry
    (target / "manifest.json").write_text(json.dumps(subset, indent=2), encoding="utf-8")

    local_dir = root / "local_samples"
    if local_dir.exists():
        shutil.rmtree(local_dir)
    copied = 0
    if local_samples is not None and Path(local_samples).is_dir():
        shutil.copytree(local_samples, local_dir)
        copied = sum(1 for p in local_dir.iterdir() if p.suffix.lower() != ".json")
    return {"root": root, "files": wanted, "local_samples": copied}


def local_samples(config=None) -> list[dict]:
    """Laptop-only samples in the offline folder, each with its ``path``."""
    import json

    config = config or deployment_config()
    folder = offline_root(config) / "local_samples"
    listing = folder / str(config.section("demo")["samples"]["file"])
    if not listing.is_file():
        return []
    samples = json.loads(listing.read_text(encoding="utf-8"))["samples"]
    return [{**s, "path": folder / s["store_name"]} for s in samples]


# ------------------------------------------------------------------ planners

PLANNERS = ("lawnmower", "greedy_entropy", "ppo")
LABELS = {"lawnmower": "Lawnmower", "greedy_entropy": "GreedyEntropy", "ppo": "PPO"}


def load_policy(store, config=None):
    """The exported PPO policy, run by onnxruntime (no torch)."""
    from scoutfield.app.export_policy import ONNXPolicy

    config = config or deployment_config()
    return ONNXPolicy(store.pull(str(config["stage"]),
                                 str(config.section("policy")["policy_file"])))


def get_planner(name: str, seed: int, policy=None):
    """Any demo planner by name; PPO needs the loaded ``policy``."""
    if name == "ppo":
        if policy is None:
            raise ValueError("the PPO planner needs the exported policy (load_policy)")
        return policy
    return make_planner(name, seed)


def _sweep_config(repo: Path | None = None):
    from scoutfield.config import load_config
    from scoutfield.utils.paths import repo_root

    return load_config((repo or repo_root()) / "configs" / "sweep.yaml")


def tau_grid(repo: Path | None = None) -> tuple[float, ...]:
    """The τ values of the τ sweep for the baselines (configs/sweep.yaml)."""
    return tuple(float(t) for t in _sweep_config(repo).section("tau_sensitivity")["tau_grid"])


def ppo_tau_grid(repo: Path | None = None) -> tuple[float, ...]:
    """The τ values at which the τ sweep evaluated PPO."""
    cfg = _sweep_config(repo).section("tau_sensitivity")
    return tuple(float(t) for t in cfg["ppo_tau_grid"])


# -------------------------------------------------------------- seed choice

def choose_seed(summary: dict, settings: DemoSettings, pool, reference_temperature: float,
                policy, candidates: int, temperatures=(1.0, 4.0)) -> dict:
    """The evaluation seed whose episodes look most like the sweep's typical result.

    For each candidate seed, one episode per planner at each temperature (τ and σ
    as configured) is compared with the sweep's IQM for that (planner, T) cell, as
    ``|episode - IQM| / IQM``. The seed with the smallest mean over the cells wins.
    """
    rows = []
    for seed in range(int(candidates)):
        errors, values = [], {}
        for name in PLANNERS:
            for t in temperatures:
                key = f"{name}@T{t:g}"
                cell = summary["cells"][key]["detections_per_joule"]["value"]
                run = run_episode(get_planner(name, seed, policy), settings, pool,
                                  reference_temperature, t, seed)
                got = float(run["info"]["detections_per_joule"])
                values[key] = got
                errors.append(abs(got - cell) / cell)
        rows.append({"seed": seed, "mean_relative_difference": float(np.mean(errors)),
                     "detections_per_joule": values})
    best = min(rows, key=lambda r: r["mean_relative_difference"])
    return {"seed": best["seed"], "rows": rows}


# ------------------------------------------------------------------ samples

def select_samples(pool_path: Path, n_correct_diseased: int, n_correct_healthy: int,
                   n_wrong: int) -> list[dict]:
    """Pick sample images from a stored logit pool, by rule rather than by eye.

    Correct images are spread evenly across interior quantiles of their confidence
    (from fairly unsure to fairly sure); the wrong ones are the most confident mistakes,
    alternating false negatives and false positives.
    """
    with np.load(pool_path, allow_pickle=False) as data:
        pools = {int(k.split("_")[1]): (data[k], data[f"paths_{k.split('_')[1]}"])
                 for k in data.files if k.startswith("logits_")}

    def spread(idx, logits, n):
        if n <= 0 or len(idx) == 0:
            return []
        # Interior quantiles of confidence: n = 4 takes the 20th, 40th, 60th and
        # 80th percentiles, n = 1 the median, so neither extreme is over-represented.
        order = idx[np.argsort(np.abs(logits[idx]))]
        q = np.linspace(0, 1, num=n + 2)[1:-1]
        picks = (q * (len(order) - 1)).round().astype(int)
        return list(order[np.unique(picks)])

    chosen = []
    for label, n in ((1, n_correct_diseased), (0, n_correct_healthy)):
        logits, paths = pools[label]
        right = np.flatnonzero((logits > 0).astype(int) == label)
        chosen += [{"key": str(paths[i]), "label": label, "logit": float(logits[i]),
                    "correct": True} for i in spread(right, logits, n)]
    wrong = []
    for label in (1, 0):
        logits, paths = pools[label]
        bad = np.flatnonzero((logits > 0).astype(int) != label)
        bad = bad[np.argsort(-np.abs(logits[bad]))]
        wrong.append([{"key": str(paths[i]), "label": label, "logit": float(logits[i]),
                       "correct": False} for i in bad])
    for i in range(int(n_wrong)):
        source = wrong[i % 2] or wrong[(i + 1) % 2]
        if source:
            chosen.append(source.pop(0))
    return chosen


def find_image(dataset_root: Path, key: str) -> Path:
    """The file for an item key (its last three path components) under a dataset."""
    parts = Path(key).parts
    for candidate in Path(dataset_root).rglob(parts[-1]):
        if candidate.parts[-3:] == parts[-3:]:
            return candidate
    raise FileNotFoundError(f"{key} not found under {dataset_root}")
