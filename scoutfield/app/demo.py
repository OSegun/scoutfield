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
