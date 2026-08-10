"""
Planner contract tests.

Every planner — the pilot's baselines, the oracle and PPO — goes through one
evaluation path, so they must all satisfy one interface: ``act(obs, env) -> int``
with an optional ``reset(env)``. A planner that needs special handling in the
sweep is a planner that is not being compared on equal terms.

The PPO test trains for a few hundred steps. That is not convergence and is not
meant to be: it checks that the training path runs, checkpoints and resumes, which
is the part that breaks silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_env_parity import make_field_env


def _synthetic_classifier(seed: int = 0):
    from scoutfield.perception.adapter import CNNClassifier

    rng = np.random.default_rng(seed)
    pool = {0: rng.normal(-0.9, 1.0, 4000), 1: rng.normal(0.9, 1.0, 4000)}
    return CNNClassifier(logit_pool=pool, temperature=1.0, rng=rng)


def _run(agent, env) -> dict:
    obs = env.reset()
    if hasattr(agent, "reset"):
        agent.reset(env)
    done = False
    info: dict = {}
    while not done:
        action = agent.act(obs, env)
        assert isinstance(action, (int, np.integer)), type(action)
        assert 0 <= int(action) < env.n_actions, action
        obs, _, done, info = env.step(int(action))
    return info


@pytest.mark.parametrize("name", ["lawnmower", "random", "greedy_entropy", "oracle"])
def test_every_planner_satisfies_the_agent_contract(name):
    """One interface for all planners, learned or not."""
    from agents import GreedyEntropyAgent, LawnmowerAgent, RandomAgent

    from scoutfield.planners.oracle import OraclePlanner

    rng = np.random.default_rng(0)
    agent = {
        "lawnmower": lambda: LawnmowerAgent(rng),
        "random": lambda: RandomAgent(rng),
        "greedy_entropy": lambda: GreedyEntropyAgent(rng),
        "oracle": OraclePlanner,
    }[name]()

    info = _run(agent, make_field_env(seed=0, budget=120.0))
    for key in ("recall", "precision", "detections_per_joule",
                "false_alarms", "coverage", "time_to_first_detection"):
        assert key in info


def test_oracle_bounds_the_baselines_on_detections_per_joule():
    """The oracle is a ceiling, not a competitor.

    A baseline beating it would mean the oracle is not seeing ground truth, or is
    paying different energy — either way the regret it defines would be wrong.
    """
    from agents import LawnmowerAgent

    from scoutfield.planners.oracle import OraclePlanner

    oracle = _run(OraclePlanner(), make_field_env(seed=0, budget=240.0))
    lawn = _run(LawnmowerAgent(np.random.default_rng(0)),
                make_field_env(seed=0, budget=240.0))
    assert oracle["detections_per_joule"] >= lawn["detections_per_joule"]


def test_regret_is_zero_when_matching_the_oracle():
    from scoutfield.planners.oracle import regret

    assert regret(0.5, 0.5) == 0.0
    assert regret(0.25, 0.5) == pytest.approx(0.5)
    # A planner cannot have negative regret; the oracle is a ceiling.
    assert regret(0.75, 0.5) == 0.0
    with pytest.raises(ValueError):
        regret(0.5, 0.0)


def test_ppo_trains_checkpoints_and_resumes(tmp_path, monkeypatch):
    """The training path runs, saves, and resumes from what it saved.

    Kaggle kills sessions, so a training loop that cannot resume is a bug. This
    is the cheapest possible version of that check — a few hundred steps, not
    convergence.
    """
    from scoutfield.config import load_config
    from scoutfield.planners.ppo import PPOPlanner, train_ppo

    # `train_ppo` imports these at call time, so patching the source module is
    # enough. Redirecting them keeps a test from overwriting a real run's
    # checkpoint — resuming a real sweep from a 256-step smoke model would be
    # both silent and badly wrong.
    monkeypatch.setattr("scoutfield.utils.paths.checkpoints_dir",
                        lambda *p: _mk(tmp_path, "checkpoints", *p))
    monkeypatch.setattr("scoutfield.utils.paths.results_dir",
                        lambda: _mk(tmp_path, "results"))

    cfg = load_config("configs/ppo_field32.yaml")
    cfg.data["ppo"].update({"n_envs": 1, "n_steps": 128, "batch_size": 64,
                            "checkpoint_freq": 10_000})
    clf = _synthetic_classifier()

    first = train_ppo(cfg, seed=0, classifier=clf, total_timesteps=256)
    assert first.exists()

    resume_marker = first.parent / "ppo_seed0_resume.zip"
    assert resume_marker.exists(), "no resume checkpoint written"

    second = train_ppo(cfg, seed=0, classifier=clf, total_timesteps=128)
    assert second.exists()

    planner = PPOPlanner(second)
    info = _run(planner, make_field_env(seed=1, budget=120.0))
    assert 0.0 <= info["coverage"] <= 1.0


def _mk(root, *parts):
    p = root.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
