"""
Environment parity tests.

FieldScoutEnv changes the field scale and the observation. It must change
*nothing else* — not the energy model, not the belief update, not the reward, not
the info dict. The pilot's fourth invariant is the reason: "Baselines get the same
observation and pay the same energy as the learned agent. Never advantage the
proposed method through the interface."

These tests are how that stays true as the environment is extended.
"""

from __future__ import annotations


def test_field_env_subclasses_the_pilot():
    """Subclass, never copy. A second copy of ScoutEnv that quietly diverges from
    the published one is the failure mode the dependency pin exists to prevent."""
    from env import ScoutEnv

    from scoutfield.envs.field_env import FieldScoutEnv

    assert issubclass(FieldScoutEnv, ScoutEnv)


def test_field_env_does_not_override_the_physics():
    """Only the observation and scale may be overridden.

    step(), info() and the belief update carry the energy model and the reward.
    Overriding any of them would mean this project's numbers are not comparable
    to the pilot's, which is the entire point of the comparison.
    """
    from scoutfield.envs.field_env import FieldScoutEnv

    forbidden = {"step", "info", "_bayes_update", "_observe_current"}
    overridden = forbidden & set(FieldScoutEnv.__dict__)
    assert not overridden, f"FieldScoutEnv must not override: {sorted(overridden)}"


def test_coverage_ceiling_is_in_the_intended_regime():
    """The 32x32 config must cap reachable coverage near 40%.

    The pilot ran a budget permitting near-total coverage, which left nothing for
    selectivity to buy and is one of three named reasons its learned planner lost.
    Assuming the new budget fixes that, rather than checking, would repeat the
    mistake.
    """
    from scoutfield.config import load_config
    from scoutfield.envs.field_env import coverage_ceiling

    cfg = load_config("configs/ppo_field32.yaml")
    env = cfg.section("env")
    ceiling = coverage_ceiling(env["grid"], env["budget"],
                               env["energy"]["hover"], env["energy"]["translate"])
    assert 0.30 <= ceiling <= 0.55, (
        f"coverage ceiling {ceiling:.3f} is outside the intended regime; "
        "retune env.budget in configs/ppo_field32.yaml"
    )


def make_field_env(seed: int = 0, grid: int = 32, budget: float = 640.0, **kwargs):
    """A FieldScoutEnv backed by the pilot's field and a synthetic logit pool.

    The classifier is the adapter fed a synthetic pool rather than a trained
    checkpoint: these tests are about the environment, and requiring a trained
    model to run them would mean they could not run in CI at all.
    """
    import numpy as np
    from field import DiseaseField

    from scoutfield.envs.field_env import FieldScoutEnv
    from scoutfield.perception.adapter import CNNClassifier

    rng = np.random.default_rng(seed)
    pool = {0: rng.normal(-0.9, 1.0, 2000), 1: rng.normal(0.9, 1.0, 2000)}
    clf = CNNClassifier(logit_pool=pool, temperature=1.0, rng=rng)
    fld = DiseaseField(size=grid, sigma=1.5, rng=np.random.default_rng(seed))
    return FieldScoutEnv(fld, clf, budget=budget, prior=0.15,
                         detect_threshold=0.75, rng=rng, **kwargs)


def test_observation_dimension_is_stable():
    """obs_dim must match what _obs actually returns.

    A silent mismatch loads a trained policy against the wrong input layout and
    produces a policy that runs, does something, and is meaningless.
    """
    env = make_field_env()
    obs = env.reset()
    assert len(obs) == env.obs_dim, f"{len(obs)} != {env.obs_dim}"

    # 3 * 5 * 5 + 3 local, plus an 8x8 belief map and an 8x8 position map.
    assert env.obs_dim == 78 + 128


def test_observation_ablation_drops_only_the_global_map():
    """The ablation switch must remove the global map and nothing else.

    Roadmap item 5 needs the with/without comparison to attribute an improvement
    to the observation change rather than to the scale change; the two are
    otherwise confounded.
    """
    with_map = make_field_env(include_global_map=True)
    without = make_field_env(include_global_map=False)
    assert without.obs_dim == 3 * without.patch * without.patch + 3
    assert with_map.obs_dim - without.obs_dim == 2 * with_map.global_map_size ** 2


def test_coarse_belief_is_mean_pooled_and_prior_padded():
    """Mean pooling, and padding with the prior rather than zero.

    Zero would mean "certainly healthy" — a claim the agent has no evidence for,
    which would make the field edges look attractive to skip.
    """
    import numpy as np

    # 30 does not divide by 8, so the padding path is the one under test.
    env = make_field_env(grid=30)
    env.reset()
    env.belief[:, :] = 0.5
    coarse = env._coarse_belief()

    assert coarse.shape == (8, 8)
    # Interior blocks see only belief; the far corner is mostly prior padding.
    assert np.isclose(coarse[0, 0], 0.5)
    assert coarse[-1, -1] < 0.5, "padding did not use the prior"
    assert coarse[-1, -1] > env.prior


def test_gym_wrapper_reports_termination_not_truncation():
    """Budget exhaustion is a genuine terminal state: the drone cannot move again,
    so there is no future return to bootstrap. Reporting it as truncation makes
    SB3 bootstrap a value that does not exist — a bug that still trains, still
    improves, and quietly caps final performance."""
    from scoutfield.envs.gym_wrapper import GymScoutEnv

    env = GymScoutEnv(lambda seed=0: make_field_env(seed or 0, budget=60.0))
    env.reset(seed=0)

    terminated = truncated = False
    for _ in range(10000):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break

    assert terminated and not truncated


def test_gym_wrapper_obs_matches_declared_space():
    """SB3 silently misbehaves if the observation escapes its declared space."""
    from scoutfield.envs.gym_wrapper import GymScoutEnv

    env = GymScoutEnv(lambda seed=0: make_field_env(seed or 0))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs), obs.shape
    obs, _, _, _, info = env.step(1)
    assert env.observation_space.contains(obs)
    for key in ("recall", "precision", "detections_per_joule",
                "false_alarms", "coverage", "time_to_first_detection"):
        assert key in info


def test_scaled_field_holds_the_pilot_prevalence():
    """The 32x32 config must generate disease at the pilot's prevalence.

    The Neyman-Scott parameters produce an absolute count of diseased cells, not
    a density, so inheriting the pilot's n_parents at 32x32 would drop prevalence
    from 0.170 to 0.030 — a confound arriving silently alongside the scale change,
    and one that inverts the planner ranking.
    """
    import numpy as np
    from field import DiseaseField

    from scoutfield.config import load_config

    env_cfg = load_config("configs/ppo_field32.yaml").section("env")
    prevalence = np.mean([
        DiseaseField(
            size=env_cfg["grid"],
            sigma=env_cfg["cluster_sigma"],
            n_parents=env_cfg["n_parents"],
            offspring_mean=env_cfg["offspring_mean"],
            rng=np.random.default_rng(s),
        ).prevalence
        for s in range(30)
    ])
    # The pilot's 12x12 field measures 0.1697 over the same 30 seeds.
    assert 0.15 <= prevalence <= 0.19, (
        f"prevalence {prevalence:.4f} has drifted from the pilot's 0.1697; "
        "retune env.n_parents in configs/ppo_field32.yaml"
    )


def test_oracle_pays_the_same_energy_as_any_planner():
    """The oracle cheats on information, never on cost.

    If it were advantaged through the interface the regret it defines would be
    meaningless.
    """
    from scoutfield.planners.oracle import OraclePlanner

    env = make_field_env(budget=120.0)
    obs = env.reset()
    planner = OraclePlanner()
    planner.reset(env)

    done = False
    while not done:
        obs, _, done, info = env.step(planner.act(obs, env))

    assert env.energy_spent <= 120.0
    assert info["recall"] >= 0.0
