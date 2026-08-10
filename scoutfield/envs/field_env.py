"""
FieldScoutEnv: the pilot's ScoutEnv at a scale where planning can pay off.

Two changes from the pilot, and the reasoning for each
------------------------------------------------------
**Scale: 12x12 -> 32x32 with coverage capped near 40%.** The pilot ran a 12x12
grid with BUDGET = 190, which permitted close to total coverage. When you can
visit almost every cell, there is nothing for selectivity to buy, and a fixed
lawnmower sweep is close to optimal by construction. That is one of the three
identified reasons the learned planner lost at every temperature. Scaling the
field while holding the budget so that at most ~40% of cells are reachable forces
genuine choices about where not to go — the regime where informative path
planning can actually win, or fail for a reason worth reporting.

**Observation: add a downsampled global belief map.** The pilot's agent saw a 5x5
local patch plus three scalars. An agent that cannot see beyond its immediate
neighbourhood cannot plan a route to a distant high-belief region; it can only
hill-climb. Adding an 8x8 downsampled view of the full belief grid gives it the
global structure it needs, at a fixed observation cost that does not grow with
field size.

What must NOT change
--------------------
The energy model, the Bayesian belief update, the reward weights and the ``info``
dict are inherited unchanged. The pilot's fourth invariant — "Baselines get the
same observation and pay the same energy as the learned agent. Never advantage
the proposed method through the interface" — means the baselines run in *this*
environment too, seeing this same observation. Do not build a separate, easier
environment for the baselines.
"""

from __future__ import annotations

import numpy as np

# The pilot ships flat top-level modules; `env` here is the PILOT's module.
from env import ScoutEnv


class FieldScoutEnv(ScoutEnv):
    """ScoutEnv scaled to a field where coverage is genuinely constrained.

    Parameters beyond the pilot's:

    global_map_size
        Side length of the downsampled belief map appended to the observation.
        Fixed rather than proportional to the grid, so observation dimension does
        not change if the field is scaled again later — a policy trained at one
        field size remains at least structurally loadable at another.
    include_global_map
        Ablation switch. Roadmap item 5 needs the with/without comparison to
        attribute any improvement to this change rather than to the scale change,
        and the two are otherwise confounded.
    """

    def __init__(self, *args, global_map_size: int = 8,
                 include_global_map: bool = True, **kwargs):
        self.global_map_size = int(global_map_size)
        self.include_global_map = bool(include_global_map)
        super().__init__(*args, **kwargs)

    # -------------------------------------------------------------- overrides

    def _coarse_belief(self) -> np.ndarray:
        """Mean-pool the belief grid down to ``global_map_size`` per side.

        Mean, not max: max pooling would let one confident cell dominate a whole
        region and hide the uncertainty structure the planner is meant to exploit.

        Grids that do not divide evenly are padded with the prior rather than
        with zeros. Zero means "certainly healthy", which is a claim the agent has
        no evidence for, and it would make the field edges look attractive to skip.
        """
        g = self.global_map_size
        n = self.n
        block = int(np.ceil(n / g))
        padded_n = block * g

        padded = np.full((padded_n, padded_n), self.prior, dtype=float)
        padded[:n, :n] = self.belief
        return padded.reshape(g, block, g, block).mean(axis=(1, 3))

    def _coarse_position(self) -> np.ndarray:
        """The agent's own position on the coarse grid.

        Without it the belief map is unanchored: the policy would see where the
        interesting regions are but not where it is relative to them, which is
        exactly the routing information the map was added to supply.

        Zero-padded, unlike the belief map — here zero means "not here", which is
        a claim the agent can make.
        """
        g = self.global_map_size
        block = int(np.ceil(self.n / g))
        r, c = self.pos
        coarse = np.zeros((g, g), dtype=float)
        coarse[min(r // block, g - 1), min(c // block, g - 1)] = 1.0
        return coarse

    def _obs(self) -> np.ndarray:
        """Pilot observation, optionally with a downsampled global belief map.

        Appending rather than replacing keeps the local patch, which carries the
        fine detail needed for the confirmation decision. The global map only
        supplies routing information.
        """
        base = super()._obs()
        if not self.include_global_map:
            return base
        return np.concatenate([
            base,
            self._coarse_belief().ravel(),
            self._coarse_position().ravel(),
        ]).astype(np.float64)

    @property
    def obs_dim(self) -> int:
        """Observation dimension. Asserted in tests/test_env_parity.py so a
        silent change cannot invalidate a trained policy."""
        base = 3 * self.patch * self.patch + 3
        if not self.include_global_map:
            return base
        return base + 2 * self.global_map_size * self.global_map_size


def make_field_env(config, seed: int = 0, temperature: float = 1.0,
                   tau: float | None = None, sigma: float | None = None,
                   classifier=None, checkpoint=None, **overrides):
    """Build a ``FieldScoutEnv`` from a config, seeded reproducibly.

    One factory shared by PPO training, the evaluation sweep and the τ sweep, so
    every planner meets an identically constructed environment. Building the
    environment twice, in two places, is how a baseline quietly ends up facing a
    different problem than the proposed method.

    ``classifier`` may be injected directly — tests do this with a synthetic logit
    pool. Otherwise it is built from the perception checkpoint. There is no
    surrogate fallback: a run that silently substitutes synthetic logits for a
    real classifier would produce a number indistinguishable from a measured one.

    A run must be reproducible from ``(agent, seed, T, tau, sigma)``, so every one
    of those is threaded through explicitly rather than read from global state.
    """
    from field import DiseaseField

    env_cfg = config.section("env")
    tau = float(env_cfg["detect_threshold"] if tau is None else tau)
    sigma = float(env_cfg["cluster_sigma"] if sigma is None else sigma)

    if classifier is None:
        from scoutfield.perception.adapter import CNNClassifier
        from scoutfield.utils.paths import checkpoints_dir

        checkpoint = checkpoint or (checkpoints_dir("perception") / "best.pt")
        classifier = CNNClassifier(
            checkpoint=checkpoint,
            temperature=temperature,
            reference_temperature=_reference_temperature(),
            rng=np.random.default_rng(seed),
        )
    else:
        classifier.set_temperature(temperature)

    field = DiseaseField(
        size=int(env_cfg["grid"]),
        sigma=sigma,
        n_parents=int(env_cfg["n_parents"]),
        offspring_mean=int(env_cfg["offspring_mean"]),
        rng=np.random.default_rng(seed),
    )

    observation = env_cfg.get("observation", {})
    kwargs = {
        "budget": float(env_cfg["budget"]),
        "patch": int(env_cfg["patch"]),
        "prior": float(env_cfg["prior"]),
        "detect_threshold": tau,
        "alpha": float(env_cfg["alpha"]),
        "lam": float(env_cfg["lam"]),
        "mu": float(env_cfg["mu"]),
        "global_map_size": int(observation.get("global_map_size", 8)),
        "include_global_map": bool(observation.get("global_belief_map", True)),
        "rng": np.random.default_rng(seed),
    }
    kwargs.update(overrides)
    return FieldScoutEnv(field, classifier, **kwargs)


def _reference_temperature() -> float:
    """The temperature fitted on validation, read from the calibration results.

    Read rather than passed so the planner sweep and the calibration run cannot
    disagree about where the calibrated point is. Falls back to 1.0 only when
    calibration has not been run, which is the case during environment tests.
    """
    import json

    from scoutfield.utils.paths import results_dir

    path = results_dir() / "calibration_sweep.json"
    if not path.exists():
        return 1.0
    with path.open("r", encoding="utf-8") as fh:
        return float(json.load(fh).get("fitted_temperature", 1.0))


def coverage_ceiling(grid: int, budget: float, e_hover: float = 1.0,
                     e_translate: float = 0.6) -> float:
    """Upper bound on the fraction of cells reachable within the budget.

    Used to verify that ``configs/ppo_field32.yaml`` actually produces the
    intended ~40% regime. Assuming it does, rather than checking, is how the
    pilot ended up with a budget that permitted near-total coverage.

    The bound is optimistic — it assumes every step is a unit move to an unvisited
    cell with no revisits — so the measured coverage will be lower. Report both.
    """
    steps = budget / (e_hover + e_translate)
    return min(1.0, steps / (grid * grid))
