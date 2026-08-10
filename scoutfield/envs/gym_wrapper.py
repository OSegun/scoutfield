"""
Gymnasium adapter for Stable-Baselines3.

The pilot's environment returns a 4-tuple ``(obs, reward, done, info)``.
Gymnasium and SB3 expect a 5-tuple ``(obs, reward, terminated, truncated, info)``
and a ``reset`` returning ``(obs, info)``.

That translation lives here and only here. Putting it in ``FieldScoutEnv`` would
mean the pilot's own baseline planners could no longer run in it unmodified,
which would break the comparison this project exists to make.

The terminated / truncated distinction matters
----------------------------------------------
It is not bookkeeping. SB3 bootstraps the value function on truncation and does
not on termination. Getting it backwards produces a value function that is wrong
in a way that still trains, still improves, and quietly caps final performance —
one of the harder RL bugs to notice.

Episodes here end because the energy budget is exhausted. That is a genuine
terminal state of the MDP: there is no future return to bootstrap, because the
drone cannot move again. So ``terminated=True, truncated=False``. Only a wall
clock or step-limit cutoff imposed from outside would be truncation.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required: pip install -e '.[planning]'") from exc


class GymScoutEnv(gym.Env):
    """Wraps a FieldScoutEnv in the Gymnasium API.

    Holds no logic of its own beyond the API translation. If something needs
    changing about how the environment behaves, it belongs in FieldScoutEnv.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, make_env, seed: int | None = None):
        """``make_env(seed)`` is a factory returning a fresh FieldScoutEnv.

        A factory rather than an instance because SB3's vectorised environments
        construct one per subprocess, and each needs its own independently seeded
        field and classifier — sharing them across subprocesses would correlate
        the runs and silently narrow the confidence intervals.
        """
        self._make_env = make_env
        self.env = make_env(seed)
        self.action_space = spaces.Discrete(self.env.n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.obs_dim,), dtype=np.float64
        )

    def reset(self, *, seed: int | None = None, options=None):
        """Rebuild through the factory when seeded, so the field layout and the
        classifier's image draws are both reproducible.

        Reseeding only the action sampler is not enough: the field and the
        classifier each carry their own generator, and leaving those untouched
        would make two "identically seeded" episodes differ.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.env = self._make_env(seed)
        obs = self.env.reset()
        return np.asarray(obs, dtype=np.float64), self.env.info()

    def step(self, action: int):
        """Translate the pilot's 4-tuple into the Gymnasium 5-tuple.

        ``done`` maps to ``terminated``; ``truncated`` stays False. See the module
        docstring for why that is not arbitrary.
        """
        obs, reward, done, info = self.env.step(int(action))
        return np.asarray(obs, dtype=np.float64), float(reward), bool(done), False, info
