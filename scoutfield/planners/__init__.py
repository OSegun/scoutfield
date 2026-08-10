"""
Planners.

The non-learned baselines — Lawnmower, Random, GreedyEntropy — and the pilot's
NumPy REINFORCE are imported from the pilot unchanged. Reimplementing them here
would mean the comparison is against this project's version of the baseline
rather than the published one, which is exactly the drift the dependency pin
exists to prevent.

New here: PPO (replacing the pilot's NumPy REINFORCE) and a ground-truth oracle
that makes regret measurable.

All planners obey the pilot's Agent contract: ``act(obs, env) -> int``, with an
optional ``reset(env)``.
"""

# Imported from the PILOT package, not redefined. The pilot ships no Spiral agent
# despite one being named in earlier drafts of this project's README.
from agents import (  # noqa: F401
    Agent,
    GreedyEntropyAgent,
    LawnmowerAgent,
    RandomAgent,
    ReinforceAgent,
)

from scoutfield.planners.oracle import OraclePlanner  # noqa: F401
from scoutfield.planners.ppo import PPOPlanner  # noqa: F401
