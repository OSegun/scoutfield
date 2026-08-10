"""
Ground-truth oracle planner. Roadmap item 8.

Why an oracle is worth building
--------------------------------
The pilot could say that GreedyEntropy beat Lawnmower by 7.4x at T = 4. It could
not say how much of the *available* performance either of them captured. Without
an upper bound, "the learned planner underperformed" and "every planner
underperformed, and the learned one least badly" are indistinguishable — and they
support very different conclusions.

The oracle sees the true disease labels and plans against them under the same
energy budget. The gap between a planner and the oracle is its regret, and regret
is the quantity that makes performance across different fields, budgets and
temperatures comparable on one scale.

The oracle is not a competitor. It cheats, deliberately and visibly, and its
number is a ceiling rather than a baseline.

Optimality caveat
-----------------
Optimal routing under an energy budget is a variant of the orienteering problem
and is NP-hard, so a genuinely optimal oracle is out of reach at 32x32. A greedy
nearest-diseased-cell oracle is a *lower bound* on the true ceiling and must be
labelled as such wherever it appears. Overstating it as "optimal" would understate
every planner's regret, which is the flattering direction — reason enough to be
careful about it.
"""

from __future__ import annotations

import numpy as np

# The pilot's move table and energy constants; imported rather than restated so
# the oracle cannot drift from the costs every other planner pays.
from env import E_TRANSLATE, MOVES


class OraclePlanner:
    """Plans against the true labels under the same energy budget.

    Pays exactly the same energy costs as every other planner. The pilot's fourth
    invariant applies in reverse here: the oracle must not be advantaged through
    the interface either, or the regret it defines is meaningless.
    """

    def __init__(self, greedy: bool = True):
        self.greedy = greedy

    def reset(self, env) -> None:
        """Cache the true diseased-cell coordinates.

        Read here rather than in ``act`` so that the one place this planner
        touches ground truth is obvious.
        """
        self._targets = {(int(r), int(c))
                         for r, c in zip(*np.nonzero(np.asarray(env.field.labels) == 1))}

    def act(self, obs, env) -> int:
        """Step towards the nearest unconfirmed diseased cell.

        Distance is the energy metric — hypot weighted by ``E_TRANSLATE`` — not
        Manhattan, because diagonal moves cost more but cover more, and ranking
        by the wrong metric would make the ceiling reflect a different budget
        than the one every planner actually pays.

        The oracle still confirms detections through the classifier, so it
        inherits the same miscalibration the planners face. That isolates the
        routing ceiling from the perception ceiling. A variant with perfect
        confirmation would separate the two and is worth adding if these regret
        numbers turn out ambiguous.
        """
        remaining = self._targets - env.detected
        if not remaining:
            # Nothing left to seek. Hold position rather than spending translation
            # energy on a move with no expected value.
            return self._cheapest_move()

        r, c = env.pos
        target = min(remaining, key=lambda t: np.hypot(t[0] - r, t[1] - c))
        return self._move_towards((r, c), target)

    @staticmethod
    def _cheapest_move() -> int:
        """The move with the lowest translation cost among the pilot's 8."""
        return int(np.argmin([E_TRANSLATE * np.hypot(dr, dc) for dr, dc in MOVES]))

    @staticmethod
    def _move_towards(pos, target) -> int:
        """The action from the pilot's move table that most reduces distance."""
        r, c = pos
        best, best_d = 0, None
        for i, (dr, dc) in enumerate(MOVES):
            d = np.hypot(r + dr - target[0], c + dc - target[1])
            if best_d is None or d < best_d:
                best, best_d = i, d
        return best


def regret(planner_metric: float, oracle_metric: float) -> float:
    """Normalised regret in [0, 1]; 0 means the planner matched the oracle.

    Report regret on detections-per-joule, which is the paper's headline metric,
    and separately on recall. A planner can have low regret on one and high regret
    on the other, and the pilot's finding that overconfidence trades precision for
    recall says that gap is exactly where the interesting behaviour lives.
    """
    if oracle_metric <= 0:
        raise ValueError("oracle metric must be positive to define regret")
    return max(0.0, 1.0 - planner_metric / oracle_metric)
