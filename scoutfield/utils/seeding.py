"""
Determinism.

The pilot's third invariant: "Every run is reproducible from (agent, seed, T,
sigma). No unseeded RNG, no wall-clock or PID entropy anywhere in the experiment
path." That rule carries over, and it is harder to keep here because torch and
Stable-Baselines3 each maintain their own generator.

``seed_everything`` seeds all of them from one integer and returns the NumPy
generator that experiment code should thread through explicitly. Passing a
generator beats relying on global state: it makes the dependency visible and it
survives multiprocessing, which global seeding does not.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> np.random.Generator:
    """Seed every RNG in the process and return a NumPy generator.

    ``deterministic_torch`` forces cuDNN into deterministic mode. It costs
    throughput, which matters on a limited Kaggle GPU quota, but a training run
    that cannot be reproduced is not a result. Turn it off only for exploratory
    runs whose numbers will never be reported.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return np.random.default_rng(seed)


def run_id(agent: str, seed: int, temperature: float, tau: float, sigma: float) -> str:
    """Canonical identifier for a single run.

    Used as the sweep's checkpoint key and as the row key in the results CSV, so
    a resumed sweep can tell exactly which runs are already done. The tuple is
    the complete description of a run — if a parameter is added that changes
    results, it belongs in this identifier too.
    """
    return f"{agent}__s{seed}__T{temperature:g}__tau{tau:g}__sig{sigma:g}"
