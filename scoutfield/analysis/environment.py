"""
Properties of the simulated field, measured rather than asserted.

Aggregation is the dial the pilot's Neyman-Scott field turns with ``sigma``, and the
pilot measured it with the Morisita index rather than assuming it. The same
measurement is repeated at 32x32 with the pilot's own implementation,
``DiseaseField.morisita_index``, imported from the pinned pilot's ``field`` module,
so the two phases describe aggregation with one definition.
"""

from __future__ import annotations

import numpy as np


def field_statistics(env_cfg: dict, sigmas, n_seeds: int) -> dict:
    """Prevalence and Morisita index per cluster scale, over ``n_seeds`` fields.

    ``env_cfg`` is the ``env`` section of ``configs/ppo_field32.yaml``. Returns
    ``{"<sigma>": {"morisita_mean", "morisita_sd", "prevalence_mean", "n_seeds"}}``.
    """
    from field import DiseaseField  # the PILOT's flat module

    out = {}
    for sigma in sigmas:
        fields = [
            DiseaseField(size=int(env_cfg["grid"]), sigma=float(sigma),
                         n_parents=int(env_cfg["n_parents"]),
                         offspring_mean=int(env_cfg["offspring_mean"]),
                         rng=np.random.default_rng(seed))
            for seed in range(int(n_seeds))
        ]
        morisita = np.array([f.morisita_index() for f in fields], dtype=float)
        prevalence = np.array([f.prevalence for f in fields], dtype=float)
        out[f"{float(sigma):g}"] = {
            "morisita_mean": float(np.nanmean(morisita)),
            "morisita_sd": float(np.nanstd(morisita)),
            "prevalence_mean": float(prevalence.mean()),
            "n_seeds": int(n_seeds),
        }
    return out
