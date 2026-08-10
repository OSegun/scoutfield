"""
scoutfield
==========
Calibration-aware informative path planning for energy-constrained aerial crop scouting.

This package replaces the two synthetic components of the pilot study with real ones:

    pilot (scoutplan @ v1.0.0)           this package
    -----------------------------------  ------------------------------------
    perception.CalibratedClassifier   ->  perception.adapter.CNNClassifier
      (temperature-scaled Gaussian          (fine-tuned EfficientNet-B0,
       surrogate)                            temperature-scaled logits)

    agents.ReinforceAgent             ->  planners.ppo.PPOPlanner
      (NumPy REINFORCE, 12x12 field)        (Stable-Baselines3, 32x32 field)

Everything else — the Neyman-Scott field, the Bayesian belief update, the energy model,
the four baseline planners — is imported from the pilot, never copied. The pilot is
pinned to tag v1.0.0 so the baseline this work is measured against cannot drift.

This is a library: importable, side-effect free, no CLI. Runnable entry points live in
``experiments/``.

Note on imports: the pilot ships flat top-level modules. After installation ``import env``
and ``import perception`` resolve to the *pilot*. Anything defined here is always
``scoutfield.<...>``.
"""

__version__ = "0.1.0.dev0"

#: The pilot tag this package is developed against. Verified by
#: tests/test_pilot_contract.py rather than at import time, to keep import cheap.
PILOT_TAG = "v1.0.0"
PILOT_REPO = "https://github.com/OSegun/scoutplan"
