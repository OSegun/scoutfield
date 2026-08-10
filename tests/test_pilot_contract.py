"""
Contract tests against the pilot.

These are the tests that make the two-repo split safe. They fail if the pilot
dependency is missing, if its interface has drifted, or if this project's
replacement classes stop satisfying that interface.

They are cheap and they run on every push. If one fails, stop and fix it before
writing anything else — everything downstream assumes these hold.
"""

from __future__ import annotations

import inspect

import pytest

# --------------------------------------------------------------- the pilot

def test_pilot_is_installed():
    """The pilot ships flat top-level modules, so these names come from it."""
    import agents  # noqa: F401
    import env  # noqa: F401
    import field  # noqa: F401
    import perception  # noqa: F401


def test_classifier_observe_signature():
    """observe(true_label) -> (probability, prediction), one positional arg.

    The whole design rests on this. From the pilot's developer contract:
    "Swapping in the real CNN means changing only this method."
    """
    from perception import CalibratedClassifier

    sig = inspect.signature(CalibratedClassifier.observe)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["true_label"], f"pilot interface drifted: {params}"


def test_accuracy_is_invariant_to_temperature():
    """The study's entire premise, asserted to four decimal places.

    Temperature scaling divides logits by a scalar. The link is strictly
    monotone and the decision boundary sits at logit 0, so the arg-max cannot
    move: accuracy is invariant while confidence is not.
    """
    from perception import measure_calibration

    accs = {measure_calibration(0.816, T, n=20000, seed=1)["accuracy"]
            for T in (0.3, 1.0, 4.0)}
    assert len(accs) == 1, f"instrument broken: {accs}"


def test_ece_is_minimised_at_unit_temperature():
    """ECE must rise in both directions from T = 1.

    If it becomes monotone in T, the temperature parameterisation has been
    broken — and the pilot's refuted hypothesis H2, that performance is monotone
    in ECE, would no longer be testable.
    """
    from perception import measure_calibration

    ece = {T: measure_calibration(0.816, T, n=20000, seed=1)["ece"]
           for T in (0.3, 1.0, 4.0)}
    assert ece[1.0] < ece[0.3], ece
    assert ece[1.0] < ece[4.0], ece


def test_env_info_keys_are_present():
    """Figures and analysis read these keys by name. Losing one breaks them
    silently, producing a KeyError far from the cause."""
    from env import ScoutEnv

    src = inspect.getsource(ScoutEnv.info)
    for key in ("recall", "precision", "detections_per_joule",
                "false_alarms", "coverage", "time_to_first_detection"):
        assert key in src, f"pilot info dict lost '{key}'"


# ----------------------------------------------------- this project's side

def test_cnn_classifier_matches_the_contract():
    """CNNClassifier.observe must have the same signature as the pilot's.

    Checked on the class, not an instance, so this test is meaningful before
    the implementation exists and starts failing the moment it drifts.
    """
    from perception import CalibratedClassifier

    from scoutfield.perception.adapter import CNNClassifier

    pilot = [p for p in inspect.signature(CalibratedClassifier.observe).parameters
             if p != "self"]
    ours = [p for p in inspect.signature(CNNClassifier.observe).parameters
            if p != "self"]
    assert ours == pilot, f"adapter drifted from the contract: {ours} != {pilot}"


def _synthetic_pool(mu: float = 0.9, n: int = 4000, seed: int = 0):
    """A logit pool with known separation, standing in for a trained network.

    The adapter's contract is about the shape and temperature-independence of what
    `observe` returns, none of which depends on the logits coming from a real
    network — so this exercises the contract without a checkpoint.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    return {0: rng.normal(-mu, 1.0, n), 1: rng.normal(mu, 1.0, n)}


def test_cnn_classifier_returns_probability_and_prediction():
    """Returns (float in [0, 1], int in {0, 1})."""
    from scoutfield.perception.adapter import CNNClassifier

    clf = CNNClassifier(logit_pool=_synthetic_pool(), temperature=1.0)
    p, yhat = clf.observe(1)
    assert isinstance(p, float) and 0.0 <= p <= 1.0
    assert yhat in (0, 1)


def test_cnn_classifier_prediction_is_independent_of_temperature():
    """Same seed, same draw: temperature moves the probability, never the label.

    This is the accuracy-invariance invariant at the level of a single
    observation, and it is the property the whole study rests on.
    """
    import numpy as np

    from scoutfield.perception.adapter import CNNClassifier

    pool = _synthetic_pool()
    preds, probs = {}, {}
    for T in (0.3, 1.0, 4.0):
        clf = CNNClassifier(logit_pool=pool, temperature=T,
                            rng=np.random.default_rng(7))
        drawn = [clf.observe(i % 2) for i in range(500)]
        probs[T] = [p for p, _ in drawn]
        preds[T] = [yhat for _, yhat in drawn]

    assert preds[0.3] == preds[1.0] == preds[4.0], "hard prediction moved with temperature"
    assert probs[0.3] != probs[1.0], "temperature did not change the reported probability"


def test_cnn_classifier_accuracy_is_invariant_to_temperature():
    """Pool-level accuracy, asserted to four decimal places across the sweep."""
    from scoutfield.perception.adapter import CNNClassifier

    pool = _synthetic_pool()
    accs = {round(CNNClassifier(logit_pool=pool, temperature=T).accuracy, 4)
            for T in (0.3, 0.5, 1.0, 2.0, 3.0, 4.0)}
    assert len(accs) == 1, f"instrument broken: {accs}"


def test_cnn_classifier_refuses_to_invent_a_surrogate():
    """With no checkpoint and no pool it must fail loudly.

    A silent fallback to synthetic logits would make a fabricated result
    indistinguishable from a measured one.
    """
    from scoutfield.perception.adapter import CNNClassifier

    with pytest.raises(ValueError, match="logit_pool"):
        CNNClassifier(temperature=1.0)


def test_sweep_is_relative_to_the_fitted_temperature():
    """ECE bottoms out at sweep T = 1 and rises in both directions.

    The pilot's second invariant, carried onto a real classifier. Raw network
    logits are miscalibrated, so this only holds because the sweep is applied to
    logits already divided by the fitted temperature.
    """
    import numpy as np

    from scoutfield.perception.calibrate import fit_temperature, summarise

    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=20000)
    logits = rng.normal(np.where(labels == 1, 0.9, -0.9), 1.0)

    s = summarise(logits, labels, [0.3, 1.0, 4.0], fit_temperature(logits, labels))
    ece = {t: v["ece"] for t, v in s["sweep"].items()}
    assert ece["1"] < ece["0.3"], ece
    assert ece["1"] < ece["4"], ece
