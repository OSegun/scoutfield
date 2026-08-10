"""
Calibration: fitting the temperature, sweeping it, and the alternatives.

Run:
    python -m scoutfield.perception.calibrate --config configs/perception.yaml

Three things happen here, in order
----------------------------------
1. **Fit T on the validation split** by minimising NLL with the network frozen
   (Guo et al., 2017). This gives the model's *actual* calibration state and is
   the T = 1 reference point of the sweep.

2. **Sweep T** over the configured range to reproduce the pilot's instrument on a
   real classifier. This is the step that tests whether the pilot's effect was an
   artefact of the Gaussian surrogate. The invariant below must hold.

3. **MC-dropout and deep ensembles** as alternative uncertainty estimates
   (roadmap item 7), evaluated on the same axis so the three are comparable.

The invariant, restated because it is the whole experiment
----------------------------------------------------------
Temperature scaling divides logits by a scalar. The link is strictly monotone and
the decision boundary is at logit 0, so accuracy cannot move while confidence
does. Assert it:

    accs = {sweep_result[t]["accuracy"] for t in temperatures}
    assert len(accs) == 1, f"instrument broken: {accs}"

If this fails, the temperature is being applied after a softmax, or the hard
prediction is being recomputed from scaled values. It is not a tolerance issue —
the invariance is exact, to floating-point equality of the arg-max.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

# The pilot's ECE is the reference implementation. Importing it rather than
# reimplementing keeps this phase's numbers comparable with the published
# 0.0037 / 0.1980 — two binning conventions would silently disagree.
from perception import expected_calibration_error  # the PILOT's flat module

from scoutfield.perception.metrics import (
    brier_score,
    maximum_calibration_error,
    negative_log_likelihood,
    per_class_breakdown,
    reliability_data,
    signed_calibration_error,
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically clipped logistic link, matching the pilot's clipping range."""
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -60, 60)))


def collect_logits(model, loader, device=None, max_batches: int | None = None):
    """Run the model over a loader once and return ``(logits, binary, fine)``.

    One forward pass, cached afterwards. Everything calibration does acts on the
    logits alone, so caching them is exact rather than approximate.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    logits, binary, fine = [], [], []
    with torch.no_grad():
        for i, (x, y, y_fine) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            z = model(x.to(device, non_blocking=True))
            logits.append(z.detach().float().cpu().numpy().ravel())
            binary.append(np.asarray(y).ravel())
            fine.append(np.asarray(y_fine).ravel())

    return (
        np.concatenate(logits) if logits else np.empty(0),
        np.concatenate(binary).astype(int) if binary else np.empty(0, dtype=int),
        np.concatenate(fine).astype(int) if fine else np.empty(0, dtype=int),
    )


def evaluate_at_temperature(logits, labels, temperature: float, bins: int = 15) -> dict:
    """All calibration metrics at one temperature.

    The hard prediction comes from ``logit > 0``, never from the scaled
    probability, so accuracy is invariant to ``temperature`` by construction
    rather than by luck.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    probs = sigmoid(logits / float(temperature))
    preds = (logits > 0).astype(int)
    correct = (preds == labels).astype(float)
    conf = np.maximum(probs, 1.0 - probs)

    return {
        "temperature": float(temperature),
        "accuracy": float(correct.mean()),
        "ece": float(expected_calibration_error(conf, correct, n_bins=bins)),
        "signed_calibration_error": signed_calibration_error(probs, labels, bins),
        "maximum_calibration_error": maximum_calibration_error(probs, labels, bins),
        "nll": negative_log_likelihood(probs, labels),
        "brier": brier_score(probs, labels),
        "mean_confidence": float(conf.mean()),
        "n": int(labels.size),
    }


def fit_temperature(val_logits, val_labels) -> float:
    """Fit a single scalar T by minimising NLL on the validation split.

    Optimises log-T so T stays positive without a constraint (Guo et al., 2017).
    LBFGS over one parameter converges well within 50 iterations.
    """
    import torch

    z = torch.as_tensor(np.asarray(val_logits, dtype=np.float64), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels, dtype=np.float64), dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).item())


def sweep_temperature(logits, labels, temperatures: list[float], bins: int = 15) -> dict:
    """Evaluate accuracy, ECE, signed error, NLL and Brier at each temperature.

    Asserts accuracy invariance before returning. The invariance is exact — it
    follows from the arg-max being unmoved by a positive scalar divisor — so this
    is an equality check, not a tolerance check.
    """
    result = {float(t): evaluate_at_temperature(logits, labels, t, bins) for t in temperatures}

    accs = {r["accuracy"] for r in result.values()}
    if len(accs) != 1:
        raise AssertionError(
            f"instrument broken: accuracy varies with temperature: {sorted(accs)}. "
            "The temperature is being applied after the link function, or the hard "
            "prediction is being recomputed from scaled values."
        )
    return result


def mc_dropout_predict(model, loader, n_samples: int = 30, device=None):
    """Predictive mean and variance under MC-dropout.

    Reports both the mean probability (comparable to the temperature-scaled
    probability) and the variance, which temperature scaling cannot produce at
    all — that difference is the reason for comparing the methods.
    """
    from scoutfield.perception.model import enable_mc_dropout

    enable_mc_dropout(model)
    draws = []
    for _ in range(n_samples):
        z, labels, fine = collect_logits(model, loader, device=device)
        draws.append(sigmoid(z))
    probs = np.stack(draws)
    return {
        "mean": probs.mean(axis=0),
        "variance": probs.var(axis=0),
        "labels": labels,
        "fine_labels": fine,
        "n_samples": int(n_samples),
    }


def deep_ensemble_predict(checkpoints: list[str], loader, config=None, device=None):
    """Ensemble prediction over independently seeded fine-tuning runs.

    Members must differ in initialisation *and* data order. Ensembling
    checkpoints that differ only in data order understates a real ensemble's
    diversity, and therefore understates its calibration benefit.

    Five members is what a weekly Kaggle quota affords; that constraint is a
    reported limitation, not a silent choice.
    """
    from scoutfield.perception.model import build_model

    if config is None:
        raise ValueError("deep_ensemble_predict needs the config used to build the members")

    members = []
    for ckpt in checkpoints:
        model = build_model(config)
        state = _load_model_state(ckpt)
        model.load_state_dict(state)
        z, labels, fine = collect_logits(model, loader, device=device)
        members.append(sigmoid(z))

    probs = np.stack(members)
    return {
        "mean": probs.mean(axis=0),
        "variance": probs.var(axis=0),
        "labels": labels,
        "fine_labels": fine,
        "n_members": len(members),
    }


def _load_model_state(checkpoint):
    """Accept either a bare state_dict or a full training checkpoint."""
    from scoutfield.utils.checkpoint import load_training_state

    state = load_training_state(checkpoint)
    if state is None:
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return state.get("model", state)


def summarise(logits, labels, temperatures, fitted_t: float, bins: int = 15) -> dict:
    """Sweep plus the fitted-temperature reference point and per-class detail.

    The sweep is applied to *calibrated* logits — the raw logits divided by the
    fitted temperature — so that sweep T = 1 is the calibrated point, exactly as
    it was in the pilot. Sweeping the raw temperature instead would place the
    calibrated point somewhere other than 1 and make "T = 4" mean a different
    manipulation in each phase.
    """
    logits = np.asarray(logits, dtype=float)
    calibrated = logits / float(fitted_t)
    sweep = sweep_temperature(calibrated, labels, temperatures, bins)
    return {
        "fitted_temperature": float(fitted_t),
        "at_fitted_temperature": evaluate_at_temperature(calibrated, labels, 1.0, bins),
        "raw_at_unit_temperature": evaluate_at_temperature(logits, labels, 1.0, bins),
        "sweep": {f"{t:g}": v for t, v in sweep.items()},
        "sweep_is_relative_to_fitted_temperature": True,
        "per_class_at_fitted": per_class_breakdown(sigmoid(calibrated), labels, bins),
        "reliability_at_fitted": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in reliability_data(sigmoid(calibrated), labels, bins).items()
        },
    }


def run(config, checkpoint=None) -> dict:
    """Fit the temperature, sweep it on every split, and write the results file.

    The planner sweep reads its temperatures and measured ECEs from
    ``results/calibration_sweep.json`` rather than recomputing them, so the two
    phases cannot disagree about what T = 4 meant.
    """
    from scoutfield.perception.datasets import build_dataloaders
    from scoutfield.perception.model import build_model
    from scoutfield.utils.paths import find_checkpoint, results_dir

    cal_cfg = config.section("calibration")
    bins = int(cal_cfg.get("ece_bins", 15))
    temperatures = [float(t) for t in cal_cfg["temperature_sweep"]]

    # Not `checkpoints_dir(...)`: this notebook consumes what notebook 01 wrote, and
    # on Kaggle that arrives read-only under /kaggle/input, not in this session's
    # writable directory. See `find_checkpoint`.
    checkpoint = checkpoint or find_checkpoint("perception", "best.pt")
    model = build_model(config)
    model.load_state_dict(_load_model_state(checkpoint))

    loaders = build_dataloaders(config)
    val_logits, val_labels, _ = collect_logits(model, loaders["val"])
    fitted_t = fit_temperature(val_logits, val_labels)

    out = {
        "config": str(config.path),
        "checkpoint": str(checkpoint),
        "fitted_temperature": fitted_t,
        "splits": {"val": summarise(val_logits, val_labels, temperatures, fitted_t, bins)},
    }
    for split in ("test", "shift"):
        logits, labels, _ = collect_logits(model, loaders[split])
        out["splits"][split] = summarise(logits, labels, temperatures, fitted_t, bins)

    path = results_dir() / "calibration_sweep.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"fitted temperature: {fitted_t:.4f}")
    print(f"wrote {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/perception.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    from scoutfield.config import load_config
    from scoutfield.utils.seeding import seed_everything

    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    run(config, args.checkpoint)


if __name__ == "__main__":
    main()
