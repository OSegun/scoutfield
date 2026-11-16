"""
Roadmap item 7: three uncertainty methods compared on one axis.

Run:
    python experiments/07_uncertainty.py train --member 1     # ... to 4, GPU
    python experiments/07_uncertainty.py evaluate             # GPU

The three methods
-----------------
* **Temperature scaling** (Guo et al., 2017): the stage 01 classifier, its logit
  divided by one fitted scalar. It changes confidence and never the prediction.
* **MC-dropout** (Gal and Ghahramani, 2016): the same classifier with its head's
  dropout left on at prediction time; the probability is the mean over
  ``calibration.mc_dropout_samples`` draws, and their spread is a second output.
* **Deep ensemble** (Lakshminarayanan et al., 2017): five classifiers fine-tuned
  with the same recipe and split, differing in seed; the probability is the mean of
  their probabilities, and their disagreement is a second output.

Each is scored raw and after its own temperature is fitted by the stage 02 NLL fit,
on the split ``uncertainty.fit_splits`` pairs with the scored split. Temperature
scaling raw is the uncalibrated classifier; fitted, it is the calibration point the
planner sweep used, and that is checked against stage 02 before anything else.

What this comparison is not
---------------------------
* Not at fixed accuracy. Averaging draws or members can move predictions near the
  boundary, so accuracy is reported per method; the temperature sweep's invariance
  holds within a method, not across methods.
* Not a planner comparison. The planners were flown on temperature-scaled logits
  only; whether MC-dropout or an ensemble would change where a drone flies is not
  measured.
"""

from __future__ import annotations

import json

import numpy as np

from scoutfield.perception.calibrate import (
    evaluate_at_temperature,
    fit_temperature,
    sigmoid,
)
from scoutfield.perception.metrics import reliability_data

#: Method names as the results file and Figure 16 spell them.
METHODS = ("temperature_scaling", "mc_dropout", "deep_ensemble")
LABELS = {"temperature_scaling": "Temperature scaling", "mc_dropout": "MC-dropout",
          "deep_ensemble": "Deep ensemble"}


def probability_logit(p) -> np.ndarray:
    """Log-odds of a probability, clipped to ±60, the range ``sigmoid`` clips to.

    An averaged probability can be exactly 0 or 1 in floating point, whose log-odds
    are infinite. Clipping the log-odds (not the probability: 1 - sigmoid(-60)
    rounds to exactly 1) keeps them finite with the same bound every other
    probability in this project is computed with.
    """
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        z = np.log(p) - np.log1p(-p)
    return np.clip(z, -60.0, 60.0)


def _reliability(probs, labels, bins: int) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in reliability_data(probs, labels, bins).items()}


def spread_summary(variance, probs, labels) -> dict:
    """Mean predictive variance overall, on correct and on wrong predictions.

    Temperature scaling has no such output. For MC-dropout and the ensemble, a
    variance that is larger on wrong predictions than on right ones is evidence it
    carries information the probability alone does not.
    """
    variance = np.asarray(variance, dtype=float)
    correct = (np.asarray(probs) > 0.5).astype(int) == np.asarray(labels, dtype=int)
    def mean(mask):
        return float(variance[mask].mean()) if mask.any() else None
    return {"mean_variance": float(variance.mean()),
            "mean_variance_correct": mean(correct),
            "mean_variance_wrong": mean(~correct),
            "n_correct": int(correct.sum()), "n_wrong": int((~correct).sum())}


def score_method(probs: dict, labels: dict, fit_splits: dict, bins: int,
                 variance: dict | None = None) -> dict:
    """One method's rows: for each scored split, raw and with its fitted temperature.

    ``probs`` and ``labels`` map split names to arrays; ``fit_splits`` maps each
    scored split to the split its temperature is fitted on.
    """
    out = {}
    for split, fit_split in fit_splits.items():
        z = probability_logit(probs[split])
        fitted_t = fit_temperature(probability_logit(probs[fit_split]), labels[fit_split])
        raw = evaluate_at_temperature(z, labels[split], 1.0, bins)
        fitted = evaluate_at_temperature(z, labels[split], fitted_t, bins)
        if raw["accuracy"] != fitted["accuracy"]:
            raise AssertionError(f"{split}: accuracy moved with the fitted temperature")
        entry = {
            "fit_split": fit_split,
            "fitted_temperature": float(fitted_t),
            "raw": raw,
            "fitted": fitted,
            "reliability_raw": _reliability(sigmoid(z), labels[split], bins),
            "reliability_fitted": _reliability(sigmoid(z / fitted_t), labels[split], bins),
        }
        if variance is not None:
            entry["spread"] = spread_summary(variance[split], probs[split], labels[split])
        out[split] = entry
    return out


def compare(member_logits: dict, mc_logits: dict, labels: dict, fit_splits: dict,
            bins: int) -> dict:
    """Score the three methods.

    ``member_logits[split]`` has shape ``(members, images)``, member 0 first (stage
    01's ``best.pt``); ``mc_logits[split]`` has shape ``(draws, images)``, drawn on
    member 0. Every member and every draw must cover the same images in the same
    order, which holds because they read the same unshuffled evaluation loaders.
    """
    splits = sorted(set(fit_splits) | set(fit_splits.values()))
    for split in splits:
        n = len(labels[split])
        if member_logits[split].shape[1] != n or mc_logits[split].shape[1] != n:
            raise ValueError(f"{split}: logits do not cover the {n} labelled images")

    ts = {s: sigmoid(member_logits[s][0]) for s in splits}
    mc_draws = {s: sigmoid(mc_logits[s]) for s in splits}
    ens_members = {s: sigmoid(member_logits[s]) for s in splits}
    return {
        "temperature_scaling": score_method(ts, labels, fit_splits, bins),
        "mc_dropout": score_method({s: d.mean(axis=0) for s, d in mc_draws.items()},
                                   labels, fit_splits, bins,
                                   variance={s: d.var(axis=0) for s, d in mc_draws.items()}),
        "deep_ensemble": score_method({s: m.mean(axis=0) for s, m in ens_members.items()},
                                      labels, fit_splits, bins,
                                      variance={s: m.var(axis=0)
                                                for s, m in ens_members.items()}),
    }


def check_consistency(methods: dict, calibration: dict, atol: float) -> dict:
    """Temperature scaling here must reproduce stage 02 on ``shift_eval``.

    Member 0 is stage 01's ``best.pt``, the classifier stage 02 calibrated. If the
    shift-fitted temperature or the calibrated ``shift_eval`` ECE recomputed here
    differ from stage 02's by more than ``atol``, this comparison is not looking at
    the classifier the rest of the run used, and it stops.
    """
    ts = methods["temperature_scaling"]["shift_eval"]
    stage02 = calibration["sweeps"]["shift_eval"]
    checks = {
        "shift_fitted_temperature": (ts["fitted_temperature"],
                                     float(calibration["shift_fitted_temperature"])),
        "shift_eval_ece_at_fitted": (ts["fitted"]["ece"],
                                     float(stage02["at_fitted_temperature"]["ece"])),
        "shift_eval_accuracy": (ts["fitted"]["accuracy"],
                                float(stage02["at_fitted_temperature"]["accuracy"])),
    }
    out = {k: {"here": a, "stage02": b, "abs_difference": abs(a - b)}
           for k, (a, b) in checks.items()}
    bad = {k: v for k, v in out.items() if v["abs_difference"] > atol}
    if bad:
        raise AssertionError(
            "temperature scaling here does not reproduce stage 02: "
            + ", ".join(f"{k} {v['here']:.6f} vs {v['stage02']:.6f}" for k, v in bad.items())
            + f" (tolerance {atol:g}). Member 0 is not the classifier stage 02 used.")
    return {"atol": float(atol), "checks": out}


def run(config, store=None) -> dict:
    """Evaluate the three methods and push the comparison to the uncertainty stage."""
    from scoutfield.perception.calibrate import (
        _load_model_state,
        collect_features,
        collect_logits,
        mc_dropout_logits,
    )
    from scoutfield.perception.datasets import build_dataloaders
    from scoutfield.perception.model import build_model
    from scoutfield.perception.train import member_files
    from scoutfield.utils.artifacts import get_store, git_commit, sha256_of
    from scoutfield.utils.paths import results_dir

    store = store or get_store()
    unc = config.section("uncertainty")
    cal = config.section("calibration")
    stage = str(unc["stage"])
    bins = int(cal.get("ece_bins", 15))
    fit_splits = {str(k): str(v) for k, v in dict(unc["fit_splits"]).items()}
    n_members = int(cal["ensemble_members"])
    if 1 + len(unc["member_seeds"]) != n_members:
        raise ValueError("uncertainty.member_seeds must list ensemble_members - 1 seeds")

    checkpoints = [("01_perception", "best.pt")] + [
        (stage, member_files(k)["best"]) for k in range(1, n_members)]
    missing = [f"{s}/{n}" for s, n in checkpoints if not store.has(s, n)]
    if missing:
        raise FileNotFoundError(
            f"run {store.run_id} in {store.backend.description} lacks {missing}. Train "
            "each missing member first: `python experiments/07_uncertainty.py train "
            "--member <k>`.")

    loaders = build_dataloaders(config)
    splits = sorted(set(fit_splits) | set(fit_splits.values()))
    member_logits = {s: [] for s in splits}
    labels, files, mc_logits = {}, [], {}
    for index, (ckpt_stage, name) in enumerate(checkpoints):
        path = store.pull(ckpt_stage, name)
        files.append({"member": index, "stage": ckpt_stage, "name": name,
                      "sha256": sha256_of(path)})
        model = build_model(config)
        model.load_state_dict(_load_model_state(path))
        for split in splits:
            z, y, _ = collect_logits(model, loaders[split])
            member_logits[split].append(z)
            if split in labels and not np.array_equal(labels[split], y):
                raise AssertionError(f"{split}: member {index} saw the images in another order")
            labels[split] = y
        if index == 0:
            for split in splits:
                feats, y, _ = collect_features(model, loaders[split])
                if not np.array_equal(labels[split], y):
                    raise AssertionError(f"{split}: features and logits disagree on order")
                mc_logits[split] = mc_dropout_logits(
                    model, feats, int(cal["mc_dropout_samples"]), int(unc["mc_dropout_seed"]))
        del model
    member_logits = {s: np.stack(v) for s, v in member_logits.items()}

    methods = compare(member_logits, mc_logits, labels, fit_splits, bins)
    calibration = store.pull_json("02_calibration", "calibration_sweep.json")
    consistency = check_consistency(methods, calibration, float(unc["consistency_atol"]))
    membership = loaders["meta"]["membership"]

    out = {
        "run_id": store.run_id,
        "methods": methods,
        "consistency_with_stage02": consistency,
        "settings": {"mc_dropout_samples": int(cal["mc_dropout_samples"]),
                     "mc_dropout_seed": int(unc["mc_dropout_seed"]),
                     "ensemble_members": n_members,
                     "member_seeds": [int(config["seed"])] + [int(s) for s in
                                                              unc["member_seeds"]],
                     "fit_splits": fit_splits, "ece_bins": bins},
        "members": files,
        "split_sizes": {s: len(labels[s]) for s in splits},
        "membership": {s: membership[s] for s in splits},
        "config": str(config.path),
        "git_commit": git_commit(),
    }
    path = results_dir() / str(unc["results_file"])
    with path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    for name in METHODS:
        row = methods[name]["shift_eval"]
        print(f"{LABELS[name]:20s} shift_eval accuracy {row['raw']['accuracy']:.4f}  "
              f"ECE raw {row['raw']['ece']:.4f}  fitted {row['fitted']['ece']:.4f}")
    store.push(stage, [path], "uncertainty-method comparison")
    store.record_inputs(stage)
    print(f"pushed {path.name} to {stage} of run {store.run_id}")
    return out
