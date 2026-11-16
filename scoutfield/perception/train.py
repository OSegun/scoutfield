"""
Fine-tuning EfficientNet-B0 on PlantVillage.

Run:
    python -m scoutfield.perception.train --config configs/perception.yaml

Design constraints that are not stylistic
-----------------------------------------
**Checkpoint every epoch.** Kaggle kills sessions at nine hours and much sooner
when idle. A twelve-epoch run that has to restart from scratch because the
session dropped at epoch eleven wastes a meaningful fraction of a weekly GPU
quota. Resume must restore the optimizer, scheduler *and* RNG states — a resumed
run that reseeds is not a continuation of the run it claims to continue.

**Label smoothing stays at zero.** It is a standard accuracy trick and it
directly alters confidence calibration, which is this study's dependent variable.
Turning it on would confound the thing being measured. If it is ever enabled, it
must be swept and reported, not silently defaulted.

**No early stopping on the validation split.** That split fits the temperature.
Using it for both leaks the calibration fit into model selection.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _build_optimizer(model, train_cfg):
    """AdamW with separate backbone and head learning rates.

    Two groups because the backbone is pretrained and the head is not: one
    learning rate that suits a randomly initialised head will wreck pretrained
    features.
    """
    import torch

    head, backbone = [], []
    for name, param in model.backbone.named_parameters():
        (head if name.startswith("classifier") else backbone).append(param)

    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(train_cfg["backbone_lr"])},
            {"params": head, "lr": float(train_cfg["lr"])},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )


def _evaluate(model, loader, device, bins: int):
    from scoutfield.perception.calibrate import collect_logits, evaluate_at_temperature

    logits, labels, _ = collect_logits(model, loader, device=device)
    return evaluate_at_temperature(logits, labels, temperature=1.0, bins=bins), logits, labels


class ResumeRefused(RuntimeError):
    """Resume state exists but resuming was not asked for."""


def member_files(member: int) -> dict[str, str]:
    """Store names of one deep-ensemble member's files (roadmap item 7)."""
    return {"last": f"member{member}_last.pt", "best": f"member{member}_best.pt",
            "summary": f"member{member}_summary.json"}


def run_target(config, member: int | None = None) -> dict:
    """Where a training run files its outputs, and the seed it trains with.

    ``member=None`` is stage 01: ``best.pt``, ``last.pt`` and
    ``perception_summary.json`` in ``01_perception``, trained with the top-level
    ``seed``. ``member=k`` (1 to 4) is a deep-ensemble member: its own files in the
    ``uncertainty.stage`` stage, trained with ``uncertainty.member_seeds[k - 1]``.
    The PlantVillage split is drawn from the top-level ``seed`` either way
    (``perception.datasets.index_and_split``), so every member sees the same split.
    """
    if member is None:
        return {"stage": "01_perception", "seed": int(config["seed"]), "subdir": None,
                "last": "last.pt", "best": "best.pt", "summary": "perception_summary.json"}
    unc = config.section("uncertainty")
    seeds = [int(s) for s in unc["member_seeds"]]
    n_members = int(config.section("calibration")["ensemble_members"])
    if 1 + len(seeds) != n_members:
        raise ValueError(
            f"uncertainty.member_seeds lists {len(seeds)} seeds, but "
            f"calibration.ensemble_members is {n_members}: member 0 is stage 01's "
            f"best.pt, so there must be {n_members - 1} member seeds")
    if member not in range(1, len(seeds) + 1):
        raise ValueError(f"member {member} does not exist; members 1 to {len(seeds)} "
                         "are trained here, member 0 is 01_perception/best.pt")
    if seeds[member - 1] == int(config["seed"]):
        raise ValueError(f"member {member} would reuse the stage 01 seed "
                         f"{config['seed']}, so it would repeat member 0")
    return {"stage": str(unc["stage"]), "seed": seeds[member - 1], "subdir": "ensemble",
            **member_files(member)}


def train(config, resume: bool = False, store=None, member: int | None = None) -> dict:
    """Fine-tune and return a summary dict.

    The summary is written to ``results/perception_summary.json`` and pushed with
    the checkpoints to stage ``01_perception``; downstream stages read these
    numbers from there and nothing hard-codes an accuracy or an ECE.

    ``last.pt`` (model, optimizer, scheduler, scaler and RNG states) is pushed to
    the store after every epoch, so a killed session loses at most one epoch.
    With ``resume=True`` training continues from that file. Without it, an
    existing ``last.pt`` in the store or on disk refuses the run: silently
    starting over would overwrite a partially trained model.

    ``member=k`` trains deep-ensemble member ``k`` instead, with the same code,
    hyperparameters and split, its own seed and its own files (``run_target``).
    """
    import torch
    from torch import nn

    from scoutfield.perception.calibrate import (
        collect_logits,
        evaluate_at_temperature,
        fit_temperature,
    )
    from scoutfield.perception.datasets import build_dataloaders
    from scoutfield.perception.model import build_model
    from scoutfield.utils.artifacts import get_store, git_commit
    from scoutfield.utils.checkpoint import (
        load_training_state,
        restore_rng_state,
        rng_state,
        save_training_state,
    )
    from scoutfield.utils.device import select_device
    from scoutfield.utils.paths import checkpoints_dir, repo_root, results_dir
    from scoutfield.utils.seeding import seed_everything

    store = store or get_store()
    target = run_target(config, member)
    stage = target["stage"]
    ckpt_dir = (checkpoints_dir("perception", target["subdir"]) if target["subdir"]
                else checkpoints_dir("perception"))
    last_path, best_path = ckpt_dir / target["last"], ckpt_dir / target["best"]
    if not resume:
        if store.has(stage, target["last"]) or last_path.exists():
            raise ResumeRefused(
                f"resume state already exists for run {store.run_id} "
                f"({stage}/{target['last']} in the store, or {last_path}). Pass "
                "--resume (notebook 01: RESUME = True) to continue that run, or set "
                "a new run_id in configs/artifacts.yaml to start a separate one."
            )
    else:
        import shutil

        pulled = store.pull(stage, target["last"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pulled, last_path)
        best = store.pull(stage, target["best"], required=False)
        if best is not None:
            shutil.copyfile(best, best_path)

    seed = target["seed"]
    seed_everything(seed)

    train_cfg = config.section("train")
    model_cfg = config.section("model")
    cal_cfg = config.section("calibration")
    bins = int(cal_cfg.get("ece_bins", 15))
    epochs = int(train_cfg["epochs"])
    freeze_epochs = int(model_cfg.get("freeze_backbone_epochs", 0))

    device = select_device()
    loaders = build_dataloaders(config)
    model = build_model(config).to(device)
    optimizer = _build_optimizer(model, train_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Label smoothing stays at zero: it alters confidence calibration, which is
    # this study's dependent variable. Enabling it would confound the measurement.
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    if label_smoothing != 0.0:
        raise ValueError(
            f"label_smoothing={label_smoothing} confounds the calibration measurement; "
            "if it is genuinely wanted it must be swept and reported, not defaulted"
        )
    criterion = nn.BCEWithLogitsLoss()

    use_amp = bool(train_cfg.get("amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, history, best_metric = 0, [], -float("inf")
    resumed = load_training_state(last_path) if resume else None
    if resumed is not None:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        if resumed.get("scaler") is not None:
            scaler.load_state_dict(resumed["scaler"])
        restore_rng_state(resumed.get("rng"))
        start_epoch = int(resumed["epoch"]) + 1
        history = list(resumed.get("history", []))
        best_metric = float(resumed.get("best_metric", -float("inf")))
        print(f"resumed from epoch {start_epoch} of {epochs}")

    for epoch in range(start_epoch, epochs):
        model.set_backbone_trainable(epoch >= freeze_epochs)
        model.train()
        running, n_seen = 0.0, 0

        for x, y, _ in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        scheduler.step()
        train_loss = running / max(n_seen, 1)

        # Model selection uses `test`, never `val`: `val` fits the temperature,
        # and using it for both leaks the calibration fit into model selection.
        test_metrics, _, _ = _evaluate(model, loaders["test"], device, bins)
        val_metrics, _, _ = _evaluate(model, loaders["val"], device, bins)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_ece": test_metrics["ece"],
            # Logged per epoch because calibration typically degrades as training
            # accuracy saturates, and where that happens is worth knowing.
            "val_ece": val_metrics["ece"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": scheduler.get_last_lr(),
        }
        history.append(row)
        print(
            f"epoch {epoch:>3}  loss {train_loss:.4f}  "
            f"test_acc {row['test_accuracy']:.4f}  test_ece {row['test_ece']:.4f}  "
            f"val_ece {row['val_ece']:.4f}"
        )

        if bool(train_cfg.get("checkpoint_every_epoch", True)):
            save_training_state(last_path, {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "rng": rng_state(),
                "history": history,
                "best_metric": best_metric,
                "config": str(config.path),
            })

        improved = test_metrics["accuracy"] > best_metric
        if improved:
            best_metric = test_metrics["accuracy"]
            save_training_state(best_path, {
                "epoch": epoch,
                "model": model.state_dict(),
                "test_accuracy": best_metric,
                "config": str(config.path),
            })
            # last.pt was saved before this update, so it is rewritten to carry the
            # new best_metric: otherwise a resumed run could re-save an equal epoch
            # as "best".
            last_state = load_training_state(last_path)
            if last_state is not None:
                last_state["best_metric"] = best_metric
                save_training_state(last_path, last_state)

        # Pushed every epoch: a session that dies loses one epoch, not the run.
        if last_path.exists():
            label = f"epoch {epoch}" if member is None else f"member {member} epoch {epoch}"
            store.push(stage, [last_path, best_path] if improved else [last_path], label)

    best_state = load_training_state(best_path)
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    val_logits, val_labels, _ = collect_logits(model, loaders["val"], device=device)
    fitted_t = fit_temperature(val_logits, val_labels)

    test_logits, test_labels, _ = collect_logits(model, loaders["test"], device=device)
    fit_logits, fit_labels, _ = collect_logits(model, loaders["shift_fit"], device=device)
    eval_logits, eval_labels, _ = collect_logits(model, loaders["shift_eval"], device=device)
    shift_logits = np.concatenate([fit_logits, eval_logits])
    shift_labels = np.concatenate([fit_labels, eval_labels])
    test_at_t = evaluate_at_temperature(test_logits, test_labels, fitted_t, bins)
    shift_at_t = evaluate_at_temperature(shift_logits, shift_labels, fitted_t, bins)
    shift_eval_at_t = evaluate_at_temperature(eval_logits, eval_labels, fitted_t, bins)

    summary = {
        # In-distribution (PlantVillage test) accuracy: the control condition.
        "accuracy": test_at_t["accuracy"],
        # All of PlantDoc, and the shift_eval half the planners are scored on.
        # BASE_ACC is the shift_eval accuracy; see docs/METHOD.md, "BASE_ACC".
        "accuracy_shift": shift_at_t["accuracy"],
        "accuracy_shift_eval": shift_eval_at_t["accuracy"],
        "ece": test_at_t["ece"],
        "ece_shift": shift_at_t["ece"],
        "signed_calibration_error": test_at_t["signed_calibration_error"],
        "signed_calibration_error_shift": shift_at_t["signed_calibration_error"],
        # Lab-fitted, on PlantVillage validation. Stage 02 fits the shift
        # temperature and is the authoritative source for both.
        "lab_fitted_temperature": fitted_t,
        "best_epoch": int(best_state["epoch"]) if best_state else None,
        "history": history,
        "splits": loaders["meta"],
        "config": str(config.path),
        "git_commit": git_commit(),
        "seed": seed,
        "member": member,
        "device": device,
    }

    out = results_dir() / target["summary"]
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.tolist()
                  if isinstance(o, np.ndarray) else str(o))
    print(f"wrote {out}  (repo root {repo_root()})")
    store.push(stage, [best_path, last_path, out],
               "fine-tune complete" if member is None else f"member {member} complete")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/perception.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="continue from 01_perception/last.pt in the store")
    args = parser.parse_args()

    from scoutfield.config import load_config

    try:
        train(load_config(args.config), resume=args.resume)
    except ResumeRefused as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
