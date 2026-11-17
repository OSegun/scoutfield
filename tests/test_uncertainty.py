"""
Roadmap item 7: the uncertainty-method comparison.

The comparison is only as good as its two least obvious parts: that MC-dropout
draws really differ (with dropout switched off, all thirty would be one
prediction), and that every ensemble member trains on the
stage 01 split with its own seed. These tests pin both, and the scoring, with an
untrained network and synthetic logits: no datasets, no downloaded weights, no GPU.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
FIT_SPLITS = {"shift_eval": "shift_fit", "test": "val"}


def _model(seed=0):
    from scoutfield.perception.model import BinaryClassifier

    torch.manual_seed(seed)
    return BinaryClassifier(pretrained=False, head_dropout=0.3).eval()


def _loader(n=4, size=64, seed=0):
    """A loader shaped like the real ones: (image, binary label, fine label)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, size, size, generator=g)
    y = torch.arange(n) % 2
    data = [(x[i], y[i], y[i]) for i in range(n)]
    return torch.utils.data.DataLoader(data, batch_size=2, shuffle=False)


def _config(**changes):
    from scoutfield.config import Config, load_config

    base = load_config(ROOT / "configs" / "perception.yaml")
    data = copy.deepcopy(base.data)
    for dotted, value in changes.items():
        section, key = dotted.split(".") if "." in dotted else (None, dotted)
        (data[section] if section else data)[key] = value
    return Config(data=data, path=base.path)


# ------------------------------------------------------------------ MC-dropout

def test_mc_dropout_draws_differ_and_reproduce():
    from scoutfield.perception.calibrate import mc_dropout_logits

    model = _model()
    features = torch.randn(6, 1280, generator=torch.Generator().manual_seed(1))
    draws = mc_dropout_logits(model, features, n_samples=30, seed=0)
    assert draws.shape == (30, 6)
    assert np.all(draws.std(axis=0) > 0), "every draw was the same: dropout is off"
    assert np.array_equal(draws, mc_dropout_logits(model, features, 30, seed=0))
    assert not np.array_equal(draws, mc_dropout_logits(model, features, 30, seed=1))
    # The model itself is left untouched: still in eval mode, dropout included.
    assert not model.training
    assert not any(m.training for m in model.modules())


def test_feature_path_reproduces_the_full_forward_pass():
    """Head on cached features = the model's own logit; MC draws average to it."""
    from scoutfield.perception.calibrate import (
        collect_features,
        collect_logits,
        mc_dropout_logits,
    )

    model = _model()
    loader = _loader()
    logits, labels, _ = collect_logits(model, loader, device="cpu")
    features, labels_f, _ = collect_features(model, loader, device="cpu")
    assert np.array_equal(labels, labels_f)
    with torch.no_grad():
        head = model.backbone.classifier.eval()
        direct = head(features).squeeze(-1).numpy()
    np.testing.assert_allclose(direct, logits, rtol=1e-5, atol=1e-5)

    # Inverted dropout keeps the expectation: the mean of many draws is the logit.
    draws = mc_dropout_logits(model, features, n_samples=4000, seed=0)
    np.testing.assert_allclose(draws.mean(axis=0), logits, atol=0.05 * np.abs(logits).max()
                               + 0.02)


def test_mc_dropout_refuses_a_head_without_dropout():
    from scoutfield.perception.calibrate import mc_dropout_logits
    from scoutfield.perception.model import BinaryClassifier

    model = BinaryClassifier(pretrained=False, head_dropout=0.0)
    with pytest.raises(RuntimeError, match="dropout"):
        mc_dropout_logits(model, torch.zeros(2, 1280), n_samples=3)


# ------------------------------------------------------------ ensemble members

def test_each_member_has_its_own_seed_and_files():
    from scoutfield.perception.train import member_files, run_target

    cfg = _config()
    stage01 = run_target(cfg, None)
    assert (stage01["stage"], stage01["best"], stage01["seed"]) == (
        "01_perception", "best.pt", int(cfg["seed"]))
    seeds = cfg.section("uncertainty")["member_seeds"]
    for k in range(1, len(seeds) + 1):
        target = run_target(cfg, k)
        assert target["stage"] == cfg.section("uncertainty")["stage"]
        assert target["seed"] == seeds[k - 1]
        assert target["best"] == member_files(k)["best"] == f"member{k}_best.pt"
    with pytest.raises(ValueError, match="member 0 is 01_perception"):
        run_target(cfg, 0)
    with pytest.raises(ValueError, match="does not exist"):
        run_target(cfg, len(seeds) + 1)


def test_member_settings_are_checked():
    from scoutfield.perception.train import run_target

    with pytest.raises(ValueError, match="ensemble_members"):
        run_target(_config(**{"uncertainty.member_seeds": [1, 2]}), 1)
    with pytest.raises(ValueError, match="repeat member 0"):
        run_target(_config(**{"uncertainty.member_seeds": [0, 2, 3, 4]}), 1)


def test_the_split_ignores_the_member_seed():
    """The PlantVillage split reads the top-level seed, never a member seed.

    ``index_and_split`` draws the split from ``config["seed"]``; ``train`` puts the
    member seed only into ``seed_everything``. If the split read the global RNG, a
    member's test images could be another member's training images.
    """
    import inspect

    from scoutfield.perception import datasets, train

    split_src = inspect.getsource(datasets.index_and_split)
    assert 'seed = int(config["seed"])' in split_src
    assert "np.random.default_rng" in inspect.getsource(datasets._split_by_source)
    train_src = inspect.getsource(train.train)
    assert 'seed = target["seed"]' in train_src
    assert "build_dataloaders(config)" in train_src


def _driver():
    spec = importlib.util.spec_from_file_location(
        "uncertainty_driver", ROOT / "experiments" / "07_uncertainty.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_member_training_skips_finished_and_resumes_unfinished(store, tmp_path,
                                                                monkeypatch):
    from scoutfield.perception import train

    calls = []
    monkeypatch.setattr(train, "train",
                        lambda config, resume, store, member: calls.append((member, resume)))
    driver = _driver()
    cfg = _config()
    stage = cfg.section("uncertainty")["stage"]

    assert driver.train_member(cfg, 1, store=store) == "trained"
    last = tmp_path / "member2_last.pt"
    last.write_bytes(b"state")
    store.push(stage, [last], "member 2 epoch 3")
    assert driver.train_member(cfg, 2, store=store) == "resumed"
    summary = tmp_path / "member3_summary.json"
    summary.write_text("{}", encoding="utf-8")
    store.push(stage, [summary], "member 3 complete")
    assert driver.train_member(cfg, 3, store=store) == "skipped"
    assert calls == [(1, False), (2, True)]


# --------------------------------------------------------------------- scoring

def _synthetic(seed=0, n=400, members=5, draws=30):
    rng = np.random.default_rng(seed)
    labels, member_logits, mc_logits = {}, {}, {}
    for split in ("shift_eval", "shift_fit", "test", "val"):
        y = rng.integers(0, 2, n)
        signal = (2 * y - 1) * 2.0 + rng.normal(0, 2.0, n)
        labels[split] = y
        member_logits[split] = signal + rng.normal(0, 0.5, (members, n))
        mc_logits[split] = signal + rng.normal(0, 0.8, (draws, n))
    return member_logits, mc_logits, labels


def test_three_methods_are_scored_raw_and_fitted():
    from scoutfield.perception.calibrate import sigmoid
    from scoutfield.perception.uncertainty import METHODS, compare

    member_logits, mc_logits, labels = _synthetic()
    out = compare(member_logits, mc_logits, labels, FIT_SPLITS, bins=15)
    assert set(out) == set(METHODS)
    for name in METHODS:
        for split, fit in FIT_SPLITS.items():
            row = out[name][split]
            assert row["fit_split"] == fit
            assert row["raw"]["accuracy"] == row["fitted"]["accuracy"]
            assert row["fitted_temperature"] > 0
            assert sum(row["reliability_fitted"]["count"]) == len(labels[split])
    # The temperature is fitted on the fitting split, never on the scored one.
    from scoutfield.perception.calibrate import fit_temperature

    ts_fit = fit_temperature(member_logits["shift_fit"][0], labels["shift_fit"])
    assert out["temperature_scaling"]["shift_eval"]["fitted_temperature"] == \
        pytest.approx(ts_fit, rel=1e-6)
    # Temperature scaling is member 0 alone, and has no spread.
    ts = out["temperature_scaling"]["shift_eval"]
    expected = ((member_logits["shift_eval"][0] > 0).astype(int)
                == labels["shift_eval"]).mean()
    assert ts["raw"]["accuracy"] == pytest.approx(expected)
    assert "spread" not in ts
    # The ensemble's probability is the mean of its members' probabilities.
    p = sigmoid(member_logits["shift_eval"]).mean(axis=0)
    ens = out["deep_ensemble"]["shift_eval"]
    assert ens["raw"]["accuracy"] == pytest.approx(
        ((p > 0.5).astype(int) == labels["shift_eval"]).mean())
    assert ens["spread"]["mean_variance"] == pytest.approx(
        sigmoid(member_logits["shift_eval"]).var(axis=0).mean())
    assert out["mc_dropout"]["shift_eval"]["spread"]["n_correct"] + \
        out["mc_dropout"]["shift_eval"]["spread"]["n_wrong"] == len(labels["shift_eval"])


def test_identical_members_have_no_spread():
    from scoutfield.perception.uncertainty import compare

    member_logits, mc_logits, labels = _synthetic()
    same = {s: np.repeat(v[:1], 5, axis=0) for s, v in member_logits.items()}
    out = compare(same, mc_logits, labels, FIT_SPLITS, bins=15)
    assert out["deep_ensemble"]["shift_eval"]["spread"]["mean_variance"] == \
        pytest.approx(0.0, abs=1e-20)
    assert out["deep_ensemble"]["shift_eval"]["raw"]["ece"] == pytest.approx(
        out["temperature_scaling"]["shift_eval"]["raw"]["ece"])


def test_logits_must_cover_the_labelled_images():
    from scoutfield.perception.uncertainty import compare

    member_logits, mc_logits, labels = _synthetic()
    member_logits["shift_eval"] = member_logits["shift_eval"][:, :-1]
    with pytest.raises(ValueError, match="shift_eval"):
        compare(member_logits, mc_logits, labels, FIT_SPLITS, bins=15)


def test_probability_logit_is_finite_at_zero_and_one():
    from scoutfield.perception.uncertainty import probability_logit

    z = probability_logit([0.0, 0.5, 1.0])
    assert np.all(np.isfinite(z)) and z[1] == 0.0 and z[0] == -z[2]


def test_temperature_scaling_must_reproduce_stage02():
    from scoutfield.perception.uncertainty import check_consistency, compare

    member_logits, mc_logits, labels = _synthetic()
    methods = compare(member_logits, mc_logits, labels, FIT_SPLITS, bins=15)
    ts = methods["temperature_scaling"]["shift_eval"]
    calibration = {"shift_fitted_temperature": ts["fitted_temperature"],
                   "sweeps": {"shift_eval": {"at_fitted_temperature": {
                       "ece": ts["fitted"]["ece"], "accuracy": ts["fitted"]["accuracy"]}}}}
    ok = check_consistency(methods, calibration, atol=1e-3)
    assert all(v["abs_difference"] == 0 for v in ok["checks"].values())

    calibration["shift_fitted_temperature"] += 0.01
    with pytest.raises(AssertionError, match="does not reproduce stage 02"):
        check_consistency(methods, calibration, atol=1e-3)


def _fake_pipeline(store, tmp_path, monkeypatch, members):
    """Five tiny checkpoints in the store and loaders that need no dataset."""
    from scoutfield.perception import datasets, model
    from scoutfield.utils import paths
    from scoutfield.utils.checkpoint import save_training_state

    for k in range(members):
        path = tmp_path / ("best.pt" if k == 0 else f"member{k}_best.pt")
        save_training_state(path, {"model": _model(seed=k).state_dict(), "epoch": 0})
        store.push("01_perception" if k == 0 else "07_uncertainty", [path], f"m{k}")
    splits = ("shift_eval", "shift_fit", "test", "val")
    loaders = {s: _loader(n=6, seed=i) for i, s in enumerate(splits)}
    loaders["meta"] = {"membership": {s: f"hash-{s}" for s in splits}}
    monkeypatch.setattr(datasets, "build_dataloaders", lambda config: loaders)
    monkeypatch.setattr(model, "build_model",
                        lambda config: model.BinaryClassifier(pretrained=False))
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path)


def test_evaluate_runs_end_to_end_and_files_its_result(store, tmp_path, monkeypatch):
    import json

    from scoutfield.perception import uncertainty

    _fake_pipeline(store, tmp_path, monkeypatch, members=5)
    calibration = tmp_path / "calibration_sweep.json"
    calibration.write_text(json.dumps({"shift_fitted_temperature": 1.0, "sweeps": {
        "shift_eval": {"at_fitted_temperature": {"ece": 0.0, "accuracy": 0.5}}}}),
        encoding="utf-8")
    store.push("02_calibration", [calibration], "calibration")
    # The tiny models are untrained, so stage 02's numbers cannot match; the check
    # itself is tested above. Here only the wiring is under test.
    cfg = _config(**{"uncertainty.consistency_atol": 1e9})

    out = uncertainty.run(cfg, store=store)
    assert [m["member"] for m in out["members"]] == [0, 1, 2, 3, 4]
    assert out["split_sizes"] == {"shift_eval": 6, "shift_fit": 6, "test": 6, "val": 6}
    assert out["settings"]["member_seeds"] == [0, 1, 2, 3, 4]
    assert set(out["methods"]) == set(uncertainty.METHODS)
    assert store.has("07_uncertainty", "uncertainty_comparison.json")
    assert store.pull_json("07_uncertainty", "uncertainty_comparison.json")["run_id"] == \
        store.run_id


def test_a_member_trains_into_its_own_files(store, tmp_path, monkeypatch):
    """One real epoch of member 1: its files in stage 07, stage 01 untouched."""
    import json

    from scoutfield.perception import datasets, model
    from scoutfield.perception import train as train_module
    from scoutfield.utils import paths

    splits = ("train", "val", "test", "shift_fit", "shift_eval")
    loaders = {s: _loader(n=4, seed=i) for i, s in enumerate(splits)}
    loaders["meta"] = {"membership": {s: f"hash-{s}" for s in splits}}
    monkeypatch.setattr(datasets, "build_dataloaders", lambda config: loaders)
    monkeypatch.setattr(model, "build_model",
                        lambda config: model.BinaryClassifier(pretrained=False))
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path)
    monkeypatch.setattr(paths, "checkpoints_dir", lambda *p: tmp_path.joinpath(*p))
    cfg = _config(**{"train.epochs": 1, "train.amp": False,
                     "model.freeze_backbone_epochs": 0})

    summary = train_module.train(cfg, store=store, member=1)
    assert summary["member"] == 1
    assert summary["seed"] == cfg.section("uncertainty")["member_seeds"][0]
    files = set(store.stage_entry("07_uncertainty")["files"])
    assert {"member1_best.pt", "member1_last.pt", "member1_summary.json"} <= files
    assert store.stage_entry("01_perception") == {}
    written = json.loads((tmp_path / "member1_summary.json").read_text(encoding="utf-8"))
    assert written["member"] == 1
    with pytest.raises(train_module.ResumeRefused, match="member1_last.pt"):
        train_module.train(cfg, store=store, member=1)


def test_evaluate_names_the_members_it_lacks(store, tmp_path, monkeypatch):
    from scoutfield.perception import uncertainty

    _fake_pipeline(store, tmp_path, monkeypatch, members=3)
    with pytest.raises(FileNotFoundError, match="member3_best.pt.*--member"):
        uncertainty.run(_config(), store=store)


def test_figure_16_draws_the_three_methods(monkeypatch):
    from scoutfield.perception.uncertainty import LABELS, compare

    spec = importlib.util.spec_from_file_location("figs", ROOT / "experiments" /
                                                  "make_figures.py")
    figs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs)
    drawn = {}
    monkeypatch.setattr(figs, "_save", lambda fig, number: drawn.setdefault(number, fig))

    member_logits, mc_logits, labels = _synthetic()
    comparison = {"methods": compare(member_logits, mc_logits, labels, FIT_SPLITS, 15)}
    figs.figure_16_uncertainty_methods(comparison, figs._style())
    titles = [ax.get_title() for ax in drawn[16].axes]
    for label in LABELS.values():
        assert any(t.startswith(label) for t in titles)
