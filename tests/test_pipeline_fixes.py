"""
Regression tests for the pipeline changes: where the reference temperature comes
from, the sweep's refusal to write a row without its calibration, the shift-centred
calibration, logit pools tied to their split, the three degradation spans, and the
content-hash grouping of the split.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _push_json(store, stage, name, data, tmp_path):
    path = tmp_path / "push" / stage / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    store.push(stage, [path], name)
    return path


def _driver(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  ROOT / "experiments" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------- B1: reference temperature

def test_reference_temperature_is_read_per_condition(store, tmp_path):
    from scoutfield.envs.field_env import _reference_temperature

    _push_json(store, "02_calibration", "calibration_sweep.json",
               {"lab_fitted_temperature": 0.8, "shift_fitted_temperature": 2.5}, tmp_path)
    assert _reference_temperature("lab_fitted", store=store) == 0.8
    assert _reference_temperature("shift_fitted", store=store) == 2.5


def test_reference_temperature_raises_when_calibration_is_missing(store):
    from scoutfield.envs.field_env import _reference_temperature

    with pytest.raises(FileNotFoundError, match="02_calibration"):
        _reference_temperature("shift_fitted", store=store)


def test_reference_temperature_falls_back_only_when_asked(store):
    from scoutfield.envs.field_env import _reference_temperature

    assert _reference_temperature("shift_fitted", required=False, store=store) == 1.0


def test_reference_temperature_rejects_an_unknown_condition(store):
    from scoutfield.envs.field_env import _reference_temperature

    with pytest.raises(KeyError, match="unknown reference condition"):
        _reference_temperature("fitted", store=store)


# ---------------------------------------------------- B2: no row without an ECE

def test_sweep_aborts_on_a_row_without_an_ece(store, tmp_path, monkeypatch):
    sweep = _driver("04_sweep.py")
    _push_json(store, "02_calibration", "calibration_sweep.json",
               {"lab_fitted_temperature": 0.8, "shift_fitted_temperature": 2.5,
                "sweeps": {}}, tmp_path)
    job = {"run_id": "lawnmower__s0__T1__tau0.75__sig1.5", "agent": "lawnmower",
           "seed": 0, "temperature": 1.0, "tau": 0.75, "sigma": 1.5,
           "ppo_train_seed": None, "reference_condition": "shift_fitted"}
    row = {**job, "episodes": 1, "detections_per_joule": 0.1, "recall": 0.5,
           "ece": None}

    monkeypatch.setattr(sweep, "get_store", lambda: store)
    monkeypatch.setattr(sweep, "build_job_list", lambda cfg: [job])
    monkeypatch.setattr(sweep, "run_job", lambda job, cfg, store=None: row)
    monkeypatch.setattr(sweep, "results_dir", lambda: tmp_path / "results")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    for name in ("sweep.yaml", "ppo_field32.yaml"):
        (tmp_path / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())
    monkeypatch.setattr(sys, "argv", ["04_sweep.py"])

    with pytest.raises(RuntimeError, match="no ECE"):
        sweep.main()
    csv_path = tmp_path / "results" / "sweep_results.csv"
    assert not csv_path.exists(), "a row without an ECE was written"


def test_resume_drops_rows_of_jobs_never_marked_done(store, tmp_path):
    """A crash between writing a row and marking its job done leaves a row the
    registry does not hold; the job runs again on resume, so the stale row must go
    or the CSV ends up with two rows for one job."""
    sweep = _driver("04_sweep.py")
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "_done.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")
    with (remote / "sweep_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_id", "x"])
        writer.writeheader()
        writer.writerows([{"run_id": "a", "x": 1}, {"run_id": "b", "x": 2},
                          {"run_id": "c", "x": 3}])
    store.push("04_sweep", [remote / "_done.json", remote / "sweep_results.csv"], "state")

    local = tmp_path / "local"
    sweep.sync_resume_state(store, "04_sweep", local / "_done.json",
                            local / "sweep_results.csv")
    with (local / "sweep_results.csv").open(newline="", encoding="utf-8") as fh:
        assert [r["run_id"] for r in csv.DictReader(fh)] == ["a", "b"]


def test_resume_with_nothing_stored_clears_local_leftovers(store, tmp_path):
    sweep = _driver("04_sweep.py")
    local = tmp_path / "local"
    local.mkdir()
    (local / "_done.json").write_text('["old-run-job"]', encoding="utf-8")
    (local / "sweep_results.csv").write_text("run_id\nold-run-job\n", encoding="utf-8")
    sweep.sync_resume_state(store, "04_sweep", local / "_done.json",
                            local / "sweep_results.csv")
    assert not (local / "_done.json").exists()
    assert not (local / "sweep_results.csv").exists()


def test_calibration_table_refuses_a_condition_it_does_not_hold(store, tmp_path):
    sweep = _driver("04_sweep.py")
    sweep._CALIBRATION.clear()
    _push_json(store, "02_calibration", "calibration_sweep.json",
               {"sweeps": {"shift_eval": {"split": "shift_eval",
                                          "relative_to": "shift_fitted",
                                          "sweep": {"1": {"ece": 0.01}}}}}, tmp_path)
    assert sweep._calibration_table(store, "shift_eval", "shift_fitted") == {"1": {"ece": 0.01}}
    with pytest.raises(KeyError, match="no sweep"):
        sweep._calibration_table(store, "shift_eval", "lab_fitted")


def test_named_condition_jobs_are_distinct_from_the_grid():
    from scoutfield.config import load_config

    sweep = _driver("04_sweep.py")
    cfg = load_config(ROOT / "configs" / "sweep.yaml")
    jobs = sweep.build_job_list(cfg)
    ids = [j["run_id"] for j in jobs]
    assert len(ids) == len(set(ids)), "two jobs share a run id"
    lab = [j for j in jobs if j["reference_condition"] == "lab_fitted"]
    assert lab and all(j["temperature"] == 1.0 for j in lab)


def test_episode_metrics_add_f1_and_false_alarms_per_joule():
    sweep = _driver("04_sweep.py")
    info = {"recall": 0.5, "precision": 0.25, "detections_per_joule": 0.1,
            "false_alarms": 6, "coverage": 0.3, "time_to_first_detection": 4,
            "energy_spent": 300.0}
    out = sweep.episode_metrics(info)
    assert out["f1"] == pytest.approx(2 * 0.5 * 0.25 / 0.75)
    assert out["false_alarms_per_joule"] == pytest.approx(0.02)
    assert sweep.episode_metrics({**info, "precision": 0.0, "recall": 0.0})["f1"] == 0.0


# ------------------------------------------------ B5: shift-centred calibration

def _overconfident_logits(n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    # Calibrated logits scaled up by 3: the network is overconfident, as under shift.
    return 3.0 * rng.normal(np.where(labels == 1, 0.8, -0.8), 1.0), labels


def test_shift_fitted_sweep_changes_sign_at_the_calibrated_point():
    from scoutfield.perception.calibrate import check_sign_change, fit_temperature, summarise

    fit_logits, fit_labels = _overconfident_logits(20000, 1)
    eval_logits, eval_labels = _overconfident_logits(20000, 2)
    t_shift = fit_temperature(fit_logits, fit_labels)
    sweep = summarise(eval_logits, eval_labels, [0.3, 0.5, 1.0, 2.0, 3.0, 4.0],
                      t_shift)["sweep"]
    check_sign_change(sweep)  # does not raise
    assert sweep["0.3"]["signed_calibration_error"] > 0
    assert sweep["4"]["signed_calibration_error"] < 0


def test_a_sweep_centred_on_the_wrong_temperature_is_refused():
    """Centred on a temperature that is too sharp for this data (as a lab-fitted
    one is under shift), the sweep is still overconfident above T = 1, and stops.

    The fixture's calibrated log-odds is 1.6 z (two unit-variance Gaussians at
    +/-0.8), and its logits are 3 z, so it is calibrated near T = 1.875 relative to
    a reference of 1.0. T = 1.5 is therefore still overconfident.
    """
    from scoutfield.perception.calibrate import check_sign_change, summarise

    logits, labels = _overconfident_logits(20000, 3)
    sweep = summarise(logits, labels, [0.3, 1.0, 1.5], 1.0)["sweep"]
    with pytest.raises(AssertionError, match="does not change sign"):
        check_sign_change(sweep)


# --------------------------------------------------- B6: pools tied to the split

def test_pool_carries_its_membership(tmp_path):
    from scoutfield.perception.adapter import load_logit_pool, pool_membership, save_logit_pool

    path = tmp_path / "logit_pool_shift_eval.npz"
    save_logit_pool(path, {0: np.array([-1.0]), 1: np.array([2.0])},
                    {0: ["train/a/x.jpg"], 1: ["train/b/y.jpg"]},
                    membership="abc123", split="shift_eval")
    assert pool_membership(path) == "abc123"
    assert set(load_logit_pool(path)) == {0, 1}


def _store_with_pool(store, tmp_path, stored_membership, recorded_membership):
    from scoutfield.perception.adapter import save_logit_pool

    pool_path = tmp_path / "pool" / "logit_pool_shift_eval.npz"
    save_logit_pool(pool_path, {0: np.array([-1.0, -2.0]), 1: np.array([2.0])},
                    membership=stored_membership, split="shift_eval")
    store.push("02_calibration", [pool_path], "pool")
    _push_json(store, "02_calibration", "calibration_sweep.json",
               {"pools": {"shift_eval": {"membership": recorded_membership}}}, tmp_path)


def test_a_pool_matching_its_recorded_split_is_used(store, tmp_path):
    from scoutfield.perception.adapter import load_or_build_logit_pool

    _store_with_pool(store, tmp_path, "same", "same")
    pool = load_or_build_logit_pool("shift_eval", store=store)
    assert pool[0].tolist() == [-1.0, -2.0]


def test_a_pool_from_another_split_is_rebuilt_not_used(store, tmp_path):
    """A mismatched pool goes to the rebuild path, which needs the stage 01
    checkpoint. It is absent here, so the attempt raises instead of using the
    wrong pool."""
    from scoutfield.perception.adapter import load_or_build_logit_pool

    _store_with_pool(store, tmp_path, "old-split", "new-split")
    with pytest.raises(FileNotFoundError, match="best.pt"):
        load_or_build_logit_pool("shift_eval", store=store)


# -------------------------------------------------- B8: three degradation spans

def _span_csv(path):
    rows = []
    values = {"lawnmower": {0.3: 0.10, 1.0: 0.08, 4.0: 0.01},
              "greedy_entropy": {0.3: 0.06, 1.0: 0.04, 4.0: 0.035}}
    for agent, by_t in values.items():
        for t, base in by_t.items():
            for seed in range(5):
                for sigma in (0.75, 1.5, 3.0):
                    rows.append({"run_id": f"{agent}{seed}{t}{sigma}", "agent": agent,
                                 "seed": seed, "temperature": t, "sigma": sigma,
                                 "reference_condition": "shift_fitted",
                                 "detections_per_joule": base * (1 + 0.01 * seed + 0.001 * sigma),
                                 "precision": 0.4, "recall": 0.3, "f1": 0.34,
                                 "false_alarms_per_joule": 0.02,
                                 "ece": 0.02, "signed_calibration_error": 0.01,
                                 "accuracy": 0.8, "reference_temperature": 2.0,
                                 "effective_divisor": 2.0 * t})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_summary_reports_three_spans_with_intervals(tmp_path):
    from scoutfield.analysis.summary import build_summary
    from scoutfield.config import load_config

    cfg = load_config(ROOT / "configs" / "sweep.yaml")
    cfg.data["statistics"]["bootstrap_resamples"] = 300
    summary = build_summary(_span_csv(tmp_path / "sweep.csv"), cfg)
    spans = summary["degradation"]
    assert set(spans) == {"full", "overconfident", "underconfident"}

    lawn = spans["full"]["agents"]["lawnmower"]
    assert lawn["ratio"] == pytest.approx(10.0, rel=0.05)
    assert lawn["ci_low"] <= lawn["ratio"] <= lawn["ci_high"]
    assert lawn["probability_of_improvement"] == 1.0
    # The full span factors into the two halves.
    over = spans["overconfident"]["agents"]["lawnmower"]["ratio"]
    under = spans["underconfident"]["agents"]["lawnmower"]["ratio"]
    assert over * under == pytest.approx(lawn["ratio"], rel=1e-9)
    assert summary["cells"]["lawnmower@T1"]["f1"]["value"] == pytest.approx(0.34)


def test_named_conditions_are_kept_out_of_the_temperature_cells(tmp_path):
    from scoutfield.analysis.summary import build_summary
    from scoutfield.config import load_config

    path = _span_csv(tmp_path / "sweep.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    extra = [{**r, "reference_condition": "lab_fitted", "precision": 0.1,
              "run_id": r["run_id"] + "lab"}
             for r in rows if float(r["temperature"]) == 1.0]
    with open(path, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=list(rows[0])).writerows(extra)

    cfg = load_config(ROOT / "configs" / "sweep.yaml")
    cfg.data["statistics"]["bootstrap_resamples"] = 200
    summary = build_summary(path, cfg)
    assert summary["cells"]["lawnmower@T1"]["precision"]["value"] == pytest.approx(0.4)
    assert summary["conditions"]["lab_fitted"]["lawnmower"]["precision"]["value"] \
        == pytest.approx(0.1)


# ---------------------------------------------------- B10: duplicates stay together

def _images(tmp_path, spec):
    items = []
    for rel, payload in spec:
        path = tmp_path / "color" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        items.append((path, rel.split("/")[0]))
    return items


def test_a_duplicated_image_stays_within_one_split(tmp_path):
    from scoutfield.perception.datasets import _split_by_source, content_hashes, item_key

    spec = [(f"Tomato___healthy/leaf_{i:03d}.jpg", f"img{i}".encode()) for i in range(40)]
    # The same bytes under an unrelated name: invisible to the source key.
    spec.append(("Tomato___healthy/unrelated_copy.jpg", b"img7"))
    items = _images(tmp_path, spec)
    hashes = content_hashes(items, tmp_path / "cache.json")

    for seed in range(10):
        parts = _split_by_source(items, 0.3, 0.3, seed, hashes=hashes)
        homes = {name for name, members in parts.items()
                 for path, _ in members if hashes[item_key(path)] == hashes[item_key(items[7][0])]}
        assert len(homes) == 1, f"seed {seed}: identical files in {homes}"
        assert sum(len(v) for v in parts.values()) == len(items)


def test_content_hashes_are_cached(tmp_path):
    from scoutfield.perception.datasets import content_hashes

    items = _images(tmp_path, [("Apple___healthy/a.jpg", b"x")])
    cache = tmp_path / "cache.json"
    first = content_hashes(items, cache)
    assert cache.exists()
    assert content_hashes(items, cache) == first


def test_shift_split_is_stratified_and_keeps_copies_together(tmp_path):
    from scoutfield.perception.datasets import _split_shift, content_hashes, item_key

    spec = [(f"Apple leaf/h{i}.jpg", f"h{i}".encode()) for i in range(20)]
    spec += [(f"Apple Scab Leaf/d{i}.jpg", f"d{i}".encode()) for i in range(40)]
    spec.append(("Apple Scab Leaf/copy_of_d3.jpg", b"d3"))
    items = _images(tmp_path, spec)
    hashes = content_hashes(items)
    parts = _split_shift(items, 0.5, seed=0, hashes=hashes)

    healthy = {k: sum(1 for _, c in v if c == "Apple leaf") for k, v in parts.items()}
    assert healthy == {"shift_fit": 10, "shift_eval": 10}
    homes = {k for k, v in parts.items() for p, _ in v if hashes[item_key(p)] == hashes[
        item_key(items[23][0])]}
    assert len(homes) == 1
