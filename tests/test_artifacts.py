"""
The artefact store, against a local directory standing in for the hub.

Every stage reads its inputs through ``pull`` and writes its outputs through
``push``, so these four properties carry the whole pipeline: a missing input
raises, a changed input raises, what is pushed comes back byte for byte, and the
manifest records where every file came from.
"""

from __future__ import annotations

import json

import pytest


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_pull_of_a_missing_required_file_raises_naming_where_it_looked(store):
    with pytest.raises(FileNotFoundError) as excinfo:
        store.pull("02_calibration", "calibration_sweep.json")
    message = str(excinfo.value)
    assert "runs/rtest/02_calibration/calibration_sweep.json" in message
    assert "rtest" in message and "02_calibration" in message


def test_pull_of_a_missing_optional_file_returns_none(store):
    assert store.pull("02_calibration", "calibration_sweep.json", required=False) is None


def test_push_then_pull_round_trips(store, tmp_path):
    src = _write(tmp_path / "out" / "summary.json", '{"a": 1}')
    store.push("04_sweep", [src], "test push")
    pulled = store.pull("04_sweep", "summary.json")
    assert pulled.read_bytes() == src.read_bytes()
    assert store.pull_json("04_sweep", "summary.json") == {"a": 1}


def test_a_file_changed_after_it_was_recorded_raises(store, tmp_path):
    src = _write(tmp_path / "out" / "summary.json", '{"a": 1}')
    store.push("04_sweep", [src], "test push")
    # Change the stored bytes behind the manifest's back.
    stored = store.backend.root / store.remote_path("04_sweep", "summary.json")
    stored.write_text('{"a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        store.pull("04_sweep", "summary.json")


def test_a_file_uploaded_but_not_recorded_counts_as_missing(store):
    # An interrupted stage can leave an upload with no manifest entry.
    store.backend.upload({store.remote_path("03_ppo", "ppo_seed0.zip"): b"partial"}, "x")
    assert not store.has("03_ppo", "ppo_seed0.zip")
    with pytest.raises(FileNotFoundError):
        store.pull("03_ppo", "ppo_seed0.zip")


def test_manifest_records_files_commits_and_inputs(store, tmp_path):
    upstream = _write(tmp_path / "a" / "calibration_sweep.json", "{}")
    upload_commit = store.push("02_calibration", [upstream], "calibration")

    store.pull("02_calibration", "calibration_sweep.json")
    out = _write(tmp_path / "b" / "summary.json", "{}")
    store.push("04_sweep", [out], "sweep")
    store.record_inputs("04_sweep")

    manifest = store.manifest()
    entry = manifest["stages"]["02_calibration"]["files"]["calibration_sweep.json"]
    assert entry["hub_commit"] == upload_commit
    assert entry["size"] == 2 and len(entry["sha256"]) == 64
    assert "git_commit" in manifest["stages"]["02_calibration"]
    assert "pushed_at" in entry

    inputs = manifest["stages"]["04_sweep"]["inputs"]
    assert inputs == [{"stage": "02_calibration", "name": "calibration_sweep.json",
                       "sha256": entry["sha256"], "hub_commit": upload_commit}]


def test_record_inputs_refuses_a_file_the_run_never_produced(store):
    with pytest.raises(FileNotFoundError, match="not in the manifest"):
        store.record_inputs("04_sweep", [("02_calibration", "calibration_sweep.json")])


def test_unknown_stage_names_are_refused(store, tmp_path):
    src = _write(tmp_path / "x.json", "{}")
    with pytest.raises(ValueError, match="unknown stage"):
        store.push("04_sweeps", [src], "typo")


def test_hub_writes_without_a_token_raise_with_instructions(monkeypatch):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    from scoutfield.utils.artifacts import HubBackend, StoreWriteError

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    backend = HubBackend("someone/somewhere")
    with pytest.raises(StoreWriteError, match="HF_TOKEN"):
        backend.upload({"runs/r/01_perception/x": b"x"}, "no token")


def test_local_store_switch(monkeypatch, tmp_path):
    from scoutfield.utils.artifacts import LOCAL_STORE_ENV, LocalBackend, get_store

    monkeypatch.setenv(LOCAL_STORE_ENV, str(tmp_path / "offline"))
    store = get_store(run_id="r2026-01-01",
                      settings={"hub_repo": "x/y", "run_id": "r2026-01-01",
                                "local_cache": str(tmp_path / "cache")})
    assert isinstance(store.backend, LocalBackend)
    src = _write(tmp_path / "f.json", json.dumps({"ok": True}))
    store.push("05_tau", [src], "offline")
    assert store.pull_json("05_tau", "f.json") == {"ok": True}
