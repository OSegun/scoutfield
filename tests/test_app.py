"""
Export and inference tests.

The deployment path is where a calibration project can quietly undo itself: an
export that shifts logits, or a preprocessing mismatch between training and
serving, degrades calibration far more than accuracy and presents as a mysterious
regression rather than a bug. These tests are the guard.

They export an *untrained* model. Nothing here measures accuracy — the weights are
random — but numerical equivalence and latency are properties of the graph and the
machine, not of the weights.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """An exported ONNX model plus the torch model it came from."""
    from scoutfield.app.export_onnx import export
    from scoutfield.config import load_config
    from scoutfield.perception.model import build_model
    from scoutfield.utils.checkpoint import save_training_state

    tmp = tmp_path_factory.mktemp("export")
    cfg = load_config("configs/perception.yaml")
    model = build_model(cfg)

    checkpoint = tmp / "untrained.pt"
    save_training_state(checkpoint, {"model": model.state_dict(), "epoch": -1})

    onnx_path = tmp / "model.onnx"
    report = export(str(checkpoint), str(onnx_path), config=cfg)
    return {"onnx": onnx_path, "report": report, "config": cfg, "checkpoint": checkpoint}


def test_export_preserves_logits(exported):
    """A 1e-3 logit shift moves ECE, and ECE is what this project measures."""
    assert exported["report"]["max_abs_diff"] < 1e-4, exported["report"]


def test_export_accepts_a_dynamic_batch(exported):
    """The batch axis must stay dynamic or serving is stuck at batch 1."""
    import onnxruntime as ort

    from scoutfield.app.export_onnx import INPUT_NAME, OUTPUT_NAME

    session = ort.InferenceSession(str(exported["onnx"]),
                                   providers=["CPUExecutionProvider"])
    for batch in (1, 4):
        x = np.random.randn(batch, 3, 224, 224).astype(np.float32)
        out = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
        assert out.ravel().shape == (batch,)


def test_inference_prediction_is_independent_of_temperature(exported):
    """The deployed path must preserve the invariance the study rests on."""
    from PIL import Image

    from scoutfield.app.inference import ScoutPlanInference

    image = Image.fromarray(
        (np.random.default_rng(0).random((256, 256, 3)) * 255).astype("uint8")
    )

    results = [ScoutPlanInference(exported["onnx"], temperature=T).predict(image)
               for T in (0.3, 1.0, 4.0)]

    assert len({r["prediction"] for r in results}) == 1, "prediction moved with T"
    assert len({round(r["logit"], 6) for r in results}) == 1, "logit moved with T"
    # The reported probability must move, or the slider is doing nothing.
    assert len({round(r["probability"], 6) for r in results}) == 3


def test_inference_probability_is_a_probability(exported):
    from PIL import Image

    from scoutfield.app.inference import ScoutPlanInference

    image = Image.fromarray(np.zeros((224, 224, 3), dtype="uint8"))
    out = ScoutPlanInference(exported["onnx"], temperature=1.7).predict(image)
    assert 0.0 <= out["probability"] <= 1.0
    assert out["prediction"] in (0, 1)
    assert out["temperature"] == 1.7


def test_inference_fails_loudly_without_an_exported_model(tmp_path):
    from scoutfield.app.inference import ScoutPlanInference

    with pytest.raises(FileNotFoundError, match="export_onnx"):
        ScoutPlanInference(tmp_path / "absent.onnx")


def test_export_rejects_a_graph_that_differs_from_the_model(exported, tmp_path):
    """The equivalence check must fail loudly, not report a difference and carry on."""
    from scoutfield.app.export_onnx import _model_from_checkpoint, verify_equivalence
    from scoutfield.perception.model import build_model
    from scoutfield.utils.checkpoint import save_training_state

    other = build_model(exported["config"])
    with torch.no_grad():
        other.backbone.classifier[1].bias.add_(5.0)
    changed = tmp_path / "changed.pt"
    save_training_state(changed, {"model": other.state_dict(), "epoch": -1})
    model = _model_from_checkpoint(changed, exported["config"])

    with pytest.raises(AssertionError, match="disagree"):
        verify_equivalence(model, exported["onnx"], image_size=224, n_inputs=4,
                           batch_size=2, seed=0, atol=1e-4)


def test_export_record_names_its_inputs(exported):
    """Opset, input shape, checkpoint SHA-256 and commit travel with the file."""
    from pathlib import Path

    from scoutfield.app.export_onnx import export_record

    report = exported["report"]
    record = export_record(report, Path(report["path"]), exported["checkpoint"],
                           "local file", "rtest")
    assert record["onnx"]["opset"] == report["opset"]
    assert record["onnx"]["input"]["shape"] == ["batch", 3, 224, 224]
    assert len(record["checkpoint"]["sha256"]) == 64
    assert "git_commit" in record and "git_dirty" in record
    assert record["verification"]["predictions_identical"] is True


def test_a_missing_local_checkpoint_raises_and_names_it(tmp_path):
    from scoutfield.app.export_onnx import load_deployment_config, resolve_checkpoint

    with pytest.raises(FileNotFoundError, match="absent.pt"):
        resolve_checkpoint(load_deployment_config(), override=tmp_path / "absent.pt")


def test_push_is_refused_for_a_checkpoint_the_run_did_not_record(exported, store,
                                                                monkeypatch, tmp_path):
    """A pushed model must trace to the checkpoint filed under the same run."""
    from scoutfield.app import export_onnx

    store.push("01_perception", [exported["checkpoint"]], "a different checkpoint",
               names={str(exported["checkpoint"]): "best.pt"})
    other = tmp_path / "other.pt"
    other.write_bytes(exported["checkpoint"].read_bytes() + b"\0")
    monkeypatch.setattr(export_onnx, "work_dir", lambda cfg, run: tmp_path / run)
    monkeypatch.setattr(export_onnx, "export", lambda ckpt, out: {
        **exported["report"], "path": str(exported["onnx"])})
    (tmp_path / store.run_id).mkdir()
    (tmp_path / store.run_id / "classifier.onnx").write_bytes(exported["onnx"].read_bytes())

    with pytest.raises(ValueError, match="refusing to push"):
        export_onnx.run_export(checkpoint=other, push=True, store=store)
    assert not store.has("08_deployment", "classifier.onnx")


def test_latency_reports_the_hardware_it_was_measured_on(exported):
    """A latency figure without the machine behind it is not reproducible."""
    from scoutfield.app.export_onnx import measure_latency

    result = measure_latency(str(exported["onnx"]), n_runs=5, warmup=2)
    assert result["median_ms"] > 0
    # p95 cannot be below the median.
    assert result["p95_ms"] >= result["median_ms"]
    for key in ("processor", "platform", "onnxruntime", "providers"):
        assert result["hardware"][key]


def test_latency_record_carries_hardware_and_every_thread_count(exported, store,
                                                                 monkeypatch, tmp_path):
    from scoutfield.app import export_onnx

    monkeypatch.setattr(export_onnx, "work_dir", lambda cfg, run: tmp_path)
    (tmp_path / "classifier.onnx").write_bytes(exported["onnx"].read_bytes())
    real = export_onnx.measure_latency
    monkeypatch.setattr(export_onnx, "measure_latency",
                        lambda path, **kw: real(path, **{**kw, "n_runs": 3, "warmup": 1}))

    record = export_onnx.run_latency(store=store)
    assert [m["intra_op_threads"] for m in record["measurements"]] == \
        record["settings"]["intra_op_threads"]
    for key in ("cpu_model", "logical_cores", "ram_gb", "os", "python", "onnxruntime"):
        assert record["hardware"][key], key
    assert (tmp_path / "deployment_latency.json").is_file()
    assert "not drone hardware" in record["what"]


def test_latency_push_is_refused_for_a_model_not_filed(exported, store, monkeypatch,
                                                       tmp_path):
    from scoutfield.app import export_onnx

    monkeypatch.setattr(export_onnx, "work_dir", lambda cfg, run: tmp_path)
    (tmp_path / "classifier.onnx").write_bytes(exported["onnx"].read_bytes())
    real = export_onnx.measure_latency
    monkeypatch.setattr(export_onnx, "measure_latency",
                        lambda path, **kw: real(path, **{**kw, "n_runs": 2, "warmup": 1}))
    with pytest.raises(ValueError, match="refusing to push"):
        export_onnx.run_latency(store=store, push=True)
    assert not store.has("08_deployment", "deployment_latency.json")


def test_latency_without_an_export_names_the_command(store, monkeypatch, tmp_path):
    from scoutfield.app import export_onnx

    monkeypatch.setattr(export_onnx, "work_dir", lambda cfg, run: tmp_path)
    with pytest.raises(FileNotFoundError, match="08_export_onnx.py"):
        export_onnx.run_latency(store=store)
