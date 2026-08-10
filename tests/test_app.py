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
    return {"onnx": onnx_path, "report": report, "config": cfg}


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


def test_latency_reports_the_hardware_it_was_measured_on(exported):
    """A latency figure without the machine behind it is not reproducible."""
    from scoutfield.app.export_onnx import measure_latency

    result = measure_latency(str(exported["onnx"]), n_runs=5, warmup=2)
    assert result["median_ms"] > 0
    # p95 cannot be below the median.
    assert result["p95_ms"] >= result["median_ms"]
    for key in ("processor", "platform", "onnxruntime", "providers"):
        assert result["hardware"][key]
