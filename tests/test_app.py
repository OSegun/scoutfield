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

pytest.importorskip("onnxruntime")

try:  # only the export tests need torch; the demonstration's tests run without it
    import torch
except ImportError:  # pragma: no cover - depends on the environment
    torch = None


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """An exported ONNX model plus the torch model it came from."""
    pytest.importorskip("torch")
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

    from scoutfield.app.inference import ScoutFieldInference

    image = Image.fromarray(
        (np.random.default_rng(0).random((256, 256, 3)) * 255).astype("uint8")
    )

    results = [ScoutFieldInference(exported["onnx"], temperature=T).predict(image)
               for T in (0.3, 1.0, 4.0)]

    assert len({r["prediction"] for r in results}) == 1, "prediction moved with T"
    assert len({round(r["logit"], 6) for r in results}) == 1, "logit moved with T"
    # The reported probability must move, or the slider is doing nothing.
    assert len({round(r["probability"], 6) for r in results}) == 3


def test_inference_probability_is_a_probability(exported):
    from PIL import Image

    from scoutfield.app.inference import ScoutFieldInference

    image = Image.fromarray(np.zeros((224, 224, 3), dtype="uint8"))
    out = ScoutFieldInference(exported["onnx"], temperature=1.7).predict(image)
    assert 0.0 <= out["probability"] <= 1.0
    assert out["prediction"] in (0, 1)
    assert out["temperature"] == 1.7


def test_inference_fails_loudly_without_an_exported_model(tmp_path):
    from scoutfield.app.inference import ScoutFieldInference

    with pytest.raises(FileNotFoundError, match="export_onnx"):
        ScoutFieldInference(tmp_path / "absent.onnx")


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


# ------------------------------------------------- demo correctness (Panels 1, 2)

FITTED = 3.3843  # any value other than 1 shows whether the fitted temperature is used


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_panel1_slider_at_one_gives_the_calibrated_probability(exported, tmp_path):
    """Slider = 1 must divide the logit by the fitted temperature, not by 1."""
    import json

    from PIL import Image

    from scoutfield.app.inference import ScoutFieldInference

    results = tmp_path / "calibration_sweep.json"
    results.write_text(json.dumps({"shift_fitted_temperature": FITTED,
                                   "lab_fitted_temperature": 0.99}), encoding="utf-8")
    model = ScoutFieldInference.from_results(exported["onnx"], results_json=results,
                                            condition="shift_fitted")
    image = Image.fromarray((np.random.default_rng(1).random((240, 240, 3)) * 255)
                            .astype("uint8"))

    at_one = model.predict(image, relative_temperature=1.0)
    assert at_one["probability"] == pytest.approx(_sigmoid(at_one["logit"] / FITTED))
    assert at_one["effective_temperature"] == pytest.approx(FITTED)

    at_four = model.predict(image, relative_temperature=4.0)
    assert at_four["probability"] == pytest.approx(_sigmoid(at_four["logit"] / (4 * FITTED)))
    assert at_four["prediction"] == at_one["prediction"]


def _synthetic_calibration_store(store, tmp_path, split="shift_eval"):
    """Stage 02 as the sweep reads it, with a tiny synthetic pool: no network, no CNN."""
    import json

    from scoutfield.perception.adapter import logit_pool_filename, save_logit_pool

    rng = np.random.default_rng(0)
    pool = {0: rng.normal(-1.5, 1.0, 60), 1: rng.normal(1.5, 1.0, 60)}
    pool_path = tmp_path / logit_pool_filename(split)
    save_logit_pool(pool_path, pool, membership="m-synthetic", split=split)
    cal = tmp_path / "calibration_sweep.json"
    cal.write_text(json.dumps({"shift_fitted_temperature": FITTED,
                               "lab_fitted_temperature": 0.99,
                               "pools": {split: {"membership": "m-synthetic"}}}),
                   encoding="utf-8")
    store.push("02_calibration", [cal, pool_path], "synthetic stage 02")
    return pool


def test_panel2_observes_the_sweep_pool_at_the_shift_fitted_reference(store, tmp_path):
    from scoutfield.app import demo

    settings = demo.load_settings()
    # The conditions come from the sweep's own config, not from the app.
    assert settings.pool_split == "shift_eval"
    assert settings.reference_condition == "shift_fitted"

    expected = _synthetic_calibration_store(store, tmp_path, settings.pool_split)
    pool = demo.load_pool(settings.pool_split, store=store)
    reference = demo.fitted_temperature(settings.reference_condition, store=store)
    assert reference == pytest.approx(FITTED)
    for label in (0, 1):
        np.testing.assert_allclose(pool[label], expected[label])

    run = demo.run_episode(demo.make_planner("lawnmower", 0), settings, pool, reference,
                           temperature=4.0, seed=0)
    clf = run["classifier"]
    assert clf.pool_split == "shift_eval"
    assert clf.reference_temperature * clf.T == pytest.approx(4.0 * FITTED)
    assert run["env"].tau == settings.tau
    for key in ("recall", "precision", "detections_per_joule", "coverage"):
        assert key in run["info"]


def test_panel2_runs_without_importing_torch(store, tmp_path):
    """Panel 2 must not need PyTorch: the hosted app has none."""
    import subprocess
    import sys

    _synthetic_calibration_store(store, tmp_path)
    script = (
        "import sys\n"
        "from scoutfield.app import demo\n"
        "from scoutfield.utils.artifacts import ArtifactStore, LocalBackend\n"
        f"store = ArtifactStore(LocalBackend(r'{store.backend.root}'), 'rtest', "
        f"r'{tmp_path / 'cache2'}')\n"
        "s = demo.load_settings()\n"
        "pool = demo.load_pool(s.pool_split, store=store)\n"
        "ref = demo.fitted_temperature(s.reference_condition, store=store)\n"
        "for name in demo.BASELINES:\n"
        "    demo.run_episode(demo.make_planner(name, 0), s, pool, ref, 1.0, 0)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=300, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


# ------------------------------- demonstration: policy, offline folder, samples
#
# None of these needs the network, downloaded weights or PyTorch: the classifier
# and the policy are tiny ONNX graphs built here, and the store is a directory.

N_OBS, N_ACTIONS = 206, 8


def _tiny_classifier(path):
    """``image`` (N, 3, 224, 224) -> ``logit`` (N,): the mean pixel times 3."""
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["image"], ["m"], axes=[1, 2, 3], keepdims=0),
         helper.make_node("Mul", ["m", "k"], ["logit"])],
        "tiny_classifier",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, 224, 224])],
        [helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["N"])],
        [helper.make_tensor("k", TensorProto.FLOAT, [], [3.0])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path


def _tiny_policy(path, seed=0):
    """``observation`` (N, 206) -> ``action_scores`` (N, 8): one linear layer."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = np.random.default_rng(seed).normal(size=(N_OBS, N_ACTIONS)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["observation", "w"], ["action_scores"])],
        "tiny_policy",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["N", N_OBS])],
        [helper.make_tensor_value_info("action_scores", TensorProto.FLOAT,
                                       ["N", N_ACTIONS])],
        [numpy_helper.from_array(w, "w")],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path, w


def test_onnx_policy_acts_by_the_highest_score(tmp_path):
    from scoutfield.app.export_policy import ONNXPolicy

    path, w = _tiny_policy(tmp_path / "policy.onnx")
    policy = ONNXPolicy(path)
    rng = np.random.default_rng(1)
    for _ in range(50):
        obs = rng.normal(size=N_OBS)
        assert policy.act(obs, None) == int(np.argmax(obs.astype(np.float32) @ w))


def test_onnx_policy_without_a_file_names_the_export_command(tmp_path):
    from scoutfield.app.export_policy import ONNXPolicy

    with pytest.raises(FileNotFoundError, match="08_export_policy.py"):
        ONNXPolicy(tmp_path / "absent.onnx")


def test_panel2_flies_ppo_through_the_onnx_policy(store, tmp_path):
    """PPO runs in Panel 2 from the exported policy, on the sweep's pool."""
    from scoutfield.app import demo
    from scoutfield.app.export_policy import ONNXPolicy

    _synthetic_calibration_store(store, tmp_path)
    settings = demo.load_settings()
    pool = demo.load_pool(settings.pool_split, store=store)
    policy = ONNXPolicy(_tiny_policy(tmp_path / "policy.onnx")[0])
    run = demo.run_episode(demo.get_planner("ppo", 0, policy), settings, pool, FITTED,
                           temperature=1.0, seed=0)
    assert run["info"]["coverage"] > 0
    assert run["classifier"].pool_split == "shift_eval"
    with pytest.raises(ValueError, match="policy"):
        demo.get_planner("ppo", 0, None)


def _offline_config(tmp_path):
    from scoutfield.app import demo
    from scoutfield.config import Config

    cfg = demo.deployment_config()
    data = {**cfg.data, "demo": {**cfg.data["demo"], "offline_dir": str(tmp_path / "offline")}}
    return Config(data=data, path=cfg.path)


def _full_source_store(store, tmp_path):
    """A source store holding every file the app reads, all synthetic."""
    import json

    from scoutfield.app import demo

    config = demo.deployment_config()
    _synthetic_calibration_store(store, tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"cells": {}, "agents": [], "degradation": {}}),
                       encoding="utf-8")
    store.push("04_sweep", [summary], "synthetic summary")
    classifier = _tiny_classifier(tmp_path / "classifier.onnx")
    policy, _ = _tiny_policy(tmp_path / config.section("policy")["policy_file"])
    store.push("08_deployment", [classifier, policy], "synthetic exports")
    return config


def test_offline_folder_runs_the_demo_with_the_store_unreachable(store, tmp_path,
                                                                  monkeypatch):
    """Filled once, the offline folder serves every panel with no network."""
    from PIL import Image

    from scoutfield.app import demo
    from scoutfield.app.inference import ScoutFieldInference
    from scoutfield.utils import artifacts

    _full_source_store(store, tmp_path)
    config = _offline_config(tmp_path)
    report = demo.fill_offline(store, config)
    assert len(report["files"]) == len(demo.required_files(config))

    # From here on, any attempt to reach the hub fails, and the local-store
    # override is cleared, so only the offline folder can answer.
    def unreachable(*args, **kwargs):
        raise OSError("network is switched off in this test")

    monkeypatch.setattr(artifacts, "HubBackend", unreachable)
    monkeypatch.delenv(artifacts.LOCAL_STORE_ENV, raising=False)
    monkeypatch.setenv(demo.SOURCE_ENV, "offline")
    monkeypatch.setattr(demo, "deployment_config", lambda: config)
    settings_file = artifacts.load_settings()
    monkeypatch.setattr(artifacts, "load_settings",
                        lambda path=None: {**settings_file, "run_id": store.run_id})

    offline = demo.open_store(config)
    settings = demo.load_settings()
    pool = demo.load_pool(settings.pool_split, store=offline)
    reference = demo.fitted_temperature(settings.reference_condition, store=offline)
    model = ScoutFieldInference.from_results(
        offline.pull("08_deployment", "classifier.onnx"), condition="shift_fitted",
        store=offline)
    out = model.predict(Image.new("RGB", (300, 260), (120, 160, 90)))
    assert out["probability"] == pytest.approx(_sigmoid(out["logit"] / FITTED))

    policy = demo.load_policy(offline, config)
    for name in demo.PLANNERS:
        run = demo.run_episode(demo.get_planner(name, 0, policy), settings, pool, reference,
                               1.0, 0)
        assert "detections_per_joule" in run["info"]
    assert offline.pull_json("04_sweep", "summary.json")["cells"] == {}


def test_offline_folder_is_checked_against_the_manifest(store, tmp_path):
    """A file changed after filling is refused, as in the online store."""
    from scoutfield.app import demo

    _full_source_store(store, tmp_path)
    config = _offline_config(tmp_path)
    demo.fill_offline(store, config)
    pool = demo.offline_root(config) / "store" / "runs" / store.run_id / \
        "02_calibration" / "logit_pool_shift_eval.npz"
    pool.write_bytes(pool.read_bytes() + b"\0")
    offline = demo.offline_store(config, run_id=store.run_id)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        offline.pull("02_calibration", "logit_pool_shift_eval.npz")


def test_an_empty_offline_folder_names_the_fill_command(tmp_path):
    from scoutfield.app import demo

    with pytest.raises(FileNotFoundError, match="08_fill_demo_folder.py"):
        demo.offline_store(_offline_config(tmp_path), run_id="rtest")


def test_demo_source_is_a_setting_with_a_default(monkeypatch):
    from scoutfield.app import demo

    monkeypatch.delenv(demo.SOURCE_ENV, raising=False)
    assert demo.demo_source() == "store"
    monkeypatch.setenv(demo.SOURCE_ENV, "offline")
    assert demo.demo_source() == "offline"
    monkeypatch.setenv(demo.SOURCE_ENV, "elsewhere")
    with pytest.raises(ValueError, match="elsewhere"):
        demo.demo_source()


def test_torch_free_preprocessing_equals_the_training_transform():
    """The deployed preprocessing must give the pixels evaluation gave, exactly."""
    pytest.importorskip("torchvision")
    from PIL import Image

    from scoutfield.perception.datasets import _transforms, eval_array

    rng = np.random.default_rng(0)
    for w, h in [(256, 256), (640, 480), (480, 640), (300, 1000), (225, 225)]:
        image = Image.fromarray((rng.random((h, w, 3)) * 255).astype("uint8"))
        np.testing.assert_array_equal(eval_array(image, 224),
                                      _transforms(224, train=False)(image).numpy())


def test_samples_are_chosen_by_rule_from_the_pool(tmp_path):
    from scoutfield.app import demo
    from scoutfield.perception.adapter import save_logit_pool

    logits = {1: np.array([3.0, 2.0, 1.0, 0.5, -4.0, -0.2]),
              0: np.array([-3.0, -2.0, -1.0, -0.5, 5.0, 0.1])}
    paths = {k: [f"test/c{k}/img{i}.jpg" for i in range(6)] for k in logits}
    pool = tmp_path / "pool.npz"
    save_logit_pool(pool, logits, paths)

    picks = demo.select_samples(pool, 2, 2, 2)
    right = [p for p in picks if p["correct"]]
    wrong = [p for p in picks if not p["correct"]]
    assert len(right) == 4 and all((p["logit"] > 0) == bool(p["label"]) for p in right)
    # The most confident mistakes, one of each kind: a missed disease (-4.0) and a
    # false alarm (+5.0).
    assert sorted(p["logit"] for p in wrong) == [-4.0, 5.0]
    assert picks == demo.select_samples(pool, 2, 2, 2)  # deterministic


def test_find_image_accepts_a_kaggle_single_file_download(tmp_path):
    """``kaggle datasets download -f`` saves spaces in a name as %20."""
    from scoutfield.app import demo

    folder = tmp_path / "train" / "Apple_leaf"
    folder.mkdir(parents=True)
    saved = folder / "train_Apple%20leaf_57.jpg"
    saved.write_bytes(b"jpeg")
    assert demo.find_image(tmp_path, "train/Apple_leaf/train_Apple leaf_57.jpg") == saved
    with pytest.raises(FileNotFoundError, match="Pear_leaf"):
        demo.find_image(tmp_path, "train/Pear_leaf/train_Apple leaf_57.jpg")


# --------------------------------------- demonstration: a visitor's own images
#
# The "Analyse your leaves" panel, tested through the tiny classifier above: no
# network, no downloaded weights, no PyTorch. Images are built in memory.

LEAF_FITTED = 2.0
LEAF_TEMPERATURES = (0.3, 0.5, 1.0, 2.0, 3.0, 4.0)
LEAF_TAU = 0.75
LEAF_CONTEXT = {"run_id": "rtest", "reference_condition": "shift_fitted",
                "fitted_temperature": LEAF_FITTED, "tau": LEAF_TAU,
                "repository_url": "https://example.org/scoutfield"}


def _image_bytes(value: int, size=(64, 48), fmt="PNG") -> bytes:
    """A flat image of one grey level, encoded in memory."""
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(out, format=fmt)
    return out.getvalue()


@pytest.fixture()
def leaf_model(tmp_path):
    from scoutfield.app.inference import ScoutFieldInference

    return ScoutFieldInference(_tiny_classifier(tmp_path / "tiny.onnx"),
                              temperature=LEAF_FITTED)


def _analyse(model, uploads, **overrides):
    from scoutfield.app import demo

    limits = {"max_files": 10, "max_file_mb": 10.0, "max_megapixels": 24.0,
              "thumbnail_px": 160}
    limits.update(overrides)
    return demo.analyse_uploads(model, uploads, LEAF_TEMPERATURES, LEAF_TAU,
                                demo.UploadLimits(**limits))


def test_upload_limits_are_read_from_the_config():
    from scoutfield.app.demo import UploadLimits, deployment_config

    up = deployment_config().section("demo")["upload"]
    limits = UploadLimits.from_config()
    assert limits.max_files == int(up["max_files"])
    assert limits.max_file_mb == float(up["max_file_mb"])
    assert limits.max_megapixels == float(up["max_megapixels"])
    assert limits.thumbnail_px == int(up["thumbnail_px"])


def test_a_batch_goes_through_the_deployed_path(leaf_model):
    """Each row equals the Classify panel's prediction for the same image."""
    import io

    from PIL import Image

    uploads = [("white.png", _image_bytes(255)),
               ("grey.jpg", _image_bytes(128, fmt="JPEG")),
               ("black.png", _image_bytes(0))]
    result = _analyse(leaf_model, uploads)
    assert [r["file"] for r in result["rows"]] == ["white.png", "grey.jpg", "black.png"]
    assert result["rejected"] == []
    assert (result["low_T"], result["high_T"]) == (0.3, 4.0)
    for (_, data), row in zip(uploads, result["rows"]):
        direct = leaf_model.predict(Image.open(io.BytesIO(data)))
        assert row["logit"] == pytest.approx(direct["logit"])
        assert row["probability_calibrated"] == pytest.approx(
            _sigmoid(direct["logit"] / LEAF_FITTED))
        assert row["probability_low_T"] == pytest.approx(
            _sigmoid(direct["logit"] / (0.3 * LEAF_FITTED)))
        assert row["probability_high_T"] == pytest.approx(
            _sigmoid(direct["logit"] / (4.0 * LEAF_FITTED)))
        assert row["prediction"] == ("diseased" if direct["logit"] > 0 else "healthy")
        assert max(row["thumbnail"].size) <= 160
    s = result["summary"]
    assert (s["n"], s["diseased"], s["healthy"]) == (3, 2, 1)

    # Streamlit hands over in-memory file objects; read directly, they give the
    # same rows as the bytes.
    as_files = _analyse(leaf_model, [(n, io.BytesIO(d)) for n, d in uploads])
    assert [r["logit"] for r in as_files["rows"]] == [r["logit"] for r in result["rows"]]


def test_uncertain_means_neither_class_reaches_tau(leaf_model):
    from scoutfield.app.demo import is_uncertain

    assert is_uncertain(0.5, 0.75)
    assert is_uncertain(0.26, 0.75) and is_uncertain(0.74, 0.75)
    assert not is_uncertain(0.75, 0.75) and not is_uncertain(0.25, 0.75)
    assert not is_uncertain(0.95, 0.75) and not is_uncertain(0.05, 0.75)

    result = _analyse(leaf_model, [("white.png", _image_bytes(255)),
                                   ("grey.png", _image_bytes(128)),
                                   ("black.png", _image_bytes(0))])
    assert [r["uncertain"] for r in result["rows"]] == [False, True, False]
    assert result["summary"]["uncertain"] == 1
    assert result["summary"]["uncertain_share"] == pytest.approx(1 / 3)
    assert "between 0.25 and 0.75" in result["summary"]["sentence"]


def test_the_report_holds_numbers_and_limits_never_images(leaf_model):
    import csv
    import io

    from scoutfield.app import demo

    result = _analyse(leaf_model, [("white.png", _image_bytes(255)),
                                   ("grey.png", _image_bytes(128)),
                                   ("notes.txt", b"not an image")])
    table_text = demo.report_csv(result)
    table = list(csv.reader(io.StringIO(table_text)))
    assert table[0] == ["file", "prediction", "probability_T1", "probability_T0.3",
                        "probability_T4", "logit", "uncertain"]
    assert [row[0] for row in table[1:]] == ["white.png", "grey.png"]
    assert table[2][6] == "yes"
    assert float(table[1][2]) == pytest.approx(
        result["rows"][0]["probability_calibrated"], abs=5e-5)

    text = demo.report_markdown(result, LEAF_CONTEXT)
    for line in demo.LIMITS_TEXT + (demo.PRIVACY_TEXT,):
        assert line in text
    assert "rtest" in text and "2.0000" in text and "notes.txt" in text
    assert result["summary"]["sentence"] in text
    # Numbers and text only: no encoded pixels in either file.
    for report in (text, table_text):
        assert "data:image" not in report and "base64" not in report
        assert "\x89PNG" not in report and "JFIF" not in report


def test_uploads_over_the_limits_are_refused_by_name(leaf_model, monkeypatch):
    from PIL import Image

    files = [(f"leaf{i}.png", _image_bytes(200)) for i in range(4)]
    result = _analyse(leaf_model, files, max_files=3)
    assert len(result["rows"]) == 3
    assert result["rejected"] == [("leaf3.png: over the limit of 3 images per batch; "
                                   "analyse it in another batch")]

    big = _image_bytes(200, size=(400, 400))
    result = _analyse(leaf_model, [("big.png", big)],
                      max_file_mb=len(big) / 2 / 1024 ** 2)
    assert result["rows"] == []
    assert result["rejected"][0].startswith("big.png: ")
    assert "MB limit" in result["rejected"][0]

    # Refused from the header, before any pixel is decoded.
    wide = _image_bytes(200, size=(3000, 1000))
    decoded = []
    original = Image.Image.load
    monkeypatch.setattr(Image.Image, "load",
                        lambda self: decoded.append(self.size) or original(self))
    result = _analyse(leaf_model, [("wide.png", wide)], max_megapixels=2.0)
    assert decoded == []
    assert "3000 x 1000 pixels is over the 2 megapixel limit" in result["rejected"][0]
    monkeypatch.undo()

    result = _analyse(leaf_model, [("anim.gif", _image_bytes(200, fmt="GIF")),
                                   ("text.png", b"\x89PNG broken")])
    assert result["rows"] == []
    assert result["rejected"] == ["anim.gif: GIF is not accepted; use JPG or PNG",
                                  "text.png: not a readable JPG or PNG image"]


def test_analysing_uploads_writes_nothing_to_disk(leaf_model, monkeypatch, tmp_path):
    """Privacy: a visitor's images are processed in memory and never written."""
    import builtins
    import io
    import os
    import tempfile

    from scoutfield.app import demo

    uploads = [("white.png", _image_bytes(255)),
               ("grey.jpg", _image_bytes(128, fmt="JPEG"))]
    writes = []
    real_open, real_os_open = builtins.open, os.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            writes.append(str(path))
        return real_os_open(path, flags, *args, **kwargs)

    temp_before = set(os.listdir(tempfile.gettempdir()))
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.chdir(tmp_path)

    result = _analyse(leaf_model, uploads)
    demo.report_markdown(result, LEAF_CONTEXT)
    demo.report_csv(result)
    monkeypatch.undo()

    assert len(result["rows"]) == 2
    assert writes == []
    assert [p.name for p in tmp_path.iterdir()] == ["tiny.onnx"]
    assert set(os.listdir(tempfile.gettempdir())) - temp_before == set()


def test_sample_check_compares_the_deployed_path_with_stage02(leaf_model, tmp_path):
    """A recorded logit off by more than the tolerance, or of the other sign, fails."""
    import io

    from PIL import Image

    from scoutfield.app.demo import compare_sample_logits

    samples = []
    for i, value in enumerate((255, 0)):
        path = tmp_path / f"s{i}.png"
        path.write_bytes(_image_bytes(value))
        logit = leaf_model.predict(Image.open(io.BytesIO(path.read_bytes())))["logit"]
        samples.append({"store_name": path.name, "dataset": "plantdoc", "key": f"k/{i}",
                        "logit": logit, "path": path})

    good = compare_sample_logits(leaf_model, samples, atol=1e-4)
    assert good["passed"] and good["n_samples"] == 2
    assert good["max_abs_difference"] < 1e-6

    off = [dict(samples[0], logit=samples[0]["logit"] + 1e-3), samples[1]]
    assert not compare_sample_logits(leaf_model, off, atol=1e-4)["passed"]

    flipped = [dict(samples[0], logit=-samples[0]["logit"]), samples[1]]
    result = compare_sample_logits(leaf_model, flipped, atol=1e3)
    assert not result["passed"]
    assert result["samples"][0]["same_prediction"] is False

    assert not compare_sample_logits(leaf_model, [], atol=1e-4)["passed"]


# ------------------------------------------- demonstration: config and caches
#
# The hosted app once raised KeyError on demo.upload after a redeploy: the page's
# cache was keyed on the loader's code, not on the YAML it read, so it kept serving
# the config loaded before the setting existed.

def test_config_stamp_follows_the_config_files(tmp_path):
    from scoutfield.app.demo import config_stamp

    (tmp_path / "configs").mkdir()
    yaml_file = tmp_path / "configs" / "deployment.yaml"
    yaml_file.write_text("demo: {}\n", encoding="utf-8")
    first = config_stamp(tmp_path)
    assert config_stamp(tmp_path) == first
    yaml_file.write_text("demo: {upload: {max_files: 10}}\n", encoding="utf-8")
    assert config_stamp(tmp_path) != first


def test_the_page_reloads_its_config_when_the_files_change(monkeypatch):
    """Same code, new config files: the cached config must be read again."""
    import streamlit as st

    from scoutfield.app import demo, streamlit_app

    reads = []
    monkeypatch.setattr(demo, "deployment_config",
                        lambda: reads.append(len(reads)) or {"read": len(reads)})
    st.cache_resource.clear()
    st.cache_data.clear()

    monkeypatch.setattr(demo, "config_stamp", lambda: "before")
    first = streamlit_app._resources(st)["config"]()
    assert streamlit_app._resources(st)["config"]() is first      # cached
    monkeypatch.setattr(demo, "config_stamp", lambda: "after")
    second = streamlit_app._resources(st)["config"]()
    assert second is not first and len(reads) == 2                # read again
    st.cache_resource.clear()


def test_a_missing_upload_section_is_named(tmp_path):
    from scoutfield.app.demo import UploadLimits
    from scoutfield.config import load_config

    path = tmp_path / "deployment.yaml"
    path.write_text("demo: {page_title: x}\n", encoding="utf-8")
    with pytest.raises(KeyError, match="demo.upload.*restarted"):
        UploadLimits.from_config(load_config(path))
