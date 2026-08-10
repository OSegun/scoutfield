"""
ONNX export and latency measurement.

Run:
    python -m scoutfield.app.export_onnx --checkpoint checkpoints/perception/best.pt

Why ONNX
--------
The deployment story for this work is an edge device on or near a drone, where
PyTorch is impractical. ONNX Runtime is small, CPU-friendly and has ARM builds.
Exporting also forces the model graph to be static, which surfaces any accidental
data-dependent control flow in the head.

The export trap that matters here
----------------------------------
``torch.onnx.export`` calls ``model.eval()``, which disables dropout. That is
correct for the deployed classifier but it silently destroys MC-dropout. If the
deployed app is meant to report epistemic uncertainty, dropout has to be exported
as an explicit graph node, or the uncertainty estimate has to come from an
exported ensemble instead. Decide which, and say which in the write-up — an app
that silently reports zero epistemic uncertainty looks confident and is not.

Verify numerical equivalence after export
------------------------------------------
Compare ONNX and PyTorch logits on the full test split, not on one batch. A
mismatch of even 1e-3 in logit space shifts probabilities enough to move ECE,
and calibration is the quantity this whole project measures.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

INPUT_NAME = "image"
OUTPUT_NAME = "logit"


def export(checkpoint: str, out_path: str, opset: int = 18, config=None,
           image_size: int = 224, verify_loader=None) -> dict:
    """Export to ONNX with a dynamic batch axis, then verify equivalence.

    ``torch.onnx.export`` puts the model in eval mode, which disables dropout.
    That is correct for the deployed *classifier* and fatal for MC-dropout: this
    export is the point-estimate path only. Epistemic uncertainty in the deployed
    app has to come from an exported ensemble instead, and the write-up says so
    rather than shipping an app that silently reports zero epistemic uncertainty.
    """
    import torch

    from scoutfield.config import load_config
    from scoutfield.perception.calibrate import _load_model_state
    from scoutfield.perception.model import build_model
    from scoutfield.utils.paths import repo_root

    if config is None:
        config = load_config(repo_root() / "configs" / "perception.yaml")
    image_size = int(config.section("data").get("image_size", image_size))

    model = build_model(config)
    model.load_state_dict(_load_model_state(checkpoint))
    model.eval()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        # `dynamic_shapes`, not the older `dynamic_axes`, which torch deprecates
        # under the dynamo exporter. Keyed by the forward argument name.
        dynamic_shapes={"x": {0: torch.export.Dim.AUTO}},
        opset_version=opset,
        # Silenced deliberately: the exporter's progress output contains emoji,
        # which raises UnicodeEncodeError on a Windows cp1252 console and takes
        # the whole export down with it after it has already succeeded.
        verbose=False,
    )

    report = verify_equivalence(model, out_path, verify_loader, image_size)
    report["path"] = str(out_path)
    report["opset"] = opset
    # ASCII only: a Windows cp1252 console cannot encode most of what this
    # sentence wants to say, and a crash in a log line would discard the export.
    print(f"exported {out_path} (max abs logit diff = {report['max_abs_diff']:.2e})")
    return report


def verify_equivalence(model, onnx_path, loader=None, image_size: int = 224,
                       atol: float = 1e-4) -> dict:
    """Compare ONNX and PyTorch logits over the test split, not one batch.

    A mismatch of even 1e-3 in logit space shifts probabilities enough to move
    ECE, and calibration is the quantity this project measures — so the check is
    on logits, and it is fatal rather than advisory.
    """
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    batches = []
    if loader is not None:
        for x, _, _ in loader:
            batches.append(x)
    else:
        # No data available (the usual case on a laptop with no dataset mounted).
        # Random input still exercises the whole graph numerically; it just cannot
        # speak to accuracy, and the report says which was used.
        batches = [torch.randn(8, 3, image_size, image_size) for _ in range(4)]

    max_diff, n = 0.0, 0
    with torch.no_grad():
        for x in batches:
            torch_logits = model(x).cpu().numpy().ravel()
            onnx_logits = session.run(
                [OUTPUT_NAME], {INPUT_NAME: x.cpu().numpy()}
            )[0].ravel()
            max_diff = max(max_diff, float(np.max(np.abs(torch_logits - onnx_logits))))
            n += len(torch_logits)

    if max_diff > atol:
        raise AssertionError(
            f"ONNX and PyTorch logits differ by {max_diff:.3e} > {atol:.0e} over {n} "
            "images. That is enough to move ECE, so the export is not usable for a "
            "calibration result."
        )
    return {"max_abs_diff": max_diff, "n_images": n, "atol": atol,
            "verified_on": "test split" if loader is not None else "random input"}


def measure_latency(onnx_path: str, n_runs: int = 200, warmup: int = 20,
                    image_size: int = 224, batch_size: int = 1) -> dict:
    """Measure per-image inference latency on this machine.

    Median and p95, not the mean: tail latency determines whether a flight
    controller misses its deadline, and the mean hides it. The hardware is
    recorded alongside — a latency figure without the machine it was measured on
    is not reproducible and does not belong in a paper.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32)

    for _ in range(warmup):
        session.run([OUTPUT_NAME], {INPUT_NAME: x})

    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run([OUTPUT_NAME], {INPUT_NAME: x})
        timings.append((time.perf_counter() - t0) * 1000.0)

    timings_ms = np.asarray(timings)
    return {
        "median_ms": float(np.median(timings_ms)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "mean_ms": float(timings_ms.mean()),
        "n_runs": n_runs,
        "batch_size": batch_size,
        "throughput_images_per_s": float(batch_size * 1000.0 / np.median(timings_ms)),
        "hardware": {
            "processor": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
            "intra_op_threads": ort.SessionOptions().intra_op_num_threads,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="checkpoints/perception/model.onnx")
    parser.add_argument("--config", default="configs/perception.yaml")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    from scoutfield.config import load_config
    from scoutfield.utils.paths import results_dir

    report = export(args.checkpoint, args.out, config=load_config(args.config))
    if not args.skip_latency:
        report["latency"] = measure_latency(args.out)
        lat = report["latency"]
        print(f"latency: median {lat['median_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms "
              f"on {lat['hardware']['processor']}")

    out = results_dir() / "export_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
