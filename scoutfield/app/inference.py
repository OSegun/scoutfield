"""
ONNX Runtime inference session.

Preprocessing must match training exactly — same resize, same interpolation, same
normalisation constants. A subtle mismatch here degrades accuracy by a few points
and calibration by considerably more, and it presents as a mysterious deployment
regression rather than as an obvious bug. Import the transform from
``perception.datasets`` rather than rewriting it; two copies will drift.

The fitted temperature is applied here too. A deployed model that reports raw
softmax confidence is exactly the miscalibrated system this project is about,
which would be an unfortunate thing to ship.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scoutfield.app.export_onnx import INPUT_NAME, OUTPUT_NAME


class ScoutPlanInference:
    """Loads an exported model and returns calibrated probabilities."""

    def __init__(self, onnx_path: str | Path, temperature: float = 1.0,
                 providers: list[str] | None = None, image_size: int = 224):
        import onnxruntime as ort

        self.temperature = float(temperature)
        self.image_size = int(image_size)
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"no exported model at {self.onnx_path}; run "
                "python -m scoutfield.app.export_onnx --checkpoint <ckpt>"
            )
        # CPU by default: the target is commodity hardware, and a GPU-only path
        # would not be the thing the latency numbers describe.
        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=providers or ["CPUExecutionProvider"]
        )

    @classmethod
    def from_results(cls, onnx_path: str | Path, results_json: str | Path | None = None,
                     **kwargs) -> ScoutPlanInference:
        """Build with the temperature that was actually fitted, read from disk.

        A deployed model reporting raw sigmoid confidence is precisely the
        miscalibrated system this project is about, so the fitted temperature is
        read from the calibration results rather than left to a default.
        """
        from scoutfield.utils.paths import results_dir

        path = Path(results_json) if results_json else results_dir() / "calibration_sweep.json"
        temperature = 1.0
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                temperature = float(json.load(fh).get("fitted_temperature", 1.0))
        else:
            print(f"warning: {path} not found; using T=1.0, which is uncalibrated")
        return cls(onnx_path, temperature=temperature, **kwargs)

    def preprocess(self, image) -> np.ndarray:
        """Resize and normalise exactly as evaluation did during training.

        The transform is imported from ``perception.datasets`` rather than
        rewritten: two copies drift, and a subtle mismatch here costs a few points
        of accuracy and considerably more calibration, presenting as a mysterious
        deployment regression rather than an obvious bug.
        """
        from scoutfield.perception.datasets import _transforms

        if isinstance(image, (str, Path)):
            from PIL import Image

            with Image.open(image) as im:
                image = im.convert("RGB")
        elif hasattr(image, "convert"):
            image = image.convert("RGB")
        else:
            raise TypeError(f"expected a path or a PIL image, got {type(image)}")

        tensor = _transforms(self.image_size, train=False)(image)
        return tensor.unsqueeze(0).numpy().astype(np.float32)

    def predict(self, image) -> dict:
        """Return ``{"probability": float, "prediction": int, "logit": float}``.

        The raw logit is included deliberately — it is what temperature scaling
        acts on, so exposing it lets the interface show what recalibration is
        actually doing rather than asking the user to take it on trust.
        """
        batch = self.preprocess(image) if not isinstance(image, np.ndarray) else image
        logit = float(self.session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0].ravel()[0])
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit / self.temperature, -60, 60)))
        return {
            "logit": logit,
            "probability": float(probability),
            # From the unscaled sign, so the label cannot move with temperature.
            "prediction": int(logit > 0),
            "uncalibrated_probability": float(
                1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))
            ),
            "temperature": self.temperature,
        }
