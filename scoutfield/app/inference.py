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


class ScoutFieldInference:
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
                "python experiments/08_export_onnx.py"
            )
        # CPU by default: the target is commodity hardware, and a GPU-only path
        # would not be the thing the latency numbers describe.
        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=providers or ["CPUExecutionProvider"]
        )

    @classmethod
    def from_results(cls, onnx_path: str | Path, results_json: str | Path | None = None,
                     condition: str = "shift_fitted", store=None,
                     **kwargs) -> ScoutFieldInference:
        """Build with the temperature that was actually fitted.

        A deployed model reporting raw sigmoid confidence is precisely the
        miscalibrated system this project is about, so the fitted temperature is
        read from the calibration results rather than left to a default.

        ``results_json`` is the offline override: a local copy of
        ``calibration_sweep.json``. Without it the file is read from stage 02 of the
        artefact store. ``condition`` picks the temperature: ``shift_fitted`` (field
        imagery, the default for a deployed scout) or ``lab_fitted``. A missing file
        or key raises; there is no uncalibrated fallback.
        """
        from scoutfield.envs.field_env import REFERENCE_CONDITIONS

        if condition not in REFERENCE_CONDITIONS:
            raise KeyError(f"unknown condition {condition!r}; have {sorted(REFERENCE_CONDITIONS)}")
        if results_json is not None:
            path = Path(results_json)
            if not path.exists():
                raise FileNotFoundError(f"no calibration results at {path}")
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            from scoutfield.utils.artifacts import get_store

            data = (store or get_store()).pull_json("02_calibration",
                                                    "calibration_sweep.json")
        key = REFERENCE_CONDITIONS[condition]
        if key not in data:
            raise KeyError(f"calibration results have no {key!r}; re-run stage 02")
        return cls(onnx_path, temperature=float(data[key]), **kwargs)

    def preprocess(self, image) -> np.ndarray:
        """Resize and normalise exactly as evaluation did during training.

        Uses ``perception.datasets.eval_array``, the torch-free twin of the training
        code's evaluation transform, which a test holds equal to it: a subtle
        mismatch here costs a few points of accuracy and considerably more
        calibration, presenting as a mysterious deployment regression rather than
        an obvious bug.
        """
        from scoutfield.perception.datasets import eval_array

        if isinstance(image, (str, Path)):
            from PIL import Image

            with Image.open(image) as im:
                image = im.convert("RGB")
        elif hasattr(image, "convert"):
            # convert() copies even an RGB image; skipping it saves one full-size
            # copy of a large photograph and changes no pixel.
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            raise TypeError(f"expected a path or a PIL image, got {type(image)}")

        return eval_array(image, self.image_size)[None]

    def predict(self, image, relative_temperature: float = 1.0) -> dict:
        """Return the probability, the prediction and the raw logit for one image.

        ``relative_temperature`` is the experiment's T: it multiplies the fitted
        temperature, exactly as ``CNNClassifier.observe`` divides by
        ``reference_temperature * T``. So T = 1 gives the calibrated probability,
        T < 1 an overconfident one and T > 1 an underconfident one.

        The raw logit is included deliberately — it is what temperature scaling
        acts on, so exposing it lets the interface show what recalibration is
        actually doing rather than asking the user to take it on trust.
        """
        batch = self.preprocess(image) if not isinstance(image, np.ndarray) else image
        logit = float(self.session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0].ravel()[0])
        effective = self.temperature * float(relative_temperature)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit / effective, -60, 60)))
        return {
            "logit": logit,
            "probability": float(probability),
            # From the unscaled sign, so the label cannot move with temperature.
            "prediction": int(logit > 0),
            "uncalibrated_probability": float(
                1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))
            ),
            # The fitted (reference) temperature, the T applied on top of it, and
            # the divisor actually used.
            "temperature": self.temperature,
            "relative_temperature": float(relative_temperature),
            "effective_temperature": effective,
        }
