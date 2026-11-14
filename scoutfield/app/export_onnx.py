"""
ONNX export of the trained classifier, and its latency measurement.

Run (settings in ``configs/deployment.yaml``):
    python experiments/08_export_onnx.py              # checkpoint from the store
    python experiments/08_export_onnx.py --checkpoint <path/to/best.pt>

Why ONNX
--------
The deployment story for this work is an edge device on or near a drone, where
PyTorch is impractical. ONNX Runtime is small, CPU-friendly and has ARM builds.
Exporting also forces the model graph to be static, which surfaces any accidental
data-dependent control flow in the head.

The export trap that matters here
----------------------------------
``torch.onnx.export`` exports the model in eval mode, which disables dropout. That
is correct for the deployed classifier, and it means this export is the
point-estimate path only: MC-dropout cannot be run from it. Epistemic uncertainty
in a deployed app would have to come from an exported ensemble instead.

Verify numerical equivalence after export
------------------------------------------
Every export is compared against the PyTorch model on the same inputs: the hard
predictions must be identical and the logits must agree within the tolerance in
``configs/deployment.yaml``. A failure raises; an export that shifts logits would
shift the calibration this whole project measures.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

INPUT_NAME = "image"
OUTPUT_NAME = "logit"

DEFAULT_CONFIG = "configs/deployment.yaml"


# ------------------------------------------------------------------ config

def load_deployment_config(path: str | Path | None = None):
    """``configs/deployment.yaml``, resolved against the repository root."""
    from scoutfield.config import load_config
    from scoutfield.utils.paths import repo_root

    p = Path(path) if path else Path(DEFAULT_CONFIG)
    if not p.is_absolute():
        p = repo_root() / p
    return load_config(p)


def work_dir(config, run_id: str) -> Path:
    """Where this machine's exported files for ``run_id`` live (gitignored)."""
    from scoutfield.utils.paths import repo_root

    base = Path(config["work_dir"])
    if not base.is_absolute():
        base = repo_root() / base
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _shown(path: str | Path) -> str:
    """A path as recorded in a published file: relative to the repository when it is
    inside or beside it, so no record carries this machine's user folder."""
    import os.path

    from scoutfield.utils.paths import repo_root

    p = Path(path).resolve()
    try:
        return Path(os.path.relpath(p, repo_root())).as_posix()
    except ValueError:  # another drive on Windows
        return p.name


def resolve_checkpoint(config, store=None, override: str | Path | None = None
                       ) -> tuple[Path, str]:
    """The checkpoint to export, and a description of where it came from.

    By default the run's checkpoint is pulled from the artefact store, which also
    checks its SHA-256 against the run's manifest. ``override`` is for working
    offline with a local copy; a missing file raises rather than falling back.
    """
    export_cfg = config.section("export")
    if override is not None:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(
                f"no checkpoint at {path}. Pass the path of a local best.pt, or omit "
                f"--checkpoint to pull it from the artefact store."
            )
        return path, f"local file {_shown(path)}"

    from scoutfield.utils.artifacts import get_store

    store = store or get_store()
    stage, name = export_cfg["checkpoint_stage"], export_cfg["checkpoint_file"]
    path = store.pull(stage, name)
    return path, f"store {store.remote_path(stage, name)}"


# ------------------------------------------------------------------ export

def _model_from_checkpoint(checkpoint, perception_config):
    """The trained network, with no ImageNet weights downloaded first.

    The checkpoint replaces every weight, so building the backbone with its
    pretrained weights would only add a network download to an offline step.
    """
    from scoutfield.perception.calibrate import _load_model_state
    from scoutfield.perception.model import BinaryClassifier

    model_cfg = perception_config.section("model")
    model = BinaryClassifier(
        backbone=model_cfg.get("backbone", "efficientnet_b0"),
        pretrained=False,
        head_dropout=float(model_cfg.get("head_dropout", 0.3)),
    )
    model.load_state_dict(_load_model_state(checkpoint))
    model.eval()
    return model


def export(checkpoint: str | Path, out_path: str | Path, opset: int | None = None,
           config=None, verify: dict | None = None) -> dict:
    """Export to ONNX with a dynamic batch axis, then verify equivalence.

    ``config`` is the perception config (``configs/perception.yaml`` by default),
    which fixes the architecture and the image size. ``opset`` and ``verify``
    default to ``configs/deployment.yaml``.
    """
    import torch

    from scoutfield.config import load_config
    from scoutfield.utils.paths import repo_root

    if config is None:
        config = load_config(repo_root() / "configs" / "perception.yaml")
    deployment = load_deployment_config().section("export")
    opset = int(deployment["opset"] if opset is None else opset)
    verify = dict(deployment["verify"] if verify is None else verify)
    image_size = int(config.section("data")["image_size"])

    model = _model_from_checkpoint(checkpoint, config)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, image_size, image_size)

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        # `dynamic_shapes`, not the older `dynamic_axes`, which torch deprecates
        # under the dynamo exporter. Keyed by the forward argument name.
        dynamic_shapes={"x": {0: torch.export.Dim.AUTO}},
        opset_version=opset,
        # One self-contained file. By default the exporter moves the weights into a
        # side file (`<name>.onnx.data`), so the SHA-256 of the .onnx alone would not
        # cover them and a copy of one file without the other would fail to load.
        external_data=False,
        # Silenced deliberately: the exporter's progress output contains emoji,
        # which raises UnicodeEncodeError on a Windows cp1252 console and takes
        # the whole export down with it after it has already succeeded.
        verbose=False,
    )

    report = verify_equivalence(
        model, out_path, image_size=image_size,
        n_inputs=int(verify["n_inputs"]), batch_size=int(verify["batch_size"]),
        seed=int(verify["seed"]), atol=float(verify["logit_atol"]),
    )
    report["path"] = str(out_path)
    report["opset"] = opset
    report["image_size"] = image_size
    # ASCII only: a Windows cp1252 console cannot encode most of what this
    # sentence wants to say, and a crash in a log line would discard the export.
    print(f"exported {out_path} (max abs logit diff = {report['max_abs_diff']:.2e} "
          f"over {report['n_images']} inputs; predictions identical)")
    return report


def verify_equivalence(model, onnx_path, image_size: int, n_inputs: int,
                       batch_size: int, seed: int, atol: float) -> dict:
    """Compare ONNX and PyTorch on the same seeded inputs. Fatal on any mismatch.

    Two checks: the hard predictions (the sign of the logit) must be identical,
    and the logits must agree within ``atol``. The first batch is also run at
    batch size 1, so both a single image and a batch pass through the dynamic
    batch axis.
    """
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((n_inputs, 3, image_size, image_size)).astype(np.float32)
    batches = [inputs[:1]] + [inputs[i:i + batch_size]
                              for i in range(0, n_inputs, batch_size)]

    torch_all, onnx_all = [], []
    with torch.no_grad():
        for x in batches:
            torch_all.append(model(torch.from_numpy(x)).cpu().numpy().ravel())
            onnx_all.append(session.run([OUTPUT_NAME], {INPUT_NAME: x})[0].ravel())
    torch_logits = np.concatenate(torch_all)
    onnx_logits = np.concatenate(onnx_all)

    max_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    disagree = int(np.sum((torch_logits > 0) != (onnx_logits > 0)))
    if disagree or max_diff > atol:
        raise AssertionError(
            f"ONNX and PyTorch disagree over {torch_logits.size} inputs: "
            f"{disagree} hard prediction(s) differ, max |logit diff| = {max_diff:.3e} "
            f"(tolerance {atol:.0e}, configs/deployment.yaml export.verify.logit_atol). "
            "The export would change the calibration being measured, so it is rejected."
        )
    return {
        "max_abs_diff": max_diff,
        "n_images": int(torch_logits.size),
        "predictions_identical": True,
        "atol": atol,
        "verified_on": f"seeded random input (seed {seed}), batch sizes 1 and {batch_size}",
    }


def git_state() -> dict:
    """The commit running, and whether the working tree had uncommitted changes.

    An export from uncommitted code would otherwise be attributed to a commit that
    did not produce it.
    """
    from scoutfield.utils.artifacts import git_commit
    from scoutfield.utils.paths import repo_root

    commit = git_commit()
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                             cwd=repo_root(), capture_output=True, text=True, timeout=10,
                             check=False)
        dirty = bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def export_record(report: dict, onnx_path: Path, checkpoint: Path, source: str,
                  run_id: str) -> dict:
    """What a reader needs to trust the exported file: its inputs and its checks."""
    import onnx
    import onnxruntime as ort
    import torch

    from scoutfield.utils.artifacts import sha256_of

    size = int(report["image_size"])
    return {
        "run_id": run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"source": source, "sha256": sha256_of(checkpoint)},
        "onnx": {
            "file": onnx_path.name,
            "sha256": sha256_of(onnx_path),
            "bytes": onnx_path.stat().st_size,
            "opset": int(report["opset"]),
            "input": {"name": INPUT_NAME, "shape": ["batch", 3, size, size],
                      "dtype": "float32"},
            "output": {"name": OUTPUT_NAME, "shape": ["batch"], "dtype": "float32",
                       "meaning": "raw logit of disease; no sigmoid, no temperature"},
        },
        "verification": {k: report[k] for k in ("max_abs_diff", "n_images", "atol",
                                                "predictions_identical", "verified_on")},
        **git_state(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }


# ----------------------------------------------------------------- latency

def hardware_info() -> dict:
    """The machine a measurement was taken on, read from the system, not typed.

    CPU model name, logical and physical core counts, installed RAM and the
    operating system. Standard library only: the brand string comes from the
    Windows registry, ``/proc/cpuinfo`` on Linux and ``sysctl`` on macOS.
    """
    return {
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "physical_cores": _physical_cores(),
        "ram_gb": _ram_gb(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def _cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, check=False)
        if out.stdout.strip():
            return out.stdout.strip()
    return platform.processor() or platform.machine() or "unknown"


def _physical_cores() -> int | None:
    if sys.platform == "win32":
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"],
            capture_output=True, text=True, check=False, timeout=30)
        return int(out.stdout.strip()) if out.stdout.strip().isdigit() else None
    if sys.platform.startswith("linux"):
        try:
            pairs = set()
            physical = core = None
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("physical id"):
                    physical = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                    pairs.add((physical, core))
            return len(pairs) or None
        except OSError:
            return None
    return None


def _ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / 2**30, 1)
        return None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _on_mains_power() -> bool | None:
    """Whether a laptop was plugged in. On battery, Windows throttles the CPU, and a
    latency measured that way is slower than the same machine plugged in."""
    if sys.platform == "win32":
        import ctypes

        class _PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte),
                        ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_ulong),
                        ("BatteryFullLifeTime", ctypes.c_ulong)]

        status = _PowerStatus()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return {0: False, 1: True}.get(status.ACLineStatus)
        return None
    supply = Path("/sys/class/power_supply")
    if supply.exists():
        for online in supply.glob("A*/online"):
            return online.read_text(encoding="utf-8").strip() == "1"
    return None


def measure_latency(onnx_path: str | Path, n_runs: int = 200, warmup: int = 20,
                    image_size: int = 224, batch_size: int = 1, seed: int = 0,
                    intra_op_threads: int = 0) -> dict:
    """Measure per-call inference latency on this machine.

    Median and p95, not the mean alone: tail latency determines whether a flight
    controller misses its deadline, and the mean hides it. The hardware is
    recorded alongside — a latency figure without the machine it was measured on
    is not reproducible and does not belong in a report. ``intra_op_threads`` = 0
    leaves ONNX Runtime's default thread count.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(intra_op_threads)
    session = ort.InferenceSession(str(onnx_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    x = np.random.default_rng(seed).standard_normal(
        (batch_size, 3, image_size, image_size)).astype(np.float32)

    for _ in range(warmup):
        session.run([OUTPUT_NAME], {INPUT_NAME: x})

    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run([OUTPUT_NAME], {INPUT_NAME: x})
        timings.append((time.perf_counter() - t0) * 1000.0)

    timings_ms = np.asarray(timings)
    hardware = hardware_info()
    hardware.update({
        "onnxruntime": ort.__version__,
        "providers": session.get_providers(),
        "on_mains_power": _on_mains_power(),
        # Kept for compatibility with earlier readers of this dict.
        "processor": hardware["cpu_model"],
        "platform": hardware["os"],
    })
    return {
        "median_ms": float(np.median(timings_ms)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "mean_ms": float(timings_ms.mean()),
        "min_ms": float(timings_ms.min()),
        "max_ms": float(timings_ms.max()),
        "n_runs": int(n_runs),
        "warmup_runs": int(warmup),
        "batch_size": int(batch_size),
        "intra_op_threads": int(intra_op_threads),
        "throughput_images_per_s": float(batch_size * 1000.0 / np.median(timings_ms)),
        "hardware": hardware,
    }


# ------------------------------------------------------------------- driver

def run_export(config_path: str | Path | None = None, checkpoint: str | Path | None = None,
               push: bool = False, store=None) -> dict:
    """Export the run's classifier into the local work directory; optionally push.

    Pushing writes ``classifier.onnx`` and the export record to stage
    ``08_deployment`` of the store. It is refused unless the exported checkpoint
    is byte-identical to the one the run recorded, so a pushed model always traces
    to the run it is filed under.
    """
    from scoutfield.utils.artifacts import get_store, sha256_of

    config = load_deployment_config(config_path)
    export_cfg = config.section("export")
    store = store or get_store()
    ckpt, source = resolve_checkpoint(config, store=store, override=checkpoint)

    out_dir = work_dir(config, store.run_id)
    onnx_path = out_dir / export_cfg["classifier_file"]
    report = export(ckpt, onnx_path)
    record = export_record(report, onnx_path, Path(ckpt), source, store.run_id)
    record_path = out_dir / export_cfg["record_file"]
    with record_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"wrote {record_path}")

    if push:
        stage = str(config["stage"])
        recorded = store.stage_entry(export_cfg["checkpoint_stage"]).get("files", {}) \
            .get(export_cfg["checkpoint_file"], {}).get("sha256")
        if recorded != sha256_of(ckpt):
            raise ValueError(
                f"refusing to push: the exported checkpoint ({source}) is not the "
                f"{export_cfg['checkpoint_file']} recorded for run {store.run_id}."
            )
        if record["git_dirty"]:
            raise ValueError("refusing to push an export made from uncommitted code; "
                             "commit first so the record names the code that made it.")
        store.push(stage, [onnx_path, record_path], "classifier exported to ONNX")
        store.record_inputs(stage, [(export_cfg["checkpoint_stage"],
                                     export_cfg["checkpoint_file"])])
        print(f"pushed {onnx_path.name} and {record_path.name} to {stage} "
              f"of run {store.run_id}")
    return record


def locate_classifier(config, store) -> tuple[Path, str]:
    """The exported classifier: this machine's export first, then the store's.

    A missing file raises and names both places it looked and the command that
    makes it.
    """
    export_cfg = config.section("export")
    name = export_cfg["classifier_file"]
    local = work_dir(config, store.run_id) / name
    if local.is_file():
        return local, f"local export {_shown(local)}"
    stage = str(config["stage"])
    pulled = store.pull(stage, name, required=False)
    if pulled is not None:
        return pulled, f"store {store.remote_path(stage, name)}"
    raise FileNotFoundError(
        f"no exported classifier for run {store.run_id}: looked for {local} and for "
        f"{store.remote_path(stage, name)} in {store.backend.description}. Run "
        f"`python experiments/08_export_onnx.py` first."
    )


def run_latency(config_path: str | Path | None = None, push: bool = False,
                store=None) -> dict:
    """Measure the exported classifier's latency at every configured thread count.

    Writes ``deployment_latency.json`` beside the export. ``push`` files it in
    stage ``08_deployment``, refused unless the measured file is the
    ``classifier.onnx`` already filed there and the working tree is committed, so
    a pushed latency always describes the pushed model.
    """
    import onnxruntime as ort

    from scoutfield.config import load_config
    from scoutfield.utils.artifacts import get_store, sha256_of
    from scoutfield.utils.paths import repo_root

    config = load_deployment_config(config_path)
    lat = config.section("latency")
    store = store or get_store()
    onnx_path, source = locate_classifier(config, store)
    image_size = int(load_config(repo_root() / "configs" / "perception.yaml")
                     .section("data")["image_size"])

    measurements, hardware = [], None
    for threads in lat["intra_op_threads"]:
        repeats = []
        for _ in range(int(lat["repeats"])):
            m = measure_latency(onnx_path, n_runs=int(lat["timed_runs"]),
                                warmup=int(lat["warmup_runs"]), image_size=image_size,
                                batch_size=int(lat["batch_size"]), seed=int(lat["seed"]),
                                intra_op_threads=int(threads))
            hardware = hardware or m["hardware"]
            m.pop("hardware")
            repeats.append(m)
        medians = [r["median_ms"] for r in repeats]
        p95s = [r["p95_ms"] for r in repeats]
        summary = {
            "intra_op_threads": int(threads),
            # The middle repeat's figures, and the range across repeats.
            "median_ms": float(np.median(medians)),
            "median_ms_range": [float(min(medians)), float(max(medians))],
            "p95_ms": float(np.median(p95s)),
            "p95_ms_range": [float(min(p95s)), float(max(p95s))],
            "throughput_images_per_s": float(int(lat["batch_size"]) * 1000.0
                                             / np.median(medians)),
            "repeats": repeats,
        }
        measurements.append(summary)
        print(f"threads {threads} ({'default' if int(threads) == 0 else 'fixed'}): "
              f"median {summary['median_ms']:.2f} ms "
              f"(repeats {min(medians):.2f}-{max(medians):.2f}), "
              f"p95 {summary['p95_ms']:.2f} ms (repeats {min(p95s):.2f}-{max(p95s):.2f})")

    record = {
        "run_id": store.run_id,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "what": "ONNX Runtime CPU inference of the exported classifier, model call only "
                "(no image decoding or resizing), on the laptop named under hardware; "
                "not drone hardware",
        "onnx": {"file": onnx_path.name, "sha256": sha256_of(onnx_path), "source": source},
        "settings": {"execution_provider": "CPUExecutionProvider",
                     "batch_size": int(lat["batch_size"]),
                     "warmup_runs": int(lat["warmup_runs"]),
                     "timed_runs": int(lat["timed_runs"]),
                     "repeats": int(lat["repeats"]),
                     "input_shape": [int(lat["batch_size"]), 3, image_size, image_size],
                     "intra_op_threads": [int(t) for t in lat["intra_op_threads"]]},
        "hardware": hardware,
        "measurements": measurements,
        **git_state(),
        "versions": {"python": platform.python_version(), "onnxruntime": ort.__version__,
                     "numpy": np.__version__},
    }
    out = work_dir(config, store.run_id) / lat["output_file"]
    with out.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"wrote {out}")

    if push:
        stage = str(config["stage"])
        filed = store.stage_entry(stage).get("files", {}) \
            .get(config.section("export")["classifier_file"], {}).get("sha256")
        if filed != record["onnx"]["sha256"]:
            raise ValueError(
                f"refusing to push: the measured model is not the classifier.onnx filed "
                f"in {stage} of run {store.run_id}. Push the export first "
                f"(`python experiments/08_export_onnx.py --push`)."
            )
        if record["git_dirty"]:
            raise ValueError("refusing to push a measurement made from uncommitted code; "
                             "commit first so the record names the code that made it.")
        store.push(stage, [out], "latency of the exported classifier")
        print(f"pushed {out.name} to {stage} of run {store.run_id}")
    return record
