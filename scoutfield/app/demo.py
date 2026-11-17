"""
The demonstration's logic, kept apart from the Streamlit page so it can be tested.

The demonstration must show exactly what the experiment measured, so every
condition below is read from the same config files and the same store files the
sweep used, never typed in here:

* the planners observe through the PlantDoc ``shift_eval`` logit pool
  (``pool_split`` in ``configs/sweep.yaml``), read from stage ``02_calibration``;
* the temperature a viewer chooses is *relative to* the fitted temperature of
  ``reference_condition`` in ``configs/sweep.yaml`` (the shift-fitted one), so
  T = 1 is the calibrated point, as in the sweep;
* the field, the energy model and τ come from ``configs/ppo_field32.yaml``.

The classifier is built the way ``experiments/04_sweep.py`` builds it: the pool
from ``load_or_build_logit_pool``, the reference temperature from
``_reference_temperature``, a generator seeded with the evaluation seed, and the
field seeded ``seed * 10_000 + episode``. Nothing here imports torch.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Planner names as the sweep spells them (``agents`` in configs/sweep.yaml).
BASELINES = ("lawnmower", "greedy_entropy")

#: Episode index used for the field seed, ``seed * 10_000 + episode``: the first
#: episode of each evaluation seed in the sweep.
EPISODE = 0


@dataclass(frozen=True)
class DemoSettings:
    """The experimental conditions the demonstration reproduces."""

    pool_split: str
    reference_condition: str
    temperatures: tuple[float, ...]
    tau: float
    sigma: float
    env_config: object


def load_settings(repo: Path | None = None) -> DemoSettings:
    """Read the conditions from ``configs/sweep.yaml`` and ``configs/ppo_field32.yaml``."""
    from scoutfield.config import load_config
    from scoutfield.utils.paths import repo_root

    root = repo or repo_root()
    sweep = load_config(root / "configs" / "sweep.yaml")
    env_config = load_config(root / "configs" / "ppo_field32.yaml")
    env = env_config.section("env")
    return DemoSettings(
        pool_split=str(sweep["pool_split"]),
        reference_condition=str(sweep["reference_condition"]),
        temperatures=tuple(float(t) for t in sweep["temperatures"]),
        tau=float(env["detect_threshold"]),
        sigma=float(env["cluster_sigma"]),
        env_config=env_config,
    )


def fitted_temperature(condition: str, store=None) -> float:
    """The fitted temperature of ``condition``, from stage 02 (the sweep's reader)."""
    from scoutfield.envs.field_env import _reference_temperature

    return _reference_temperature(condition, store=store)


def load_pool(pool_split: str, store=None) -> dict[int, np.ndarray]:
    """The stage 02 logit pool, verified against the recorded split (the sweep's reader)."""
    from scoutfield.perception.adapter import load_or_build_logit_pool

    return load_or_build_logit_pool(pool_split, store=store)


def make_planner(name: str, seed: int):
    """A baseline planner from the pilot, seeded as the sweep seeds it."""
    from agents import GreedyEntropyAgent, LawnmowerAgent

    rng = np.random.default_rng(seed)
    if name == "lawnmower":
        return LawnmowerAgent(rng)
    if name == "greedy_entropy":
        return GreedyEntropyAgent(rng)
    raise KeyError(f"unknown planner {name!r}; have {', '.join(BASELINES)}")


def run_episode(planner, settings: DemoSettings, pool: dict[int, np.ndarray],
                reference_temperature: float, temperature: float, seed: int,
                tau: float | None = None) -> dict:
    """One episode, built exactly as one sweep episode is built.

    Returns the final ``info`` dict, the flown path, the environment (for the
    belief map and the true field) and the classifier it observed through.
    """
    from scoutfield.envs.field_env import make_field_env
    from scoutfield.perception.adapter import CNNClassifier

    classifier = CNNClassifier(
        logit_pool=pool,
        temperature=temperature,
        reference_temperature=reference_temperature,
        rng=np.random.default_rng(seed),
        pool_split=settings.pool_split,
    )
    env = make_field_env(
        settings.env_config,
        seed=seed * 10_000 + EPISODE,
        temperature=temperature,
        tau=settings.tau if tau is None else float(tau),
        sigma=settings.sigma,
        classifier=classifier,
    )
    obs = env.reset()
    if hasattr(planner, "reset"):
        planner.reset(env)
    path = [env.pos]
    done, info = False, {}
    while not done:
        obs, _, done, info = env.step(int(planner.act(obs, env)))
        path.append(env.pos)
    return {"info": info, "path": np.asarray(path), "env": env, "classifier": classifier}


# ------------------------------------------------------------ where files live

#: Overrides ``demo.source`` in configs/deployment.yaml.
SOURCE_ENV = "SCOUTFIELD_DEMO_SOURCE"
SOURCES = ("store", "offline")
FILL_COMMAND = "python experiments/08_fill_demo_folder.py"


def deployment_config():
    from scoutfield.app.export_onnx import load_deployment_config

    return load_deployment_config()


def config_stamp(repo: Path | None = None) -> str:
    """SHA-256 over every file in ``configs/``, names and contents, in name order.

    The page's caches take this as an argument, so that a config file changed by a
    redeploy is a new cache key rather than a setting the running app never sees.
    """
    import hashlib

    from scoutfield.utils.paths import repo_root

    digest = hashlib.sha256()
    for path in sorted(((repo or repo_root()) / "configs").glob("*.yaml")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def demo_source(config=None) -> str:
    """``store`` or ``offline``: the environment variable, else the config value."""
    import os

    config = config or deployment_config()
    source = os.environ.get(SOURCE_ENV) or str(config.section("demo")["source"])
    if source not in SOURCES:
        raise ValueError(f"{SOURCE_ENV}={source!r}; use one of {', '.join(SOURCES)}")
    return source


def offline_root(config=None) -> Path:
    from scoutfield.utils.paths import repo_root

    config = config or deployment_config()
    root = Path(config.section("demo")["offline_dir"])
    return root if root.is_absolute() else repo_root() / root


def offline_store(config=None, run_id: str | None = None):
    """The offline folder as an artefact store: same paths and manifest checks."""
    from scoutfield.utils.artifacts import ArtifactStore, LocalBackend, load_settings

    root = offline_root(config)
    run = run_id or load_settings()["run_id"]
    manifest = root / "store" / "runs" / run / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"the offline folder has no files for run {run}: looked for {manifest}. "
            f"Fill it once while online with `{FILL_COMMAND}`."
        )
    return ArtifactStore(LocalBackend(root / "store"), run, root / "cache")


def open_store(config=None):
    """The store the app reads from, chosen by the ``demo.source`` setting.

    One code path for every tier: the hosted tiers read the public store with no
    token; the laptop tier reads the offline folder, which has the same layout and
    is checked against the same manifest entries.
    """
    from scoutfield.utils.artifacts import get_store

    config = config or deployment_config()
    return offline_store(config) if demo_source(config) == "offline" else get_store()


def required_files(config=None, settings: DemoSettings | None = None
                   ) -> list[tuple[str, str]]:
    """Every ``(stage, name)`` the app reads from the store, apart from samples."""
    from scoutfield.perception.adapter import logit_pool_filename

    config = config or deployment_config()
    settings = settings or load_settings()
    stage = str(config["stage"])
    summary = config.section("demo")["summary"]
    return [
        ("02_calibration", "calibration_sweep.json"),
        ("02_calibration", logit_pool_filename(settings.pool_split)),
        (str(summary["stage"]), str(summary["name"])),
        (stage, str(config.section("export")["classifier_file"])),
        (stage, str(config.section("policy")["policy_file"])),
    ]


def sample_files(store, config=None) -> list[tuple[str, str]]:
    """The sample images filed in the store, with their list, if any were filed."""
    config = config or deployment_config()
    stage, name = str(config["stage"]), str(config.section("demo")["samples"]["file"])
    if not store.has(stage, name):
        return []
    listed = store.pull_json(stage, name)
    return [(stage, name)] + [(stage, s["store_name"]) for s in listed["samples"]]


def fill_offline(source_store, config=None, local_samples: Path | None = None) -> dict:
    """Copy every file the app reads from ``source_store`` into the offline folder.

    Each file is pulled through the store (so its SHA-256 is checked against the
    run's manifest) and written at the same path under ``<offline_dir>/store``,
    with a manifest holding the same entries, so the offline app verifies every
    read exactly as the online one does. ``local_samples`` is a folder of
    laptop-only samples (with their list), copied to ``<offline_dir>/local_samples``.
    """
    import json
    import shutil

    config = config or deployment_config()
    root = offline_root(config)
    target = root / "store" / "runs" / source_store.run_id
    if target.exists():
        shutil.rmtree(target)
    wanted = required_files(config) + sample_files(source_store, config)
    full = source_store.manifest()
    subset = {"run_id": source_store.run_id, "filled_from": source_store.backend.description,
              "stages": {}}
    for stage, name in wanted:
        local = source_store.pull(stage, name)
        dest = target / stage / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
        entry = full["stages"][stage]["files"][name]
        subset["stages"].setdefault(stage, {"files": {}})["files"][name] = entry
    (target / "manifest.json").write_text(json.dumps(subset, indent=2), encoding="utf-8")

    local_dir = root / "local_samples"
    if local_dir.exists():
        shutil.rmtree(local_dir)
    copied = 0
    if local_samples is not None and Path(local_samples).is_dir():
        shutil.copytree(local_samples, local_dir)
        copied = sum(1 for p in local_dir.iterdir() if p.suffix.lower() != ".json")
    return {"root": root, "files": wanted, "local_samples": copied}


def local_samples(config=None) -> list[dict]:
    """Laptop-only samples in the offline folder, each with its ``path``."""
    import json

    config = config or deployment_config()
    folder = offline_root(config) / "local_samples"
    listing = folder / str(config.section("demo")["samples"]["file"])
    if not listing.is_file():
        return []
    samples = json.loads(listing.read_text(encoding="utf-8"))["samples"]
    return [{**s, "path": folder / s["store_name"]} for s in samples]


# ------------------------------------------------------------------ planners

PLANNERS = ("lawnmower", "greedy_entropy", "ppo")
LABELS = {"lawnmower": "Lawnmower", "greedy_entropy": "GreedyEntropy", "ppo": "PPO"}


def load_policy(store, config=None):
    """The exported PPO policy, run by onnxruntime (no torch)."""
    from scoutfield.app.export_policy import ONNXPolicy

    config = config or deployment_config()
    return ONNXPolicy(store.pull(str(config["stage"]),
                                 str(config.section("policy")["policy_file"])))


def get_planner(name: str, seed: int, policy=None):
    """Any demo planner by name; PPO needs the loaded ``policy``."""
    if name == "ppo":
        if policy is None:
            raise ValueError("the PPO planner needs the exported policy (load_policy)")
        return policy
    return make_planner(name, seed)


def _sweep_config(repo: Path | None = None):
    from scoutfield.config import load_config
    from scoutfield.utils.paths import repo_root

    return load_config((repo or repo_root()) / "configs" / "sweep.yaml")


def tau_grid(repo: Path | None = None) -> tuple[float, ...]:
    """The τ values of the τ sweep for the baselines (configs/sweep.yaml)."""
    return tuple(float(t) for t in _sweep_config(repo).section("tau_sensitivity")["tau_grid"])


def ppo_tau_grid(repo: Path | None = None) -> tuple[float, ...]:
    """The τ values at which the τ sweep evaluated PPO."""
    cfg = _sweep_config(repo).section("tau_sensitivity")
    return tuple(float(t) for t in cfg["ppo_tau_grid"])


# -------------------------------------------------------------- seed choice

def choose_seed(summary: dict, settings: DemoSettings, pool, reference_temperature: float,
                policy, candidates: int, temperatures=(1.0, 4.0)) -> dict:
    """The evaluation seed whose episodes look most like the sweep's typical result.

    For each candidate seed, one episode per planner at each temperature (τ and σ
    as configured) is compared with the sweep's IQM for that (planner, T) cell, as
    ``|episode - IQM| / IQM``. The seed with the smallest mean over the cells wins.
    """
    rows = []
    for seed in range(int(candidates)):
        errors, values = [], {}
        for name in PLANNERS:
            for t in temperatures:
                key = f"{name}@T{t:g}"
                cell = summary["cells"][key]["detections_per_joule"]["value"]
                run = run_episode(get_planner(name, seed, policy), settings, pool,
                                  reference_temperature, t, seed)
                got = float(run["info"]["detections_per_joule"])
                values[key] = got
                errors.append(abs(got - cell) / cell)
        rows.append({"seed": seed, "mean_relative_difference": float(np.mean(errors)),
                     "detections_per_joule": values})
    best = min(rows, key=lambda r: r["mean_relative_difference"])
    return {"seed": best["seed"], "rows": rows}


# ------------------------------------------------------- a visitor's own images
#
# The "Analyse your leaves" panel. Every image goes through the deployed path
# unchanged (``ScoutFieldInference.preprocess`` and ``.predict``), and nothing a
# visitor uploads is written anywhere: files are read from bytes in memory, and the
# report is returned as text for the page to offer as a download.

#: Stated on the page and in the report, so the two cannot say different things.
LIMITS_TEXT = (
    ("Two classes only: healthy or diseased. The classifier does not name a disease "
     "or judge its severity."),
    ("Trained on PlantVillage photographs of single leaves taken in a laboratory, and "
     "calibrated on held-out PlantDoc photographs taken in fields. Images unlike both "
     "may be classified poorly, and the stated probability may not hold for them."),
    "Not a diagnosis. Confirm any finding with an agronomist or a plant clinic.",
    ("Any image gets a prediction, including one that shows no leaf at all. The "
     "classifier cannot tell that an image is not a leaf."),
)
PRIVACY_TEXT = ("Uploaded images are processed in memory and discarded. They are not "
                "saved to disk, sent to the artefact store, or included in the report.")


class UploadRejected(ValueError):
    """An uploaded file the panel will not analyse, with the reason."""


@dataclass(frozen=True)
class UploadLimits:
    max_files: int
    max_file_mb: float
    max_megapixels: float
    thumbnail_px: int

    @classmethod
    def from_config(cls, config=None) -> UploadLimits:
        config = config or deployment_config()
        demo = config.section("demo")
        if "upload" not in demo:
            raise KeyError(
                f"{config.path} has no demo.upload section. The leaf-upload panel reads "
                "its limits from there; a running app that loaded an older copy of the "
                "file must be restarted to read the current one.")
        up = demo["upload"]
        return cls(int(up["max_files"]), float(up["max_file_mb"]),
                   float(up["max_megapixels"]), int(up["thumbnail_px"]))


def open_upload(name: str, data, limits: UploadLimits):
    """Decode one uploaded file from memory, refusing it if it breaks a limit.

    ``data`` is the file's bytes or an in-memory binary file (Streamlit's uploaded
    file is one, and reading it directly avoids a second copy of its bytes). The file
    size and the pixel count (read from the header, before decoding) are checked
    first, so an oversized image is refused without being decoded.
    """
    import io

    from PIL import Image, UnidentifiedImageError

    stream = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
    stream.seek(0, io.SEEK_END)
    size = stream.tell()
    stream.seek(0)
    if size > limits.max_file_mb * 1024 * 1024:
        raise UploadRejected(f"{name}: {size / 1024 / 1024:.1f} MB is over the "
                             f"{limits.max_file_mb:g} MB limit")
    try:
        image = Image.open(stream)
    except (UnidentifiedImageError, OSError) as exc:
        raise UploadRejected(f"{name}: not a readable JPG or PNG image") from exc
    if image.format not in ("JPEG", "PNG"):
        raise UploadRejected(f"{name}: {image.format} is not accepted; use JPG or PNG")
    width, height = image.size
    if width * height > limits.max_megapixels * 1e6:
        raise UploadRejected(f"{name}: {width} x {height} pixels is over the "
                             f"{limits.max_megapixels:g} megapixel limit")
    try:
        image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise UploadRejected(f"{name}: the image data could not be decoded") from exc
    return image


def _thumbnail(image, size: int):
    """A small RGB copy for display, turned upright by its EXIF orientation.

    Display only: phone photographs record their rotation in EXIF, and the
    classifier sees the pixels exactly as the deployed path reads them. The image is
    shrunk before it is turned, so no second full-size copy is made.
    """
    from PIL import Image, ImageOps

    turn = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
            4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
            6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
            8: Image.Transpose.ROTATE_90}.get(image.getexif().get(0x0112))
    small = ImageOps.contain(image, (size, size)).convert("RGB")
    return small.transpose(turn) if turn is not None else small


def is_uncertain(probability: float, tau: float) -> bool:
    """True when neither class reaches the planners' confirmation threshold τ.

    A planner confirms a cell as diseased when its belief reaches τ; by symmetry, a
    probability at or below 1 - τ is a confident "healthy". Between the two, the
    calibrated classifier is not sure enough for either.
    """
    return (1.0 - tau) < probability < tau


#: One full-size decoded image at a time in this process, whichever visitor sent
#: it: hosted, every visitor shares one process and one memory limit.
_DECODE_LOCK = threading.Lock()


def analyse_uploads(model, uploads, temperatures, tau: float, limits: UploadLimits) -> dict:
    """Run a batch of ``(name, data)`` uploads through the deployed classifier.

    ``data`` is bytes or an in-memory binary file. Each image is decoded,
    preprocessed once, and scored at T = 1 (calibrated) and at the lowest and
    highest temperatures of the sweep. Only a small thumbnail is kept; the decoded
    image is released before the next one is opened. Returns the rows, the refused
    files with their reasons, and the batch summary.
    """
    low, high = min(temperatures), max(temperatures)
    rows, rejected = [], []
    for index, (name, data) in enumerate(uploads):
        if index >= limits.max_files:
            rejected.append(f"{name}: over the limit of {limits.max_files} images per "
                            "batch; analyse it in another batch")
            continue
        with _DECODE_LOCK:
            try:
                image = open_upload(name, data, limits)
            except UploadRejected as exc:
                rejected.append(str(exc))
                continue
            batch = model.preprocess(image)
            thumbnail = _thumbnail(image, limits.thumbnail_px)
            image.close()
            del image
        at_one = model.predict(batch, relative_temperature=1.0)
        p = at_one["probability"]
        rows.append({
            "file": name,
            "prediction": "diseased" if at_one["prediction"] else "healthy",
            "probability_calibrated": p,
            "probability_low_T": model.predict(batch, relative_temperature=low)["probability"],
            "probability_high_T": model.predict(batch, relative_temperature=high)["probability"],
            "logit": at_one["logit"],
            "uncertain": is_uncertain(p, tau),
            "thumbnail": thumbnail,
        })
    return {"rows": rows, "rejected": rejected,
            "summary": batch_summary(rows, tau), "low_T": low, "high_T": high}


def batch_summary(rows: list[dict], tau: float) -> dict:
    """Counts by prediction, the uncertain share, and one plain sentence."""
    n = len(rows)
    diseased = sum(r["prediction"] == "diseased" for r in rows)
    uncertain = sum(r["uncertain"] for r in rows)
    share = uncertain / n if n else 0.0
    sentence = (
        f"Of {n} image{'s' if n != 1 else ''}, {diseased} "
        f"{'was' if diseased == 1 else 'were'} predicted diseased and {n - diseased} "
        f"healthy; {uncertain} ({share:.0%}) "
        f"{'was' if uncertain == 1 else 'were'} uncertain, with a calibrated probability "
        f"between {1 - tau:g} and {tau:g}. The classifier can say whether a photograph "
        "looks more like the diseased or the healthy leaves it learned from, and how sure "
        "it is once calibrated on field photographs. It cannot say which disease, how "
        "severe, or whether the photograph shows a leaf at all."
    ) if n else "No image was analysed."
    return {"n": n, "diseased": diseased, "healthy": n - diseased,
            "uncertain": uncertain, "uncertain_share": share, "sentence": sentence}


REPORT_COLUMNS = ("file", "prediction", "probability_calibrated", "probability_low_T",
                  "probability_high_T", "logit", "uncertain")


def report_csv(result: dict) -> str:
    """The batch as CSV: one row per image, numbers only, never the image."""
    import csv
    import io

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    low, high = result["low_T"], result["high_T"]
    writer.writerow(["file", "prediction", "probability_T1",
                     f"probability_T{low:g}", f"probability_T{high:g}", "logit",
                     "uncertain"])
    for r in result["rows"]:
        writer.writerow([r["file"], r["prediction"],
                         f"{r['probability_calibrated']:.4f}",
                         f"{r['probability_low_T']:.4f}", f"{r['probability_high_T']:.4f}",
                         f"{r['logit']:.4f}", "yes" if r["uncertain"] else "no"])
    return out.getvalue()


def report_markdown(result: dict, context: dict) -> str:
    """A short summary: settings, counts, the table, the limits. Never the images.

    ``context`` names the run, the reference condition, its fitted temperature and
    τ, so a reader can trace every number to the run that produced the model.
    """
    s = result["summary"]
    low, high = result["low_T"], result["high_T"]
    lines = [
        "# ScoutField leaf analysis",
        "",
        (f"Classifier: EfficientNet-B0 exported to ONNX, run `{context['run_id']}`. "
         "Probabilities are calibrated with the temperature fitted on held-out field "
         f"photographs ({context['reference_condition']}, "
         f"{context['fitted_temperature']:.4f}). T = 1 is that calibrated probability; "
         f"T = {low:g} and T = {high:g} show the same image made overconfident and "
         "underconfident, as in the experiment. The prediction comes from the sign of "
         "the logit, so it does not change with T."),
        "",
        ("An image is marked uncertain when its calibrated probability lies between "
         f"{1 - context['tau']:g} and {context['tau']:g}, where {context['tau']:g} is the "
         "confirmation threshold the planners use."),
        "",
        "## Summary",
        "",
        s["sentence"],
        "",
        "## Images",
        "",
        (f"| File | Prediction | P(diseased), T = 1 | T = {low:g} | T = {high:g} "
         "| Logit | Uncertain |"),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["rows"]:
        name = r["file"].replace("|", "\\|")
        lines.append(f"| {name} | {r['prediction']} | {r['probability_calibrated']:.4f} "
                     f"| {r['probability_low_T']:.4f} | {r['probability_high_T']:.4f} "
                     f"| {r['logit']:+.4f} | {'yes' if r['uncertain'] else 'no'} |")
    if result["rejected"]:
        lines += ["", "## Not analysed", ""] + [f"- {reason}" for reason in result["rejected"]]
    lines += ["", "## Limits", ""] + [f"- {text}" for text in LIMITS_TEXT]
    lines += ["", "## Privacy", "", PRIVACY_TEXT, "",
              f"Method and results: {context['repository_url']}", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ samples

def select_samples(pool_path: Path, n_correct_diseased: int, n_correct_healthy: int,
                   n_wrong: int) -> list[dict]:
    """Pick sample images from a stored logit pool, by rule rather than by eye.

    Correct images are spread evenly across interior quantiles of their confidence
    (from fairly unsure to fairly sure); the wrong ones are the most confident mistakes,
    alternating false negatives and false positives.
    """
    with np.load(pool_path, allow_pickle=False) as data:
        pools = {int(k.split("_")[1]): (data[k], data[f"paths_{k.split('_')[1]}"])
                 for k in data.files if k.startswith("logits_")}

    def spread(idx, logits, n):
        if n <= 0 or len(idx) == 0:
            return []
        # Interior quantiles of confidence: n = 4 takes the 20th, 40th, 60th and
        # 80th percentiles, n = 1 the median, so neither extreme is over-represented.
        order = idx[np.argsort(np.abs(logits[idx]))]
        q = np.linspace(0, 1, num=n + 2)[1:-1]
        picks = (q * (len(order) - 1)).round().astype(int)
        return list(order[np.unique(picks)])

    chosen = []
    for label, n in ((1, n_correct_diseased), (0, n_correct_healthy)):
        logits, paths = pools[label]
        right = np.flatnonzero((logits > 0).astype(int) == label)
        chosen += [{"key": str(paths[i]), "label": label, "logit": float(logits[i]),
                    "correct": True} for i in spread(right, logits, n)]
    wrong = []
    for label in (1, 0):
        logits, paths = pools[label]
        bad = np.flatnonzero((logits > 0).astype(int) != label)
        bad = bad[np.argsort(-np.abs(logits[bad]))]
        wrong.append([{"key": str(paths[i]), "label": label, "logit": float(logits[i]),
                       "correct": False} for i in bad])
    for i in range(int(n_wrong)):
        source = wrong[i % 2] or wrong[(i + 1) % 2]
        if source:
            chosen.append(source.pop(0))
    return chosen


def find_image(dataset_root: Path, key: str) -> Path:
    """The file for an item key (its last three path components) under a dataset.

    The file name is also looked for URL-encoded (spaces as ``%20``), which is how
    the Kaggle command-line tool saves a single file downloaded with ``-f``.
    """
    from urllib.parse import quote

    parts = Path(key).parts
    for name in dict.fromkeys((parts[-1], quote(parts[-1]))):
        for candidate in Path(dataset_root).rglob(name):
            if candidate.parts[-3:-1] == parts[-3:-1]:
                return candidate
    raise FileNotFoundError(f"{key} not found under {dataset_root}")


# ------------------------------------------------- deployed path vs stage 02

def compare_sample_logits(model, samples: list[dict], atol: float) -> dict:
    """Each sample's logit on the deployed path against the logit stage 02 recorded.

    ``samples`` are entries of a ``demo_samples.json`` listing, each with its
    ``path`` added. Stage 02 scored the same photograph with PyTorch from the
    training code's transform; here it goes through ``eval_array`` and the exported
    graph, so a difference would expose a preprocessing or export mismatch on real
    imagery. The check passes when every logit is within ``atol`` and every
    prediction (the logit's sign) agrees.
    """
    rows = []
    for s in samples:
        got = float(model.predict(Path(s["path"]))["logit"])
        recorded = float(s["logit"])
        rows.append({"store_name": s["store_name"], "dataset": s["dataset"],
                     "key": s["key"], "stage02_logit": recorded, "deployed_logit": got,
                     "abs_difference": abs(got - recorded),
                     "same_prediction": (got > 0) == (recorded > 0)})
    worst = max((r["abs_difference"] for r in rows), default=None)
    return {"n_samples": len(rows), "max_abs_difference": worst, "logit_atol": float(atol),
            "passed": bool(rows) and worst <= atol and all(r["same_prediction"] for r in rows),
            "samples": rows}


def _listed_samples(folder: Path, listing: str) -> list[dict]:
    import json

    path = folder / listing
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)["samples"]
    return [{**s, "path": folder / s["store_name"]} for s in entries]


def run_sample_check(config=None, store=None, push: bool = False) -> dict:
    """Check every demonstration sample available here and write the record.

    The PlantDoc samples come from stage ``08_deployment`` of the store; the
    PlantVillage samples, which may not be published there, from this machine's
    ``samples_local`` folder (written by experiments/08_select_samples.py), if it
    exists. The record names the source of each group. ``push`` files the record in
    the stage, refused unless the classifier checked is the one filed there, the
    check passed, and the working tree is committed.
    """
    import json
    import platform
    from datetime import datetime, timezone

    import onnxruntime as ort

    from scoutfield.app.export_onnx import git_state, locate_classifier, work_dir
    from scoutfield.app.inference import ScoutFieldInference
    from scoutfield.utils.artifacts import get_store, sha256_of

    config = config or deployment_config()
    store = store or get_store()
    stage = str(config["stage"])
    listing = str(config.section("demo")["samples"]["file"])
    onnx_path, source = locate_classifier(config, store)
    model = ScoutFieldInference(onnx_path)

    samples, sources = [], {}
    files = sample_files(store, config)
    if files:
        public = [{**s, "path": store.pull(stage, s["store_name"])}
                  for s in store.pull_json(stage, listing)["samples"]]
        samples += public
        sources["store"] = {"n": len(public), "where": store.remote_path(stage, listing)}
    local_dir = work_dir(config, store.run_id) / "samples_local"
    local = _listed_samples(local_dir, listing)
    samples += local
    if local:
        sources["laptop_only"] = {"n": len(local), "where": str(local_dir.name),
                                  "note": "licence keeps these images out of the "
                                          "public store; their logits are recorded here"}
    if not samples:
        raise FileNotFoundError(
            f"no demonstration samples: looked in {store.remote_path(stage, listing)} "
            f"and in {local_dir}. Run `python experiments/08_select_samples.py` first.")

    atol = float(config.section("export")["verify"]["logit_atol"])
    check = compare_sample_logits(model, samples, atol)
    record = {
        "run_id": store.run_id,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "what": "demonstration sample photographs through the deployed path (NumPy and "
                "Pillow preprocessing, ONNX Runtime on the CPU) against the logit stage "
                "02 recorded for the same photograph (PyTorch, from the stored logit pool)",
        "onnx": {"file": onnx_path.name, "sha256": sha256_of(onnx_path), "source": source},
        "sample_sources": sources,
        **check,
        **git_state(),
        "versions": {"python": platform.python_version(), "onnxruntime": ort.__version__,
                     "numpy": np.__version__},
    }
    out = work_dir(config, store.run_id) / str(config.section("demo")["sample_check_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    if push:
        filed = store.stage_entry(stage).get("files", {}) \
            .get(config.section("export")["classifier_file"], {}).get("sha256")
        if filed != record["onnx"]["sha256"]:
            raise ValueError(
                f"refusing to push: the classifier checked is not the classifier.onnx "
                f"filed in {stage} of run {store.run_id}.")
        if not record["passed"]:
            raise ValueError(f"refusing to push a failed check (largest difference "
                             f"{record['max_abs_difference']:.3g}, tolerance {atol:g}).")
        if record["git_dirty"]:
            raise ValueError("refusing to push a check made from uncommitted code; "
                             "commit first so the record names the code that made it.")
        store.push(stage, [out], "deployed-path logits of the demonstration samples")
    return {"record": record, "path": out, "pushed": push}
