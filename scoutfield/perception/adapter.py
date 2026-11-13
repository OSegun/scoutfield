"""
The interface contract between a real classifier and the pilot's environment.

Why this file exists
--------------------
The pilot's ``ScoutEnv`` calls perception through exactly one method:

    CalibratedClassifier.observe(true_label: int) -> tuple[float, int]

From the pilot's developer contract: "Swapping in the real CNN means changing
only this method." ``CNNClassifier`` below is that swap. It must match the
signature exactly — same call, same return shape, same types — because the
study's claim is that *calibration error at fixed accuracy* changes planner
performance, and testing whether that survives a real classifier requires the
classifier to be the only thing that changed.

How it works
------------
The environment hands over a ground-truth label for the cell the drone is over.
The pilot's surrogate answered by sampling a latent logit from a class-conditional
Gaussian. This class instead draws a real image of that class from a held-out
pool, runs it through the network, and returns the temperature-scaled probability
together with the hard prediction.

That indirection — label in, image sampled, probability out — is what lets a
grid-world environment consume a real image classifier without the environment
knowing anything about images.

The accuracy-invariance invariant
---------------------------------
Temperature scaling divides *logits* by a scalar before the link function. The
link is strictly monotone and the decision boundary sits at logit 0, so the
arg-max cannot move: accuracy is invariant to T while confidence is not. That
property is the entire experimental instrument.

It must hold for the real classifier too. If it does not, the temperature is
almost certainly being applied after a softmax, or the prediction is being
recomputed from already-scaled values. Assert it, do not assume it:

    accs = {evaluate(model, T=t)["accuracy"] for t in (0.3, 1.0, 4.0)}
    assert len(accs) == 1, f"instrument broken: {accs}"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# The cached pool's filename carries the split it was built from. A pool from the
# in-distribution test split and one from the PlantDoc shift split are different
# experimental conditions with different accuracy and calibration, and a single
# filename would let one silently overwrite the other — producing a sweep whose
# rows are labelled with one condition's ECE while the agent observed the other's
# logits. The split is in the name so that cannot happen.
def logit_pool_filename(split: str) -> str:
    return f"logit_pool_{split}.npz"


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class CNNClassifier:
    """Fine-tuned EfficientNet-B0 exposed through the pilot's observe() contract.

    Parameters
    ----------
    checkpoint
        Path to a checkpoint written by ``perception.train``.
    logit_pool
        Precomputed logits for a held-out image pool, grouped by true label.
        Precomputing matters: a scouting episode makes hundreds of observations
        and a sweep runs thousands of episodes, so a forward pass per observation
        would make the experiment take days instead of minutes. The logits are
        the only thing temperature scaling acts on, so caching them is exact
        rather than approximate — no fidelity is lost.
    temperature
        The calibration knob, expressed *relative to the calibrated point*.
        T = 1 is calibrated, T < 1 overconfident, T > 1 underconfident.
    reference_temperature
        The fitted temperature of the reference condition, which maps the
        network's raw logits onto calibrated ones: the lab temperature (fitted on
        PlantVillage validation) or the shift temperature (fitted on PlantDoc
        ``shift_fit``). The effective divisor is ``reference_temperature *
        temperature``.

        This indirection is what makes the sweep mean the same thing it meant in
        the pilot. The pilot's surrogate reported the exactly calibrated
        posterior at T = 1 by construction; a real network's raw logits are
        miscalibrated, so its ECE bottoms out at the fitted T, not at 1. Sweeping
        the raw temperature would therefore put the calibrated point somewhere
        other than 1 and make "T = 4" a different manipulation in each phase.
        Sweeping relative to the fit keeps the two comparable.
    rng
        Seeded generator. Which image is drawn must be reproducible, or a run is
        not reproducible from its seed.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        logit_pool: dict[int, np.ndarray] | None = None,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
        reference_temperature: float = 1.0,
        pool_split: str = "test",
    ):
        self.T = float(temperature)
        self.reference_temperature = float(reference_temperature)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._checkpoint = Path(checkpoint) if checkpoint else None
        # Which split the observed logits come from. Recorded rather than implied:
        # it is the experimental condition, and every result carries it.
        self.pool_split = str(pool_split)

        if logit_pool is None:
            if self._checkpoint is None:
                raise ValueError(
                    "CNNClassifier needs either a `logit_pool` or a `checkpoint`. "
                    "It deliberately has no surrogate fallback: silently standing in "
                    "for the real classifier would make a synthetic result "
                    "indistinguishable from a measured one. Train the model first "
                    "(experiments/01_finetune_perception.py), or pass a pool directly."
                )
            # An explicit checkpoint builds the pool locally. Pipeline stages use
            # `from_store` instead, which reads the pool stage 02 recorded.
            logit_pool, _ = build_logit_pool(self._checkpoint, split=self.pool_split)

        self._pool = {int(k): np.asarray(v, dtype=float).ravel() for k, v in logit_pool.items()}
        if not self._pool or any(v.size == 0 for v in self._pool.values()):
            raise ValueError(f"logit pool is empty for at least one class: "
                             f"{ {k: v.size for k, v in self._pool.items()} }")

    # ------------------------------------------------------------------ API

    def observe(self, true_label: int) -> tuple[float, int]:
        """Return ``(reported_probability_of_disease, hard_prediction)``.

        Contract, enforced by tests/test_pilot_contract.py:

        * exactly one positional argument, an int label
        * returns a 2-tuple of ``(float, int)``
        * the int is 0 or 1
        * the float is in [0, 1]
        * the hard prediction does not depend on ``self.T``

        The temperature scales the LOGIT, never the probability, and the hard
        prediction comes from the unscaled sign — which is what makes accuracy
        invariant to T.
        """
        z = self._draw_logit(true_label)
        return float(_sigmoid(z / (self.reference_temperature * self.T))), int(z > 0)

    @property
    def accuracy(self) -> float:
        """Measured accuracy of the pooled logits. Does not vary with ``self.T``.

        Note this is *measured*, unlike the pilot where accuracy was an input
        parameter set from Ahmad et al. (2023). Whatever this returns becomes the
        BASE_ACC of this phase and must be reported as such.

        Weighted by pool size per class, so it matches the accuracy that would be
        measured over the underlying split rather than a macro-average.
        """
        correct = sum(int(((z > 0).astype(int) == label).sum())
                      for label, z in self._pool.items())
        total = sum(z.size for z in self._pool.values())
        return correct / total

    def set_temperature(self, temperature: float) -> None:
        """Change T in place. Cheap by design — the sweep calls it thousands of
        times, and rebuilding the logit pool per temperature would be both slow
        and wrong, since the pool must be identical across temperatures for the
        comparison to isolate calibration."""
        self.T = float(temperature)

    # ------------------------------------------------------------ internals

    def _draw_logit(self, true_label: int) -> float:
        """Sample one cached logit for the given class, using ``self.rng``."""
        pool = self._pool.get(int(true_label))
        if pool is None:
            raise KeyError(f"no pooled logits for label {true_label}; "
                           f"pool has {sorted(self._pool)}")
        return float(pool[self.rng.integers(pool.size)])



    @classmethod
    def from_store(cls, pool_split: str, temperature: float = 1.0,
                   reference_temperature: float = 1.0,
                   rng: np.random.Generator | None = None, store=None) -> CNNClassifier:
        """Build from the logit pool stage 02 pushed for ``pool_split``.

        The pool is checked against the split membership recorded in
        ``calibration_sweep.json`` and rebuilt from the stage 01 checkpoint if it
        does not match (see ``load_or_build_logit_pool``).
        """
        pool = load_or_build_logit_pool(pool_split, store=store)
        return cls(logit_pool=pool, temperature=temperature, rng=rng,
                   reference_temperature=reference_temperature, pool_split=pool_split)


# ------------------------------------------------------------------- pools

def pool_from_logits(logits, labels, items) -> tuple[dict[int, np.ndarray], dict[int, list]]:
    """Group one split's logits by true label, with the image each came from.

    ``items`` is the split's ``(path, class)`` list in loader order, which is the
    order ``collect_logits`` returns logits in (the loaders for held-out splits do
    not shuffle). Image names are stored so a confident mistake can be traced to
    the picture that produced it.
    """
    from scoutfield.perception.datasets import item_key

    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(items) != logits.size:
        raise ValueError(f"{logits.size} logits for {len(items)} images; the loader and "
                         f"the item list are out of step")
    pool = {int(c): logits[labels == c] for c in np.unique(labels)}
    names = {int(c): [item_key(p) for (p, _), lab in zip(items, labels) if lab == c]
             for c in np.unique(labels)}
    return pool, names


def save_logit_pool(path: str | Path, pool: dict[int, np.ndarray],
                    image_paths: dict[int, list[str]] | None = None,
                    membership: str | None = None, split: str | None = None) -> None:
    """Persist a pool as npz, keyed by class, with the split it was built from.

    ``membership`` is the split's membership hash (``datasets.membership_hash``).
    It travels inside the file, so a pool cannot be mistaken for one built from a
    different split, whatever its filename says.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"logits_{k}": np.asarray(v, dtype=float) for k, v in pool.items()}
    for k, names in (image_paths or {}).items():
        payload[f"paths_{k}"] = np.asarray(names, dtype=str)
    if membership is not None:
        payload["membership"] = np.asarray(membership)
    if split is not None:
        payload["split"] = np.asarray(split)
    np.savez(path, **payload)


def load_logit_pool(path: str | Path) -> dict[int, np.ndarray]:
    """Read a pool written by ``save_logit_pool``."""
    with np.load(path, allow_pickle=False) as data:
        return {int(k.split("_")[1]): data[k] for k in data.files if k.startswith("logits_")}


def pool_membership(path: str | Path) -> str | None:
    """The membership hash stored in a pool file, or None for a pool without one."""
    with np.load(path, allow_pickle=False) as data:
        return str(data["membership"]) if "membership" in data.files else None


def load_or_build_logit_pool(split: str, store=None) -> dict[int, np.ndarray]:
    """The stage 02 pool for ``split``, verified against the recorded split.

    The expected membership is read from ``calibration_sweep.json`` in the store.
    A stored pool whose membership matches is used as is. A missing or mismatched
    pool is rebuilt from the stage 01 checkpoint, which needs the datasets on this
    machine, and the rebuilt pool must then match the recorded membership; if it
    does not, the data here is not the data the run used, and that raises.
    """
    from scoutfield.utils.artifacts import get_store

    store = store or get_store()
    calibration = store.pull_json("02_calibration", "calibration_sweep.json")
    expected = calibration.get("pools", {}).get(split, {}).get("membership")
    if expected is None:
        raise KeyError(f"calibration_sweep.json in run {store.run_id} records no pool "
                       f"for split {split!r}; it has {sorted(calibration.get('pools', {}))}")

    cached = store.pull("02_calibration", logit_pool_filename(split), required=False)
    if cached is not None and pool_membership(cached) == expected:
        return load_logit_pool(cached)

    checkpoint = store.pull("01_perception", "best.pt")
    pool, membership = build_logit_pool(checkpoint, split=split)
    if membership != expected:
        raise ValueError(
            f"the {split!r} split on this machine (membership {membership[:12]}) is not "
            f"the split run {store.run_id} recorded ({expected[:12]}). The datasets "
            f"here differ from the ones the run used; attach the same Kaggle datasets."
        )
    return pool


def build_logit_pool(checkpoint: str | Path, split: str, config=None,
                     out_path: str | Path | None = None
                     ) -> tuple[dict[int, np.ndarray], str]:
    """Run the network once over a split and cache the logits by true label.

    Returns ``(pool, membership_hash)``. Written to ``out_path`` (default: the
    writable checkpoints directory) with the membership inside the file.
    """
    from scoutfield.config import load_config
    from scoutfield.perception.calibrate import _load_model_state, collect_logits
    from scoutfield.perception.datasets import build_dataloaders
    from scoutfield.perception.model import build_model
    from scoutfield.utils.paths import checkpoints_dir, repo_root

    if config is None:
        config = load_config(repo_root() / "configs" / "perception.yaml")

    model = build_model(config)
    model.load_state_dict(_load_model_state(checkpoint))

    loaders = build_dataloaders(config)
    if split not in loaders:
        raise KeyError(f"unknown split '{split}'; have {sorted(k for k in loaders)}")

    logits, labels, _ = collect_logits(model, loaders[split])
    pool, names = pool_from_logits(logits, labels, loaders[split].dataset.items)
    membership = loaders["meta"]["membership"][split]
    out = Path(out_path) if out_path else checkpoints_dir("perception") / logit_pool_filename(split)
    save_logit_pool(out, pool, names, membership=membership, split=split)
    return pool, membership
