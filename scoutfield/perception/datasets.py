"""
Datasets: PlantVillage for training, PlantDoc for the distribution-shift test.

Why two datasets
----------------
PlantVillage is lab-condition imagery — uniform background, controlled lighting,
single leaf centred in frame. Classifiers reach very high accuracy on it and that
accuracy does not transfer to a field. PlantDoc is field imagery: cluttered
backgrounds, variable lighting, occlusion.

Training on one and testing on the other is the distribution shift the study
needs. Deep classifiers are systematically miscalibrated and degrade *further*
under shift, so the PlantDoc evaluation is where the calibration story becomes
non-trivial rather than academic.

**Never train on PlantDoc.** It is the held-out test and contaminating it
destroys the only shift measurement this project has.

Label space
-----------
The pilot's environment is binary: a cell is healthy or diseased. Both datasets
are multi-class over (crop, condition) pairs, so they are collapsed to
healthy / diseased. Keep the original fine-grained label alongside the binary
one — per-class calibration is far more informative than an aggregate ECE when
explaining *why* the model is miscalibrated.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from scoutfield.utils.paths import data_dir

# ImageNet statistics: the backbone is pretrained, so its normalisation is not a
# free choice. Applied identically to all four splits — a train/shift mismatch
# would surface as a distribution-shift effect that is really a preprocessing bug.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Evaluation resizes the shorter side to image_size * 1.14 before a centre crop to
# image_size: the usual ImageNet evaluation ratio (256 / 224), so the crop keeps the
# middle of the leaf and drops the border.
EVAL_RESIZE_FACTOR = 1.14

# --------------------------------------------------------------- label mapping
# The two datasets name their classes by incompatible conventions, and one regex
# cannot serve both.
#
# PlantVillage uses "<Crop>___<Condition>" and names the healthy class explicitly:
# "Tomato___healthy", "Apple___Black_rot".
#
# PlantDoc does not. Its healthy classes are "<Crop> leaf" with no condition word
# at all — "Apple leaf", "Tomato leaf", "grape leaf" — while its diseased classes
# carry the condition somewhere in the name: "Apple Scab Leaf", "Tomato mold leaf",
# "grape leaf black rot". The word "healthy" appears nowhere in PlantDoc.
#
# Matching only on "healthy" therefore labels EVERY PlantDoc image as diseased.
# The shift split becomes 100% positive, accuracy under shift collapses to the
# diseased prevalence, and the resulting number is plausible enough to report.
# That is the most expensive failure available here, because the PlantDoc
# evaluation is the project's only distribution-shift measurement.
#
# PlantDoc is therefore enumerated explicitly. Names verified against the canonical
# release (Singh et al., 2020; github.com/pratikkayal/PlantDoc-Dataset): `train/`
# holds 28 classes and `test/` holds 27 — the test split omits "Tomato two spotted
# spider mites leaf". An unrecognised class raises rather than defaulting, so a
# mirror that renames a directory fails loudly instead of mislabelling in silence.
_PLANTVILLAGE_SEPARATOR = "___"
_HEALTHY = re.compile(r"healthy", re.IGNORECASE)


def _normalise(class_name: str) -> str:
    """Case- and separator-insensitive key, so mirrors that swap '_' for ' ' match."""
    return re.sub(r"[\s_]+", " ", class_name).strip().casefold()


PLANTDOC_HEALTHY: frozenset[str] = frozenset(_normalise(c) for c in (
    "Apple leaf",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "Peach leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Strawberry leaf",
    "Tomato leaf",
    "grape leaf",
))

PLANTDOC_DISEASED: frozenset[str] = frozenset(_normalise(c) for c in (
    "Apple Scab Leaf",
    "Apple rust leaf",
    "Bell_pepper leaf spot",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Squash Powdery mildew leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
    "Tomato two spotted spider mites leaf",
    "grape leaf black rot",
))

# Directory names that are not classes. PlantVillage ships three renderings of the
# same photographs side by side; PlantDoc ships a train/test split. In both cases the
# real class directories sit one level below, and stopping here would turn "color" or
# "train" into a class name.
_VARIANT_DIRS = frozenset({"color", "grayscale", "greyscale", "segmented"})
_SPLIT_DIRS = frozenset({"train", "test", "val", "valid", "validation"})


@dataclass
class SplitSpec:
    """One data split, resolved to a concrete directory."""

    name: str          # "train" | "val" | "test" | "shift"
    root: Path
    dataset: str       # "plantvillage" | "plantdoc"


def binarise_label(class_name: str) -> int:
    """1 = diseased, 0 = healthy, from the fine-grained class directory name.

    Handles both naming conventions and refuses to guess at a third. See the
    comment above ``PLANTDOC_HEALTHY`` for why one regex is not enough.
    """
    if _PLANTVILLAGE_SEPARATOR in class_name:
        # PlantVillage: the condition follows the separator, and the healthy class
        # says so in as many words.
        return 0 if _HEALTHY.search(class_name) else 1

    key = _normalise(class_name)
    if key in PLANTDOC_HEALTHY:
        return 0
    if key in PLANTDOC_DISEASED:
        return 1
    if _HEALTHY.search(class_name):
        # A third mirror that follows neither convention but still marks healthy
        # explicitly. Safe to honour; anything else is not.
        return 0

    raise KeyError(
        f"unrecognised class directory {class_name!r}: it matches neither the "
        f"PlantVillage convention ('<Crop>{_PLANTVILLAGE_SEPARATOR}<Condition>') nor any "
        f"known PlantDoc class. Add it to PLANTDOC_HEALTHY or PLANTDOC_DISEASED in "
        f"scoutfield/perception/datasets.py. Do not let it default — a wrong binary "
        f"label here is invisible in every number downstream."
    )


def resolve_dataset_roots(dataset: str, kaggle_slugs: dict[str, str],
                          variant: str = "color") -> list[Path]:
    """Locate a dataset on the current machine and resolve it to class-dir parents.

    On Kaggle the datasets are attached to the notebook and mounted read-only at
    /kaggle/input/<slug>; locally they are expected under ./data/<slug>.
    ``utils.paths.data_dir`` handles the difference.

    Returns a list because PlantDoc arrives pre-split into train/test and both
    halves are wanted — see ``_descend_to_class_dirs``.
    """
    if dataset not in kaggle_slugs:
        raise KeyError(f"no kaggle slug configured for dataset '{dataset}'")
    slug = kaggle_slugs[dataset]
    root = data_dir(slug)
    if not root.exists():
        # Kaggle occasionally mounts a dataset under a differently-cased name than
        # the slug tail. That is safe to accept — an exact case-insensitive match is
        # still the same dataset. Anything looser would risk training on the wrong
        # data silently, so it stays an error.
        base = root.parent
        siblings = sorted(d.name for d in base.iterdir() if d.is_dir()) \
            if base.exists() else []
        match = next((n for n in siblings if n.casefold() == root.name.casefold()), None)
        if match is not None:
            root = base / match
        else:
            raise FileNotFoundError(
                f"dataset '{dataset}' not found at {root}. "
                f"On Kaggle, attach '{slug}' to the notebook (Add Input -> Datasets); "
                f"note that editing kernel-metadata.json does not change a session "
                f"that is already running. Locally, download it into {root}. "
                f"Present in {base}: {siblings or '(nothing)'}"
            )
    return _descend_to_class_dirs(root, variant)


def _descend_to_class_dirs(root: Path, variant: str = "color") -> list[Path]:
    """Resolve a mount point to the directories that actually hold class folders.

    Three shapes occur, and mistaking any of them yields class names that are not
    classes — which then flow into ``binarise_label`` and into the reported
    prevalence:

    * **A chain of single-child wrappers.** Kaggle archives commonly nest the real
      directories a level or two down, and the depth differs per dataset. Descend.
    * **PlantVillage's {color, grayscale, segmented}.** Three renderings of the
      same photographs. Take exactly one: keeping all three counts every leaf three
      times and, worse, lets the source-image split leak one photograph's
      renderings across train and val.
    * **PlantDoc's {train, test}.** Take both, merged. PlantDoc is the held-out
      shift set in its entirety, so its internal split carries no meaning here and
      using half of it would discard data for nothing.
    """
    current = root
    for _ in range(6):
        subdirs = [d for d in sorted(current.iterdir()) if d.is_dir()]
        if not subdirs:
            break
        names = {d.name.casefold() for d in subdirs}

        # A lone subdirectory is a wrapper only if it holds no images of its own.
        # Without that guard, a dataset with exactly one class would be mistaken for
        # a wrapper and descended straight past its only class directory.
        if len(subdirs) == 1 and not _holds_images(subdirs[0]):
            current = subdirs[0]
            continue

        # Two or more variant names at one level identify it as the variant level.
        # A strict subset test is too brittle: the circulated PlantVillage archive
        # carries a fourth sibling, a wrapper duplicating the same three renderings,
        # and the test would fail and let "color" through as a class name. One
        # variant name alone is not enough — a lone directory called "segmented"
        # says nothing — so the threshold is two.
        if len(names & _VARIANT_DIRS) >= 2:
            chosen = next((d for d in subdirs if d.name.casefold() == variant.casefold()),
                          None)
            if chosen is None:
                raise FileNotFoundError(
                    f"image variant '{variant}' not found under {current}; "
                    f"available: {sorted(d.name for d in subdirs)}"
                )
            if not any(p.is_dir() for p in chosen.iterdir()):
                raise FileNotFoundError(
                    f"variant directory {chosen} holds no class subdirectories; "
                    f"the archive layout is not what this code expects"
                )
            # A variant directory is the class-dir parent by definition, so stop
            # here rather than descending again.
            return [chosen]

        if names <= _SPLIT_DIRS:
            return list(subdirs)

        return [current]
    return [current]


def _holds_images(directory: Path) -> bool:
    """True if this directory contains image files directly — i.e. it is a class dir."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return any(p.suffix.lower() in exts and p.is_file() for p in directory.iterdir())


def _source_key(path: Path) -> str:
    """Group augmented variants of one source image under a single key.

    PlantVillage is circulated with augmented duplicates whose filenames carry a
    transform suffix. The same leaf appearing in train and val inflates accuracy
    and, worse, makes the model look better calibrated than it is — so splitting
    happens over this key, never over individual files.
    """
    stem = path.stem
    stem = re.sub(
        r"_+(?:final|aug(?:mented)?)?_*"
        r"(?:flip|mirror|rot(?:ate|ated)?|rotation|scale|shear|zoom|bright|blur|noise|"
        r"trans(?:late|lation)?)"
        r"[-_]*\d*$",
        "",
        stem,
        flags=re.IGNORECASE,
    )
    stem = (re.sub(r"_+\d+$", "", stem)
            if re.search(r"(copy|aug)", stem, re.IGNORECASE) else stem)
    return f"{path.parent.name}/{stem}"


def _index_split(roots: Path | list[Path]) -> list[tuple[Path, str]]:
    """All (image path, fine-grained class name) pairs under one or more roots.

    Accepts several roots so PlantDoc's train and test halves merge into the single
    shift set. A class appearing under more than one root keeps one name, so the
    merge is by class rather than by directory.
    """
    if isinstance(roots, Path):
        roots = [roots]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    items: list[tuple[Path, str]] = []
    for root in roots:
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for img in sorted(class_dir.rglob("*")):
                if img.suffix.lower() in exts and img.is_file():
                    items.append((img, class_dir.name))
    if not items:
        raise FileNotFoundError(f"no images found under {[str(r) for r in roots]}")
    return items


class _ImageFolderDataset:
    """Minimal image dataset returning ``(tensor, binary_label, fine_label)``.

    torchvision's ``ImageFolder`` is not used because the split is by source
    image rather than by file, and the fine-grained label must survive alongside
    the binary one for the per-class calibration breakdown.
    """

    def __init__(self, items, transform, class_to_idx, binarise: bool = True):
        self.items = list(items)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.binarise = binarise

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image

        path, class_name = self.items[i]
        with Image.open(path) as im:
            img = self.transform(im.convert("RGB"))
        y = binarise_label(class_name) if self.binarise else self.class_to_idx[class_name]
        return img, y, self.class_to_idx[class_name]


def _transforms(image_size: int, train: bool):
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * EVAL_RESIZE_FACTOR)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_array(image, image_size: int):
    """The evaluation transform in NumPy and Pillow, without torch.

    The deployed demonstration runs with ONNX Runtime and has no PyTorch, so it
    cannot call ``_transforms``. This reproduces ``_transforms(image_size,
    train=False)`` step by step: bilinear resize of the shorter side to
    ``int(image_size * EVAL_RESIZE_FACTOR)`` (the long side scaled and truncated as
    torchvision does), centre crop, scale to [0, 1], ImageNet normalisation. It
    returns a float32 array of shape ``(3, image_size, image_size)``.
    ``tests/test_app.py`` asserts it equals the torchvision transform, so the two
    cannot drift apart unnoticed.
    """
    import numpy as np
    from PIL import Image

    if image.mode != "RGB":  # convert() copies even an RGB image; no pixel changes
        image = image.convert("RGB")
    short = int(image_size * EVAL_RESIZE_FACTOR)
    w, h = image.size
    if w <= h:
        size = (short, int(short * h / w))
    else:
        size = (int(short * w / h), short)
    image = image.resize(size, Image.BILINEAR)
    w, h = image.size
    top = round((h - image_size) / 2.0)
    left = round((w - image_size) / 2.0)
    image = image.crop((left, top, left + image_size, top + image_size))
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


def item_key(path: Path) -> str:
    """A machine-independent name for one image: its last three path components.

    ``color/Tomato___healthy/x.jpg`` for PlantVillage, ``train/Apple leaf/x.jpg``
    for PlantDoc. The mount point differs between Kaggle and a laptop; these three
    components do not, so a split recorded on one machine can be checked on another.
    Three rather than two because PlantDoc's train and test halves can reuse a
    filename within one class.
    """
    return "/".join(Path(path).parts[-3:])


def membership_hash(items) -> str:
    """SHA-256 of a split's sorted member list. Equal hashes mean equal splits."""
    import hashlib

    names = sorted(item_key(path) for path, _ in items)
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def content_hashes(items, cache_file: Path | None = None) -> dict[str, str]:
    """SHA-1 of every image's bytes, keyed by ``item_key``.

    Cached in ``cache_file`` by key and file size, because hashing PlantVillage
    reads every byte of it and every stage that builds the split needs the hashes.
    SHA-1 here detects identical files; it is not a security boundary.
    """
    import hashlib
    import json

    cache: dict[str, list] = {}
    if cache_file is not None and cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as fh:
            cache = json.load(fh)

    out: dict[str, str] = {}
    changed = False
    for path, _ in items:
        key = item_key(path)
        size = Path(path).stat().st_size
        hit = cache.get(key)
        if hit and hit[0] == size:
            out[key] = hit[1]
            continue
        digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()
        cache[key] = [size, digest]
        out[key] = digest
        changed = True

    if cache_file is not None and changed:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh)
        from scoutfield.utils.checkpoint import replace_atomically

        replace_atomically(tmp, cache_file)
    return out


def _group_items(items, hashes: dict[str, str] | None) -> dict[str, list]:
    """Group images that must stay together: same source key OR same file bytes.

    The source key catches augmented variants named after their original; the
    content hash catches exact copies under unrelated names. Either link is enough
    to join two images, and links chain, so the groups are the connected
    components (a union-find). Each group is named by its smallest source key, so
    the grouping is deterministic.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first_by_hash: dict[str, str] = {}
    keys = []
    for path, _ in items:
        key = "k:" + _source_key(path)
        find(key)
        keys.append(key)
        if hashes is not None:
            digest = hashes[item_key(path)]
            if digest in first_by_hash:
                union(key, first_by_hash[digest])
            else:
                first_by_hash[digest] = key

    groups: dict[str, list] = {}
    for (path, cls), key in zip(items, keys):
        groups.setdefault(find(key)[2:], []).append((path, cls))
    return groups


def _split_by_source(items, val_fraction: float, test_fraction: float, seed: int,
                     hashes: dict[str, str] | None = None):
    """Partition by source-image group so variants and copies cannot straddle splits.

    With ``hashes`` the groups also join byte-identical files (see ``_group_items``);
    without them the split is by source key alone.
    """
    import numpy as np

    groups = _group_items(items, hashes)

    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_val = round(val_fraction * n)
    n_test = round(test_fraction * n)
    parts = {
        "val": keys[:n_val],
        "test": keys[n_val:n_val + n_test],
        "train": keys[n_val + n_test:],
    }
    return {name: [it for k in ks for it in groups[k]] for name, ks in parts.items()}


def _split_shift(items, fit_fraction: float, seed: int,
                 hashes: dict[str, str] | None = None):
    """Split PlantDoc into ``shift_fit`` and ``shift_eval``, stratified by binary label.

    ``shift_fit`` fits the shift temperature; ``shift_eval`` is what the planners
    are scored on. Grouping is the same as for PlantVillage, so near-identical
    images cannot sit on both sides. A group takes its majority binary label for
    stratification. Healthy and diseased groups are each split at
    ``fit_fraction``, so both halves keep PlantDoc's class balance.
    """
    import numpy as np

    groups = _group_items(items, hashes)
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for key, members in groups.items():
        labels = [binarise_label(cls) for _, cls in members]
        by_label[int(sum(labels) * 2 >= len(labels))].append(key)

    rng = np.random.default_rng(seed)
    fit_keys, eval_keys = [], []
    for label in (0, 1):
        keys = sorted(by_label[label])
        rng.shuffle(keys)
        n_fit = round(fit_fraction * len(keys))
        fit_keys += keys[:n_fit]
        eval_keys += keys[n_fit:]
    return {
        "shift_fit": [it for k in sorted(fit_keys) for it in groups[k]],
        "shift_eval": [it for k in sorted(eval_keys) for it in groups[k]],
    }


def _crossings(parts: dict[str, list], key_fn) -> int:
    """How many distinct keys appear in more than one of ``parts``."""
    seen: dict[str, set] = {}
    for name, items in parts.items():
        for path, _ in items:
            seen.setdefault(key_fn(path), set()).add(name)
    return sum(1 for owners in seen.values() if len(owners) > 1)


def hash_cache_file(dataset: str) -> Path:
    """Where content hashes for one dataset are cached between runs."""
    from scoutfield.utils.paths import repo_root

    return repo_root() / ".artifact_cache" / "content_hashes" / f"{dataset}.json"


def _describe(items) -> dict:
    fine = Counter(cls for _, cls in items)
    binary = Counter(binarise_label(cls) for _, cls in items)
    total = max(sum(binary.values()), 1)
    return {
        "n": len(items),
        "n_classes": len(fine),
        "healthy": binary.get(0, 0),
        "diseased": binary.get(1, 0),
        "diseased_fraction": binary.get(1, 0) / total,
    }


def index_and_split(config) -> dict[str, object]:
    """Index both datasets and build every split, without touching torch.

    Returns ``{"parts": {split: items}, "hashes": {dataset: {key: sha1}},
    "classes": [...], "roots": {...}, "variant": str}``. Splits:

    * ``train`` / ``val`` / ``test`` — PlantVillage, grouped by source key and
      content hash so variants and exact copies stay on one side.
    * ``shift`` — all of PlantDoc.
    * ``shift_fit`` / ``shift_eval`` — PlantDoc halved, stratified by binary label,
      at ``calibration.shift_fit_fraction`` with ``calibration.shift_split_seed``.

    Shared by ``build_dataloaders`` and the leakage check, so the split the check
    audits is the split training uses.
    """
    data_cfg = config.section("data")
    cal_cfg = config.section("calibration")
    seed = int(config["seed"])
    val_fraction = float(data_cfg["val_fraction"])
    # Held out from PlantVillage for early stopping and in-distribution reporting,
    # keeping the validation split exclusively for the temperature fit.
    test_fraction = float(data_cfg.get("test_fraction", val_fraction))
    slugs = data_cfg["kaggle_slugs"]
    # Which of PlantVillage's three renderings to train on. Read from config rather
    # than fixed in code, and recorded in the split metadata, so a run that used a
    # different rendering is identifiable after the fact.
    variant = str(data_cfg.get("plantvillage_variant", "color"))
    train_name, shift_name = data_cfg["train_dataset"], data_cfg["shift_dataset"]

    train_roots = resolve_dataset_roots(train_name, slugs, variant)
    shift_roots = resolve_dataset_roots(shift_name, slugs, variant)
    train_items = _index_split(train_roots)
    shift_items = _index_split(shift_roots)

    hashes = {
        train_name: content_hashes(train_items, hash_cache_file(train_name)),
        shift_name: content_hashes(shift_items, hash_cache_file(shift_name)),
    }
    parts = _split_by_source(train_items, val_fraction, test_fraction, seed,
                             hashes=hashes[train_name])
    parts["shift"] = shift_items
    parts.update(_split_shift(shift_items, float(cal_cfg["shift_fit_fraction"]),
                              int(cal_cfg["shift_split_seed"]), hashes=hashes[shift_name]))

    classes = sorted({cls for _, cls in train_items} | {cls for _, cls in shift_items})
    return {
        "parts": parts,
        "hashes": hashes,
        "classes": classes,
        "roots": {"train": [str(r) for r in train_roots],
                  "shift": [str(r) for r in shift_roots]},
        "variant": variant,
        "datasets": {"train": train_name, "shift": shift_name},
    }


def leakage_report(config, indexed: dict | None = None) -> dict:
    """Count every way an image could appear on both sides of a split.

    Reported per split family: source keys and content hashes found in more than
    one PlantVillage split (both must be zero), the same across the two PlantDoc
    halves, and exact files shared between PlantVillage and PlantDoc. The
    key-only split, which the grouping replaced, is counted too, so the report
    shows what the content hash adds.
    """
    indexed = indexed or index_and_split(config)
    parts = indexed["parts"]
    train_name = indexed["datasets"]["train"]
    shift_name = indexed["datasets"]["shift"]
    pv_hashes, pd_hashes = indexed["hashes"][train_name], indexed["hashes"][shift_name]

    def by_hash(table):
        return lambda path: table[item_key(path)]

    pv_parts = {k: parts[k] for k in ("train", "val", "test")}
    pd_parts = {k: parts[k] for k in ("shift_fit", "shift_eval")}
    all_pv = [it for k in pv_parts for it in pv_parts[k]]
    data_cfg = config.section("data")
    key_only = _split_by_source(all_pv, float(data_cfg["val_fraction"]),
                                float(data_cfg.get("test_fraction", data_cfg["val_fraction"])),
                                int(config["seed"]), hashes=None)

    pv_values = list(pv_hashes.values())
    report = {
        "plantvillage": {
            "n_images": len(all_pv),
            "n_source_keys": len({_source_key(p) for p, _ in all_pv}),
            "n_unique_contents": len(set(pv_values)),
            "n_files_with_an_identical_copy": sum(
                c for c in _counts(pv_values).values() if c > 1),
            "split_sizes": {k: len(v) for k, v in pv_parts.items()},
            "source_keys_across_splits": _crossings(pv_parts, _source_key),
            "contents_across_splits": _crossings(pv_parts, by_hash(pv_hashes)),
            "key_only_split_contents_across_splits":
                _crossings(key_only, by_hash(pv_hashes)),
        },
        "plantdoc": {
            "n_images": len(parts["shift"]),
            "n_unique_contents": len(set(pd_hashes.values())),
            "split_sizes": {k: len(v) for k, v in pd_parts.items()},
            "source_keys_across_splits": _crossings(pd_parts, _source_key),
            "contents_across_splits": _crossings(pd_parts, by_hash(pd_hashes)),
        },
        "plantvillage_plantdoc_identical_files": len(set(pv_values) & set(pd_hashes.values())),
        "membership": {k: membership_hash(v) for k, v in parts.items()},
    }
    report["cross_split_duplicates"] = (
        report["plantvillage"]["source_keys_across_splits"]
        + report["plantvillage"]["contents_across_splits"]
        + report["plantdoc"]["source_keys_across_splits"]
        + report["plantdoc"]["contents_across_splits"]
    )
    return report


def _counts(values) -> dict:
    return dict(Counter(values))


def build_dataloaders(config) -> dict[str, object]:
    """Return a dataloader per split, plus split metadata under ``"meta"``.

    Splits: ``train``, ``val``, ``test``, ``shift``, ``shift_fit``, ``shift_eval``
    (see ``index_and_split``). Constraints enforced here, each of which is
    expensive to discover late:

    * **Split by source image and by content, not by file** — see ``_group_items``.
    * **Validation fits the temperature**, so a separate ``test`` split carries
      early stopping and reporting; fitting both on one split leaks.
    * **Normalisation is identical across every split.**
    * The shuffles are seeded from the config.

    ``meta["membership"]`` holds each split's membership hash, which logit pools
    carry so a pool built from a different split is detected rather than used.
    """
    from torch.utils.data import DataLoader

    data_cfg = config.section("data")
    image_size = int(data_cfg["image_size"])
    binarise = bool(data_cfg.get("binarise", True))
    batch_size = int(config.section("train")["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 2))

    indexed = index_and_split(config)
    parts = indexed["parts"]
    classes = indexed["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    loaders, meta = {}, {}
    for name, items in parts.items():
        is_train = name == "train"
        ds = _ImageFolderDataset(
            items, _transforms(image_size, train=is_train), class_to_idx, binarise
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
        meta[name] = _describe(items)

    meta["roots"] = indexed["roots"]
    meta["plantvillage_variant"] = indexed["variant"]
    meta["n_fine_classes"] = len(classes)
    meta["membership"] = {name: membership_hash(items) for name, items in parts.items()}

    # The shift set exists to measure degradation under distribution shift. If every
    # one of its images carries the same binary label, the measurement is
    # meaningless and every downstream number derived from it is too — so fail here
    # rather than reporting an accuracy that is really a prevalence. This is exactly
    # the failure the old single-regex `binarise_label` produced on PlantDoc.
    shift_meta = meta["shift"]
    if shift_meta["diseased_fraction"] in (0.0, 1.0):
        raise ValueError(
            f"shift split is single-class: {shift_meta['diseased']} diseased / "
            f"{shift_meta['healthy']} healthy across {shift_meta['n_classes']} class "
            f"directories. The label mapping is wrong for this dataset — check "
            f"PLANTDOC_HEALTHY / PLANTDOC_DISEASED against the directory names at "
            f"{meta['roots']['shift']}."
        )

    loaders["meta"] = meta
    return loaders
