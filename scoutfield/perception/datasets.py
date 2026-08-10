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

        if names <= _VARIANT_DIRS:
            chosen = next((d for d in subdirs if d.name.casefold() == variant.casefold()),
                          None)
            if chosen is None:
                raise FileNotFoundError(
                    f"image variant '{variant}' not found under {current}; "
                    f"available: {sorted(d.name for d in subdirs)}"
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
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _split_by_source(items, val_fraction: float, test_fraction: float, seed: int):
    """Partition by source-image key so augmented variants cannot straddle splits."""
    import numpy as np

    groups: dict[str, list] = {}
    for path, cls in items:
        groups.setdefault(_source_key(path), []).append((path, cls))

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


def build_dataloaders(config) -> dict[str, object]:
    """Return train / val / test / shift dataloaders plus split metadata.

    Constraints enforced here, each of which is expensive to discover late:

    * **Split by source image, not by file** — see ``_source_key``.
    * **Validation fits the temperature**, so a separate ``test`` split carries
      early stopping and reporting; fitting both on one split leaks.
    * **Normalisation is identical across all four splits.**
    * The shuffle is seeded from ``config['seed']``.
    """
    from torch.utils.data import DataLoader

    data_cfg = config.section("data")
    seed = int(config["seed"])
    image_size = int(data_cfg["image_size"])
    binarise = bool(data_cfg.get("binarise", True))
    val_fraction = float(data_cfg["val_fraction"])
    # Held out from PlantVillage for early stopping and in-distribution reporting,
    # keeping the validation split exclusively for the temperature fit.
    test_fraction = float(data_cfg.get("test_fraction", val_fraction))
    batch_size = int(config.section("train")["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 2))
    slugs = data_cfg["kaggle_slugs"]
    # Which of PlantVillage's three renderings to train on. Read from config rather
    # than fixed in code, and recorded in the split metadata below, so a run that
    # used a different rendering is identifiable after the fact.
    variant = str(data_cfg.get("plantvillage_variant", "color"))

    train_roots = resolve_dataset_roots(data_cfg["train_dataset"], slugs, variant)
    shift_roots = resolve_dataset_roots(data_cfg["shift_dataset"], slugs, variant)

    train_items = _index_split(train_roots)
    shift_items = _index_split(shift_roots)

    classes = sorted({cls for _, cls in train_items} | {cls for _, cls in shift_items})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    parts = _split_by_source(train_items, val_fraction, test_fraction, seed)
    parts["shift"] = shift_items

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

    meta["roots"] = {
        "train": [str(r) for r in train_roots],
        "shift": [str(r) for r in shift_roots],
    }
    meta["plantvillage_variant"] = variant
    meta["n_fine_classes"] = len(classes)

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
