"""
Dataset resolution and label mapping.

These tests exist because both failures they cover are silent. A wrong binary
label and a mis-resolved directory both produce a dataset that loads cleanly,
trains without complaint, and reports a plausible number that is wrong.

The specific regression: PlantDoc names no class "healthy", so collapsing labels
on that word alone marked every PlantDoc image diseased. Accuracy under shift then
reads as the diseased prevalence, and the project's only distribution-shift
measurement becomes meaningless without ever raising.

No torch, no images on disk beyond empty files — these run in CI in milliseconds.
"""

from __future__ import annotations

import pytest

from scoutfield.perception.datasets import (
    PLANTDOC_DISEASED,
    PLANTDOC_HEALTHY,
    _descend_to_class_dirs,
    _index_split,
    binarise_label,
)

# Verified against the canonical release (Singh et al., 2020):
# github.com/pratikkayal/PlantDoc-Dataset — train/ holds 28 class directories,
# test/ holds 27, omitting "Tomato two spotted spider mites leaf".
PLANTDOC_TRAIN_CLASSES = (
    "Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf spot",
    "Bell_pepper leaf", "Blueberry leaf", "Cherry leaf", "Corn Gray leaf spot",
    "Corn leaf blight", "Corn rust leaf", "Peach leaf", "Potato leaf early blight",
    "Potato leaf late blight", "Raspberry leaf", "Soyabean leaf",
    "Squash Powdery mildew leaf", "Strawberry leaf", "Tomato Early blight leaf",
    "Tomato Septoria leaf spot", "Tomato leaf bacterial spot", "Tomato leaf late blight",
    "Tomato leaf mosaic virus", "Tomato leaf yellow virus", "Tomato leaf",
    "Tomato mold leaf", "Tomato two spotted spider mites leaf", "grape leaf black rot",
    "grape leaf",
)


def _make_tree(base, layout):
    """Create directories, and one empty .jpg inside each leaf directory."""
    for path in layout:
        d = base / path
        d.mkdir(parents=True, exist_ok=True)
        (d / "img_0.jpg").touch()
    return base


# ------------------------------------------------------------------ labels

def test_every_plantdoc_class_is_mapped():
    """The whole set is covered, so no directory falls through to a default."""
    for cls in PLANTDOC_TRAIN_CLASSES:
        assert binarise_label(cls) in (0, 1)


def test_plantdoc_is_not_all_diseased():
    """The regression itself.

    PlantDoc contains healthy classes; none of them says "healthy". If this
    returns all 1s the shift measurement is worthless and nothing warns you.
    """
    labels = [binarise_label(c) for c in PLANTDOC_TRAIN_CLASSES]
    assert 0 in labels, "no PlantDoc class mapped to healthy — the old regex bug is back"
    assert sum(labels) == 18
    assert labels.count(0) == 10


@pytest.mark.parametrize("cls", ["Apple leaf", "Tomato leaf", "grape leaf",
                                 "Bell_pepper leaf", "Soyabean leaf"])
def test_plantdoc_healthy_classes(cls):
    assert binarise_label(cls) == 0


@pytest.mark.parametrize("cls", ["Apple Scab Leaf", "Tomato mold leaf",
                                 "grape leaf black rot", "Bell_pepper leaf spot",
                                 "Corn Gray leaf spot"])
def test_plantdoc_diseased_classes(cls):
    assert binarise_label(cls) == 1


def test_the_two_plantdoc_sets_are_disjoint():
    """An overlap would make the label depend on lookup order."""
    assert not (PLANTDOC_HEALTHY & PLANTDOC_DISEASED)


def test_healthy_and_diseased_differ_only_by_the_condition():
    """`<Crop> leaf` is healthy; the same crop with a condition appended is not.

    This pair is the one a substring match gets wrong in either direction.
    """
    assert binarise_label("Bell_pepper leaf") == 0
    assert binarise_label("Bell_pepper leaf spot") == 1
    assert binarise_label("grape leaf") == 0
    assert binarise_label("grape leaf black rot") == 1


@pytest.mark.parametrize("cls,expected", [
    ("Tomato___healthy", 0),
    ("Apple___healthy", 0),
    ("Tomato___Late_blight", 1),
    ("Apple___Black_rot", 1),
    ("Corn_(maize)___Northern_Leaf_Blight", 1),
])
def test_plantvillage_convention(cls, expected):
    assert binarise_label(cls) == expected


def test_separator_and_case_insensitive():
    """Mirrors that swap underscores for spaces or change case still resolve."""
    assert binarise_label("apple  leaf") == 0
    assert binarise_label("Apple_leaf") == 0
    assert binarise_label("GRAPE LEAF BLACK ROT") == 1


def test_unknown_class_raises_rather_than_defaulting():
    """Loud failure beats a silent wrong label — that is the entire point."""
    with pytest.raises(KeyError, match="unrecognised class directory"):
        binarise_label("Banana sigatoka leaf")


# ------------------------------------------------------- directory resolution

def test_descends_through_single_child_wrappers(tmp_path):
    _make_tree(tmp_path, ["a/b/Tomato___healthy", "a/b/Tomato___Late_blight"])
    roots = _descend_to_class_dirs(tmp_path)
    assert [r.name for r in roots] == ["b"]


def test_plantvillage_variant_is_selected_not_merged(tmp_path):
    """color / grayscale / segmented are three renderings of the same photographs.

    Taking all three triples every leaf and lets one photograph's renderings
    straddle the train/val split.
    """
    _make_tree(tmp_path, [
        "plantvillage dataset/color/Tomato___healthy",
        "plantvillage dataset/grayscale/Tomato___healthy",
        "plantvillage dataset/segmented/Tomato___healthy",
    ])
    roots = _descend_to_class_dirs(tmp_path, variant="color")
    assert len(roots) == 1 and roots[0].name == "color"

    roots = _descend_to_class_dirs(tmp_path, variant="segmented")
    assert roots[0].name == "segmented"


def test_unknown_variant_raises(tmp_path):
    _make_tree(tmp_path, ["ds/color/Tomato___healthy", "ds/segmented/Tomato___healthy"])
    with pytest.raises(FileNotFoundError, match="image variant"):
        _descend_to_class_dirs(tmp_path, variant="rgb")


def test_plantdoc_train_and_test_are_merged(tmp_path):
    """PlantDoc is the shift set in full; using half of it discards data."""
    _make_tree(tmp_path, ["train/Apple leaf", "train/Apple Scab Leaf",
                          "test/Apple leaf", "test/Apple Scab Leaf"])
    roots = _descend_to_class_dirs(tmp_path)
    assert sorted(r.name for r in roots) == ["test", "train"]

    items = _index_split(roots)
    assert len(items) == 4
    assert sorted({cls for _, cls in items}) == ["Apple Scab Leaf", "Apple leaf"]


def test_class_dirs_at_the_mount_point_are_used_directly(tmp_path):
    _make_tree(tmp_path, ["Apple leaf", "Apple Scab Leaf", "Tomato leaf"])
    roots = _descend_to_class_dirs(tmp_path)
    assert [r.name for r in roots] == [tmp_path.name]


def test_index_split_is_case_insensitive_about_extensions(tmp_path):
    d = tmp_path / "Apple leaf"
    d.mkdir(parents=True)
    for name in ("a.jpg", "b.JPG", "c.PNG", "d.txt"):
        (d / name).touch()
    items = _index_split(tmp_path)
    assert len(items) == 3


def test_index_split_raises_on_an_empty_tree(tmp_path):
    (tmp_path / "Apple leaf").mkdir()
    with pytest.raises(FileNotFoundError, match="no images found"):
        _index_split(tmp_path)
