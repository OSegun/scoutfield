"""
Dataset path resolution.

A dataset reaches a Kaggle notebook in one of two layouts, attached as an input
(``/kaggle/input/<name>``) or downloaded with kagglehub
(``<cache>/datasets/<owner>/<name>``), and ``data_dir`` has to find either.
Files that cross from one stage to another do not use these paths at all; they go
through the artefact store, tested in tests/test_artifacts.py.
"""

from __future__ import annotations

from scoutfield.utils import paths


def _rooted(monkeypatch, base):
    monkeypatch.setattr(paths, "is_kaggle", lambda: False)
    monkeypatch.setattr(paths, "repo_root", lambda: base)


def test_data_dir_finds_an_attached_dataset(tmp_path, monkeypatch):
    _rooted(monkeypatch, tmp_path)
    (tmp_path / "data" / "plantdoc-dataset").mkdir(parents=True)
    found = paths.data_dir("nirmalsankalana/plantdoc-dataset")
    assert found == tmp_path / "data" / "plantdoc-dataset"


def test_data_dir_finds_a_kagglehub_download(tmp_path, monkeypatch):
    _rooted(monkeypatch, tmp_path)
    nested = tmp_path / "data" / "datasets" / "nirmalsankalana" / "plantdoc-dataset"
    nested.mkdir(parents=True)
    assert paths.data_dir("nirmalsankalana/plantdoc-dataset") == nested


def test_data_dir_names_the_canonical_location_when_absent(tmp_path, monkeypatch):
    _rooted(monkeypatch, tmp_path)
    assert paths.data_dir("owner/missing") == tmp_path / "data" / "missing"
