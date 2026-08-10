"""
Path resolution, which differs between the session that *writes* an artefact and
the session that *reads* it.

That asymmetry is the whole reason `find_checkpoint` exists: on Kaggle a run writes
to /kaggle/working, which is per-session, while a later notebook receives the same
file read-only under /kaggle/input. Code that assumes one location works in exactly
one of the two places, which is how notebook 02 came to fail looking for notebook
01's output.
"""

from __future__ import annotations

import pytest

from scoutfield.utils import paths


@pytest.fixture
def working(tmp_path, monkeypatch):
    """Redirect the writable directory, leaving the mount search to each test."""
    root = tmp_path / "working"

    def fake_output_dir(*parts: str):
        out = root.joinpath(*parts)
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(paths, "output_dir", fake_output_dir)
    monkeypatch.setattr(paths, "is_kaggle", lambda: False)
    return root


def test_finds_a_checkpoint_in_the_writable_directory(working):
    target = working / "checkpoints" / "perception"
    target.mkdir(parents=True)
    (target / "best.pt").write_bytes(b"weights")

    assert paths.find_checkpoint("perception", "best.pt") == target / "best.pt"


def test_finds_a_checkpoint_mounted_from_an_earlier_notebook(working, tmp_path,
                                                            monkeypatch):
    """The producing session's /kaggle/working is what gets mounted, so the
    checkpoints/ prefix survives into the mount."""
    mount = tmp_path / "input" / "scoutfield-01-perception-finetune"
    (mount / "checkpoints" / "perception").mkdir(parents=True)
    (mount / "checkpoints" / "perception" / "best.pt").write_bytes(b"weights")

    monkeypatch.setattr(paths, "is_kaggle", lambda: True)
    monkeypatch.setattr(paths, "Path", _rooted_path(tmp_path / "input"))

    found = paths.find_checkpoint("perception", "best.pt")
    assert found.read_bytes() == b"weights"


def test_the_writable_directory_wins_over_a_mount(working, tmp_path, monkeypatch):
    """A checkpoint written by this session is the one this session meant."""
    local = working / "checkpoints" / "perception"
    local.mkdir(parents=True)
    (local / "best.pt").write_bytes(b"this session")

    mount = tmp_path / "input" / "scoutfield-01-perception-finetune"
    (mount / "checkpoints" / "perception").mkdir(parents=True)
    (mount / "checkpoints" / "perception" / "best.pt").write_bytes(b"earlier")

    monkeypatch.setattr(paths, "is_kaggle", lambda: True)
    monkeypatch.setattr(paths, "Path", _rooted_path(tmp_path / "input"))

    assert paths.find_checkpoint("perception", "best.pt").read_bytes() == b"this session"


def test_missing_checkpoint_names_every_location_searched(working):
    """The failure has three usual causes and the message has to let the reader
    tell them apart, so it lists what was actually looked at."""
    with pytest.raises(FileNotFoundError) as excinfo:
        paths.find_checkpoint("perception", "best.pt")

    message = str(excinfo.value)
    assert "Searched:" in message
    assert str(working / "checkpoints" / "perception" / "best.pt") in message


def _rooted_path(mount_root):
    """A Path stand-in that redirects only the /kaggle/input lookup to tmp_path."""
    from pathlib import Path as _Path

    class RootedPath(_Path):
        def __new__(cls, *args, **kwargs):
            if args and str(args[0]) == "/kaggle/input":
                return _Path(mount_root)
            return _Path(*args, **kwargs)

    return RootedPath
