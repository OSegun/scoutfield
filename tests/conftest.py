"""Shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def store(tmp_path):
    """An artefact store backed by a temporary directory: no network, no token."""
    from scoutfield.utils.artifacts import ArtifactStore, LocalBackend

    return ArtifactStore(LocalBackend(tmp_path / "hub"), "rtest", tmp_path / "cache")
