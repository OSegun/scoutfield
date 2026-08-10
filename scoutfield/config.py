"""
Configuration loading.

No hyperparameter is defined in code. Every value comes from a YAML file under
``configs/`` so that a run is fully described by ``(config file, git commit)``.
This is the same discipline the pilot enforces through ``experiment.py`` being
the single source of truth for its parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    """A loaded config plus the path it came from, so runs are traceable."""

    data: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def section(self, name: str) -> dict[str, Any]:
        """Return a required top-level section, failing loudly if absent.

        Fails loudly rather than returning ``{}`` because a silently empty
        section means every downstream value falls back to a default that was
        never reviewed — the commonest way an experiment ends up not matching
        the config it claims to have used.
        """
        if name not in self.data:
            raise KeyError(f"config {self.path.name} has no section '{name}'")
        return self.data[name]


def load_config(path: str | Path) -> Config:
    """Load a YAML config from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config {p} did not parse to a mapping")
    return Config(data=data, path=p)


# TODO(implementation): add a `--set key.subkey=value` CLI override that records
# the override in the run's summary, so an ad-hoc run is still reproducible.
