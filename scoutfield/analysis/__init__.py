"""
Statistical analysis of experiment output.

Library code, not a driver: functions here are pure and take arrays, so they can be
unit-tested and reused from a notebook. The scripts that read CSVs and write summary
files live in ``experiments/``.
"""

from scoutfield.analysis.stats import iqm, stratified_bootstrap_ci  # noqa: F401
