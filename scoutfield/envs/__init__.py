"""
Environments: the pilot's ScoutEnv, scaled and extended by subclassing.

The pilot's environment is imported, never copied and never edited. What changes here is
expressed as a subclass so the published baseline stays exactly as documented, and the
override is visibly this project's contribution rather than a silent mutation of the
thing being compared against.
"""

from scoutfield.envs.field_env import FieldScoutEnv  # noqa: F401
