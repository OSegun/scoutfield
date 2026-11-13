"""
The shared artefact store: one place every stage reads its inputs from and writes
its outputs to.

Why a store rather than notebook mounts
---------------------------------------
On Kaggle a notebook writes to ``/kaggle/working``, which belongs to one session,
and a later notebook receives that output read-only under ``/kaggle/input`` at a
depth that depends on how it was attached. Code that looks for an upstream file in
one place works in exactly the environments where the file happens to be there,
and falls back silently everywhere else. Routing every cross-stage file through
one addressed location removes that whole class of error: a file is either at
``runs/<run_id>/<stage>/<name>`` in the store, or the read raises and says where
it looked.

Layout
------
::

    runs/<run_id>/manifest.json
    runs/<run_id>/<stage>/<name>

``manifest.json`` records, per stage, every file's SHA-256 and size, the git
commit of this repository that produced it, the store commit of the upload, the
Kaggle kernel when there is one, a UTC timestamp, and the upstream files the stage
read (with their SHA-256 and store commit). Every result therefore traces to the
exact inputs it was computed from.

Backends
--------
``HubBackend`` talks to the public Hugging Face dataset repository named in
``configs/artifacts.yaml``. Reads need no token; writes read one from the
``HF_TOKEN`` environment variable or the token stored by ``huggingface-cli login``.

``LocalBackend`` stands in for the hub with a plain directory. Tests use it so CI
needs no network or token, and setting ``SCOUTFIELD_ARTIFACT_STORE=<dir>`` makes
every stage use it, which is how the pipeline runs fully offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Stage names, in pipeline order. A push or pull naming anything else is a typo
#: that would otherwise create a new, unnoticed folder in the store.
STAGES = (
    "01_perception",
    "02_calibration",
    "03_ppo",
    "04_sweep",
    "05_tau",
    "06_figures",
    "07_uncertainty",
    "08_deployment",
)

#: Environment variable that switches every stage to a local directory store.
LOCAL_STORE_ENV = "SCOUTFIELD_ARTIFACT_STORE"
#: Set by the notebook bootstrap cell so the manifest can name the kernel.
KAGGLE_KERNEL_ENV = "SCOUTFIELD_KAGGLE_KERNEL"
#: Kaggle does not expose a run's version number to the running kernel in any way
#: this project has verified, so it is recorded only when something supplies it.
KAGGLE_VERSION_ENV = "SCOUTFIELD_KAGGLE_KERNEL_VERSION"

MANIFEST = "manifest.json"


class StoreWriteError(PermissionError):
    """A write was attempted without the credentials it needs."""


# ---------------------------------------------------------------- helpers

def sha256_of(path: str | Path) -> str:
    """SHA-256 of a file, streamed so a checkpoint does not have to fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    """The commit of this repository that is running, or None outside a checkout.

    Shared by every stage so a results file and the manifest entry for it record
    the same value; the stage verifier compares the two.
    """
    from scoutfield.utils.paths import repo_root

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(),
                             capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; stages are {', '.join(STAGES)}")


# --------------------------------------------------------------- backends

class LocalBackend:
    """A directory standing in for the hub: same paths, no network, no token."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.description = f"local store at {self.root}"

    def download(self, remote: str, dest_dir: Path) -> Path | None:
        src = self.root / remote
        if not src.is_file():
            return None
        dest = dest_dir / remote
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest

    def upload(self, files: dict[str, Path | bytes], message: str) -> str:
        digest = hashlib.sha1(message.encode("utf-8"))
        for remote, src in sorted(files.items()):
            dest = self.root / remote
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = src if isinstance(src, bytes) else Path(src).read_bytes()
            dest.write_bytes(data)
            digest.update(remote.encode("utf-8"))
            digest.update(data)
        # A content-derived id, so two different uploads never share a "commit".
        log = self.root / ".commits"
        count = len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0
        digest.update(str(count).encode("utf-8"))
        commit = digest.hexdigest()
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"{commit} {message}\n")
        return commit

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(p.relative_to(self.root).as_posix()
                      for p in base.rglob("*") if p.is_file())


class HubBackend:
    """The public Hugging Face dataset repository."""

    def __init__(self, repo_id: str, repo_type: str = "dataset"):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi()
        self.description = f"https://huggingface.co/datasets/{repo_id}"

    def download(self, remote: str, dest_dir: Path) -> Path | None:
        from huggingface_hub import errors, hf_hub_download

        missing = tuple(getattr(errors, name) for name in
                        ("EntryNotFoundError", "RemoteEntryNotFoundError")
                        if hasattr(errors, name))
        try:
            return Path(hf_hub_download(self.repo_id, remote, repo_type=self.repo_type,
                                        local_dir=dest_dir))
        except missing:
            return None

    def upload(self, files: dict[str, Path | bytes], message: str) -> str:
        from huggingface_hub import CommitOperationAdd, get_token

        token = get_token()
        if not token:
            raise StoreWriteError(
                f"writing to {self.description} needs a Hugging Face write token, and "
                "none was found. On Kaggle: Add-ons -> Secrets, attach HF_TOKEN to this "
                "notebook, and re-run from the first cell. On the laptop: run "
                "`huggingface-cli login`, or set the HF_TOKEN environment variable."
            )
        operations = [
            CommitOperationAdd(path_in_repo=remote,
                               path_or_fileobj=src if isinstance(src, bytes) else str(src))
            for remote, src in sorted(files.items())
        ]
        info = self.api.create_commit(self.repo_id, operations, commit_message=message,
                                      repo_type=self.repo_type, token=token)
        return info.oid

    def list(self, prefix: str) -> list[str]:
        files = self.api.list_repo_files(self.repo_id, repo_type=self.repo_type)
        return sorted(f for f in files if f.startswith(prefix))


# ------------------------------------------------------------------ store

class ArtifactStore:
    """Pull and push one run's files, keeping its manifest in step."""

    def __init__(self, backend, run_id: str, cache_dir: str | Path):
        if not run_id:
            raise ValueError("run_id is empty; set it in configs/artifacts.yaml")
        self.backend = backend
        self.run_id = str(run_id)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        #: Every file this process pulled, so a stage can record its inputs without
        #: listing them twice.
        self.reads: dict[tuple[str, str], dict[str, Any]] = {}

    # --- paths

    def remote_path(self, stage: str, name: str) -> str:
        return f"runs/{self.run_id}/{stage}/{name}"

    @property
    def manifest_path(self) -> str:
        return f"runs/{self.run_id}/{MANIFEST}"

    def url(self, stage: str, name: str) -> str:
        """A browser address for one file, for error messages and reports."""
        base = getattr(self.backend, "description", "")
        if base.startswith("https://"):
            return f"{base}/resolve/main/{self.remote_path(stage, name)}"
        return f"{base}: {self.remote_path(stage, name)}"

    # --- manifest

    def manifest(self) -> dict[str, Any]:
        """The run's manifest, fetched fresh, or an empty one for a new run."""
        local = self.backend.download(self.manifest_path, self.cache_dir)
        if local is None:
            return {"run_id": self.run_id, "stages": {}}
        with Path(local).open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_manifest(self, manifest: dict[str, Any], message: str) -> str:
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        return self.backend.upload({self.manifest_path: payload}, message)

    def stage_entry(self, stage: str) -> dict[str, Any]:
        _check_stage(stage)
        return self.manifest().get("stages", {}).get(stage, {})

    def has(self, stage: str, name: str) -> bool:
        return name in self.stage_entry(stage).get("files", {})

    # --- reads

    def pull(self, stage: str, name: str, required: bool = True) -> Path | None:
        """Download ``runs/<run_id>/<stage>/<name>`` and verify it against the manifest.

        The manifest, not the presence of a file, is the record: a file uploaded
        but never entered in the manifest was not finished, and is treated as
        missing. A file whose SHA-256 differs from its manifest entry raises,
        because it is not the file the run recorded.
        """
        _check_stage(stage)
        entry = self.stage_entry(stage).get("files", {}).get(name)
        remote = self.remote_path(stage, name)
        if entry is None:
            if not required:
                return None
            raise FileNotFoundError(
                f"{name} is not in the store for run {self.run_id}. Looked for "
                f"{remote} in {self.backend.description} and in its manifest. Run "
                f"stage {stage} for this run_id first, or check run_id in "
                f"configs/artifacts.yaml."
            )
        local = self.backend.download(remote, self.cache_dir)
        if local is None:
            raise FileNotFoundError(
                f"{remote} is listed in the manifest of run {self.run_id} but is "
                f"missing from {self.backend.description}. Re-run stage {stage}."
            )
        actual = sha256_of(local)
        if actual != entry["sha256"]:
            raise ValueError(
                f"sha256 mismatch for {remote}: the manifest records {entry['sha256']}, "
                f"the downloaded file is {actual}. The file was changed after the run "
                f"recorded it; re-run stage {stage} rather than using it."
            )
        self.reads[(stage, name)] = {"stage": stage, "name": name,
                                     "sha256": entry["sha256"],
                                     "hub_commit": entry.get("hub_commit")}
        return Path(local)

    def pull_json(self, stage: str, name: str, required: bool = True) -> Any:
        path = self.pull(stage, name, required=required)
        if path is None:
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # --- writes

    def push(self, stage: str, paths: Iterable[str | Path], message: str,
             names: dict[str, str] | None = None) -> str:
        """Upload files under ``runs/<run_id>/<stage>/`` in one commit, then record them.

        ``names`` optionally renames a local file in the store (local path to
        store name); by default the file keeps its own name. Returns the store
        commit of the upload.
        """
        _check_stage(stage)
        names = {str(Path(k)): v for k, v in (names or {}).items()}
        files: dict[str, Path] = {}
        for p in paths:
            p = Path(p)
            if not p.is_file():
                raise FileNotFoundError(f"cannot push {p}: no such file")
            files[names.get(str(p), p.name)] = p
        if not files:
            raise ValueError("push called with no files")

        upload = {self.remote_path(stage, n): p for n, p in files.items()}
        commit = self.backend.upload(upload, f"{self.run_id} {stage}: {message}")

        manifest = self.manifest()
        entry = manifest.setdefault("stages", {}).setdefault(stage, {})
        recorded = entry.setdefault("files", {})
        now = _utc_now()
        for n, p in files.items():
            recorded[n] = {"sha256": sha256_of(p), "size": p.stat().st_size,
                           "hub_commit": commit, "pushed_at": now}
        entry["git_commit"] = git_commit()
        entry["kaggle"] = _kaggle_run()
        entry["updated_at"] = now
        manifest["run_id"] = self.run_id
        self._write_manifest(manifest, f"{self.run_id} {stage}: manifest after {message}")
        return commit

    def record_inputs(self, stage: str,
                      inputs: Iterable[tuple[str, str]] | None = None) -> list[dict]:
        """Write into the manifest which upstream files ``stage`` read.

        ``inputs`` is a list of ``(stage, name)`` pairs; by default every file this
        process pulled is recorded. Each entry carries the file's SHA-256 and the
        store commit it was uploaded in, which is what makes a result traceable to
        the exact inputs it used.
        """
        _check_stage(stage)
        manifest = self.manifest()
        if inputs is None:
            records = list(self.reads.values())
        else:
            records = []
            for up_stage, name in inputs:
                up = manifest.get("stages", {}).get(up_stage, {}).get("files", {}).get(name)
                if up is None:
                    raise FileNotFoundError(
                        f"cannot record {up_stage}/{name} as an input of {stage}: it is "
                        f"not in the manifest of run {self.run_id}"
                    )
                records.append({"stage": up_stage, "name": name, "sha256": up["sha256"],
                                "hub_commit": up.get("hub_commit")})
        entry = manifest.setdefault("stages", {}).setdefault(stage, {})
        existing = {(r["stage"], r["name"]): r for r in entry.get("inputs", [])}
        existing.update({(r["stage"], r["name"]): r for r in records})
        entry["inputs"] = sorted(existing.values(), key=lambda r: (r["stage"], r["name"]))
        self._write_manifest(manifest, f"{self.run_id} {stage}: record inputs")
        return entry["inputs"]

    def list_files(self) -> list[str]:
        return self.backend.list(f"runs/{self.run_id}/")


def _kaggle_run() -> dict[str, Any] | None:
    from scoutfield.utils.paths import is_kaggle

    if not is_kaggle():
        return None
    return {
        "kernel": os.environ.get(KAGGLE_KERNEL_ENV),
        "version": os.environ.get(KAGGLE_VERSION_ENV),
        "run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    }


# ----------------------------------------------------------------- access

def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """``configs/artifacts.yaml`` as a dict."""
    from scoutfield.config import load_config
    from scoutfield.utils.paths import repo_root

    return dict(load_config(path or repo_root() / "configs" / "artifacts.yaml").data)


def get_store(run_id: str | None = None, settings: dict[str, Any] | None = None
              ) -> ArtifactStore:
    """The store every stage uses, built from ``configs/artifacts.yaml``.

    ``SCOUTFIELD_ARTIFACT_STORE=<dir>`` swaps the hub for a local directory.
    """
    from scoutfield.utils.paths import repo_root

    settings = settings or load_settings()
    cache = Path(settings.get("local_cache", ".artifact_cache"))
    if not cache.is_absolute():
        cache = repo_root() / cache
    local = os.environ.get(LOCAL_STORE_ENV)
    if local:
        backend = LocalBackend(local)
    else:
        backend = HubBackend(settings["hub_repo"], settings.get("repo_type", "dataset"))
    run = run_id or settings["run_id"]
    return ArtifactStore(backend, run, cache / "runs_cache" / run)
