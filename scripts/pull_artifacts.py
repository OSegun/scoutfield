"""
Mirror one pipeline run from the artefact store to a local folder.

    python scripts/pull_artifacts.py --run r2026-11-17
    python scripts/pull_artifacts.py --run r2026-11-17 --dest D:/scoutfield-runs

Default destination: ``../artifacts/<run_id>/``, beside the repository rather than
inside it, so nothing pulled can be committed by accident. Every file recorded in the
run's manifest is downloaded and checked against its SHA-256; files uploaded but never
recorded (an interrupted stage) are skipped. The manifest is copied too and summarised
on screen: per stage, the commit that produced it and the inputs it read.

Reads need no token: the store is public.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoutfield.utils.artifacts import get_store
from scoutfield.utils.paths import repo_root


def mirror(run_id: str | None, dest: Path | None) -> Path:
    store = get_store(run_id)
    dest = dest or repo_root().parent / "artifacts" / store.run_id
    manifest = store.manifest()
    stages = manifest.get("stages", {})
    if not stages:
        raise SystemExit(f"run {store.run_id} has no manifest in {store.backend.description}")

    for stage, entry in sorted(stages.items()):
        for name in sorted(entry.get("files", {})):
            local = store.pull(stage, name)
            target = dest / stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, target)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"run {store.run_id} -> {dest}")
    for stage, entry in sorted(stages.items()):
        files = entry.get("files", {})
        size = sum(f["size"] for f in files.values())
        print(f"\n{stage}: {len(files)} files, {size / 1e6:.1f} MB, "
              f"git {str(entry.get('git_commit'))[:10]}, updated {entry.get('updated_at')}")
        for name in sorted(files):
            print(f"    {name}  sha256 {files[name]['sha256'][:12]}")
        for item in entry.get("inputs", []):
            print(f"    read {item['stage']}/{item['name']}  sha256 {item['sha256'][:12]}  "
                  f"store commit {str(item.get('hub_commit'))[:10]}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="run_id; default: configs/artifacts.yaml")
    parser.add_argument("--dest", default=None, type=Path)
    args = parser.parse_args()
    mirror(args.run, args.dest)


if __name__ == "__main__":
    main()
