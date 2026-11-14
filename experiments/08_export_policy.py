"""
Roadmap item 9: export the trained PPO policy to ONNX for the demonstration.

    python experiments/08_export_policy.py                      # ppo_seed0.zip from the store
    python experiments/08_export_policy.py --checkpoint <file>  # a local copy
    python experiments/08_export_policy.py --push               # also file it in the store

Settings are in ``configs/deployment.yaml`` under ``policy``. The exported
``policy_ppo_seed0.onnx`` and ``policy_export_record.json`` are written beside the
classifier export, inside the gitignored artefact cache.

After the export, the ONNX policy and the Stable-Baselines3 policy choose an action
from the same observation at every step of PPO's own episodes in the real
environment, until at least ``verify.min_observations`` observations have been
compared. Any differing action makes the script exit non-zero.

``--push`` uploads both files to stage ``08_deployment``. It needs a Hugging Face
write token and is refused for a policy the run did not record, or from a working
tree with uncommitted changes.
"""

from __future__ import annotations

import argparse

from scoutfield.app.export_onnx import DEFAULT_CONFIG
from scoutfield.app.export_policy import run_policy_export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None,
                        help="a local ppo_seed0.zip to export instead of the store's")
    parser.add_argument("--push", action="store_true",
                        help="upload the export to stage 08_deployment of the store")
    args = parser.parse_args()

    record = run_policy_export(args.config, checkpoint=args.checkpoint, push=args.push)
    check = record["verification"]
    print(f"checkpoint sha256 {record['checkpoint']['sha256']}")
    print(f"onnx sha256       {record['onnx']['sha256']} ({record['onnx']['bytes']} bytes)")
    print(f"check: {check['n_observations']} observations, {check['episodes']} episodes, "
          f"actions identical, max |score diff| {check['max_abs_score_diff']:.2e}")
    if not args.push:
        print("not pushed to the store (run again with --push to do that)")


if __name__ == "__main__":
    main()
