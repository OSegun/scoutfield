"""
Roadmap item 9: export the trained classifier to ONNX.

    python experiments/08_export_onnx.py                      # best.pt from the store
    python experiments/08_export_onnx.py --checkpoint <file>  # a local copy, offline
    python experiments/08_export_onnx.py --push               # also file it in the store

Settings are in ``configs/deployment.yaml``. The exported ``classifier.onnx`` and
``export_record.json`` are written under ``work_dir`` there (inside the gitignored
artefact cache, one folder per run_id), never into the repository.

After the export, ONNX and PyTorch are run on the same inputs. The hard predictions
must be identical and the logits must agree within ``export.verify.logit_atol``;
otherwise the script exits non-zero and the file is not usable.

``--push`` uploads both files to stage ``08_deployment`` of the artefact store. It
needs a Hugging Face write token, and it is refused when the checkpoint is not the
one the run recorded or when the working tree has uncommitted changes.
"""

from __future__ import annotations

import argparse

from scoutfield.app.export_onnx import DEFAULT_CONFIG, run_export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None,
                        help="a local best.pt to export instead of the one in the store")
    parser.add_argument("--push", action="store_true",
                        help="upload the export to stage 08_deployment of the store")
    args = parser.parse_args()

    record = run_export(args.config, checkpoint=args.checkpoint, push=args.push)
    check = record["verification"]
    print(f"checkpoint sha256 {record['checkpoint']['sha256']}")
    print(f"onnx sha256       {record['onnx']['sha256']} ({record['onnx']['bytes']} bytes, "
          f"opset {record['onnx']['opset']})")
    print(f"check: {check['n_images']} inputs, predictions identical, "
          f"max |logit diff| {check['max_abs_diff']:.2e} <= {check['atol']:.0e}")
    if not args.push:
        print("not pushed to the store (run again with --push to do that)")


if __name__ == "__main__":
    main()
