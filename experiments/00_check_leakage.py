"""
Leakage check: can the same image sit on both sides of a split?

    python experiments/00_check_leakage.py --config configs/perception.yaml

Runs before fine-tuning, in notebook 01. Every PlantVillage colour image and every
PlantDoc image is hashed (SHA-1 of the file bytes, cached), and the report counts:

* source keys, and identical files, found in more than one of train / val / test;
* the same across PlantDoc's ``shift_fit`` and ``shift_eval`` halves;
* identical files shared between PlantVillage and PlantDoc;
* for comparison, identical files that a split by source key alone would leave on
  both sides, which is what grouping by content hash prevents.

The split groups images by source key OR content hash, so the cross-split counts are
zero by construction; the check asserts it rather than assuming it. A shared file
between PlantVillage and PlantDoc does not stop the run, because PlantDoc is never
trained on, but it is reported.

Writes ``results/leakage_report.json`` and pushes it to stage ``01_perception``.
"""

from __future__ import annotations

import argparse
import json

from scoutfield.config import load_config
from scoutfield.perception.datasets import leakage_report
from scoutfield.utils.artifacts import get_store, git_commit
from scoutfield.utils.paths import results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/perception.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = leakage_report(cfg)
    report["git_commit"] = git_commit()
    report["config"] = str(cfg.path)

    path = results_dir() / "leakage_report.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    pv, pd = report["plantvillage"], report["plantdoc"]
    print(f"PlantVillage: {pv['n_images']} images, {pv['n_unique_contents']} distinct "
          f"files, split {pv['split_sizes']}")
    print(f"  across train/val/test: {pv['source_keys_across_splits']} source keys, "
          f"{pv['contents_across_splits']} identical files "
          f"(a key-only split would leave {pv['key_only_split_contents_across_splits']})")
    print(f"PlantDoc: {pd['n_images']} images, split {pd['split_sizes']}")
    print(f"  across shift_fit/shift_eval: {pd['source_keys_across_splits']} source keys, "
          f"{pd['contents_across_splits']} identical files")
    print(f"identical files in both datasets: {report['plantvillage_plantdoc_identical_files']}")

    store = get_store()
    store.push("01_perception", [path], "leakage report")
    print(f"pushed {path.name} to run {store.run_id}")

    if report["cross_split_duplicates"]:
        raise SystemExit(f"{report['cross_split_duplicates']} cross-split duplicates: "
                         "the split is not leak-free. Do not fine-tune on it.")


if __name__ == "__main__":
    main()
