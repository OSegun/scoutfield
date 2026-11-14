"""
Choose the demonstration's sample images by rule, from the stage 02 records.

    python experiments/08_select_samples.py --plan    # which images, and how to fetch them
    python experiments/08_select_samples.py           # writes the local sample folders
    python experiments/08_select_samples.py --push    # also files the PlantDoc ones

Settings are in ``configs/deployment.yaml`` under ``demo.samples``. The images come
from the datasets on this machine (``data/<dataset>``, slugs in
``configs/perception.yaml``); none is committed to the repository.

* PlantDoc ``shift_eval`` (the pool the planners observe through): correctly
  classified diseased and healthy images at evenly spaced quantiles of confidence,
  and the classifier's most confident mistakes. Licensed CC BY 4.0, so these may be
  filed in the public store (``--push``) for the hosted demonstration, credited on
  the page.
* PlantVillage ``test`` (lab imagery, for contrast): one per class. Licensed CC
  BY-NC-SA 4.0, so these stay in the laptop's offline folder only.

Each image is listed with the logit the classifier gave it in stage 02, its true
class and the licence text.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scoutfield.app import demo
from scoutfield.app.export_onnx import DEFAULT_CONFIG, load_deployment_config, work_dir
from scoutfield.config import load_config
from scoutfield.perception.adapter import logit_pool_filename
from scoutfield.utils.artifacts import get_store
from scoutfield.utils.paths import data_dir, repo_root


def _dataset_root(name: str) -> Path:
    slug = load_config(repo_root() / "configs" / "perception.yaml") \
        .section("data")["kaggle_slugs"][name]
    root = data_dir(slug)
    if not root.exists():
        raise FileNotFoundError(
            f"dataset {name!r} not found at {root}. Run this script with --plan to "
            f"download only the chosen images from the Kaggle dataset {slug}, or unpack "
            f"the whole dataset there, then run it again."
        )
    return root


def _write(folder: Path, picks: list[dict], dataset: str, root: Path, licence: str,
           prefix: str) -> list[dict]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    listed = []
    for i, pick in enumerate(picks, start=1):
        src = demo.find_image(root, pick["key"])
        store_name = f"{prefix}_{i:02d}{src.suffix.lower()}"
        shutil.copyfile(src, folder / store_name)
        listed.append({"store_name": store_name, "dataset": dataset, "key": pick["key"],
                       "true_label": "diseased" if pick["label"] else "healthy",
                       "class_folder": Path(pick["key"]).parts[-2],
                       "logit": pick["logit"], "correct": pick["correct"],
                       "licence": licence})
    return listed


def _print_plan(chosen: dict) -> None:
    """One Kaggle command per chosen image, saving it where ``find_image`` looks.

    Only these few files are needed, not the whole datasets. Each lands in
    ``data/<dataset folder>/<class folder>/``, so its last three path components
    are the image key recorded in the pool.
    """
    from posixpath import join

    slugs = load_config(repo_root() / "configs" / "perception.yaml")         .section("data")["kaggle_slugs"]
    for name, (picks, prefix) in chosen.items():
        slug = slugs[name]
        target = data_dir(slug).relative_to(repo_root())
        print(f"# {name}: {len(picks)} images from Kaggle {slug}")
        for pick in picks:
            key = Path(pick["key"])
            remote = join(prefix, key.as_posix()) if prefix else key.as_posix()
            dest = (target / key.parent).as_posix()
            print(f'kaggle datasets download -d {slug} -f "{remote}" -p "{dest}"')
    print("# then run this script again without --plan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--push", action="store_true",
                        help="file the PlantDoc samples in stage 08_deployment")
    parser.add_argument("--plan", action="store_true",
                        help="choose the images and print a Kaggle command to download "
                             "each one, instead of needing the whole datasets")
    parser.add_argument("--plantdoc-prefix", default="",
                        help="folder(s) inside the Kaggle PlantDoc dataset above the "
                             "image keys, if any, for --plan")
    parser.add_argument("--plantvillage-prefix", default="",
                        help="folder(s) inside the Kaggle PlantVillage dataset above the "
                             "image keys, if any, for --plan")
    args = parser.parse_args()

    config = load_deployment_config(args.config)
    cfg = config.section("demo")["samples"]
    store = get_store()
    settings = demo.load_settings()
    out = work_dir(config, store.run_id)

    plantdoc = demo.select_samples(
        store.pull("02_calibration", logit_pool_filename(settings.pool_split)),
        int(cfg["plantdoc_correct_diseased"]), int(cfg["plantdoc_correct_healthy"]),
        int(cfg["plantdoc_wrong"]))
    n = int(cfg["plantvillage_per_class"])
    plantvillage = demo.select_samples(store.pull("02_calibration", logit_pool_filename("test")),
                                       n, n, 0)

    if args.plan:
        _print_plan({"plantdoc": (plantdoc, args.plantdoc_prefix),
                     "plantvillage": (plantvillage, args.plantvillage_prefix)})
        return

    public = _write(out / "samples_public", plantdoc, "plantdoc", _dataset_root("plantdoc"),
                    str(cfg["licences"]["plantdoc"]), "plantdoc")

    local = _write(out / "samples_local", plantvillage, "plantvillage",
                   _dataset_root("plantvillage"), str(cfg["licences"]["plantvillage"]),
                   "plantvillage")

    for folder, listed in ((out / "samples_public", public), (out / "samples_local", local)):
        with (folder / cfg["file"]).open("w", encoding="utf-8") as fh:
            json.dump({"run_id": store.run_id, "rule": "configs/deployment.yaml demo.samples",
                       "samples": listed}, fh, indent=2)
        print(f"{folder}: {len(listed)} images")
    for s in public + local:
        print(f"  {s['store_name']:18s} {s['true_label']:9s} logit {s['logit']:+7.3f} "
              f"{'correct' if s['correct'] else 'WRONG':7s} {s['key']}")

    if args.push:
        folder = out / "samples_public"
        store.push(str(config["stage"]), sorted(folder.iterdir()),
                   "PlantDoc sample images for the demonstration (CC BY 4.0)")
        print(f"pushed {len(public)} PlantDoc images and {cfg['file']} to "
              f"{config['stage']} of run {store.run_id}")
    else:
        print("not pushed to the store (run again with --push to file the PlantDoc ones)")


if __name__ == "__main__":
    main()
