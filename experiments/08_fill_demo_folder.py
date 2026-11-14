"""
Fill the demonstration's offline folder, once, while online.

    python experiments/08_fill_demo_folder.py

Copies every file the demonstration reads from the artefact store (the logit pool,
the calibration results, the sweep summary, the exported classifier and policy,
and any sample images filed there) into ``demo.offline_dir`` of
``configs/deployment.yaml``, laid out as the store lays them out, with their
manifest entries. Each file's SHA-256 is checked on the way in and again on every
read. It also copies this machine's laptop-only samples (PlantVillage), if
``experiments/08_select_samples.py`` has made them.

After this, the demonstration runs with the network switched off:

    SCOUTFIELD_DEMO_SOURCE=offline streamlit run scoutfield/app/streamlit_app.py
"""

from __future__ import annotations

import argparse

from scoutfield.app import demo
from scoutfield.app.export_onnx import DEFAULT_CONFIG, load_deployment_config, work_dir
from scoutfield.utils.artifacts import get_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_deployment_config(args.config)
    store = get_store()
    report = demo.fill_offline(store, config,
                               local_samples=work_dir(config, store.run_id) / "samples_local")
    for stage, name in report["files"]:
        print(f"  {stage}/{name}")
    print(f"filled {report['root']} from {store.backend.description}: "
          f"{len(report['files'])} store files, {report['local_samples']} laptop-only samples")
    if report["local_samples"] == 0:
        print("no laptop-only samples: run experiments/08_select_samples.py first to "
              "add the PlantVillage images")


if __name__ == "__main__":
    main()
