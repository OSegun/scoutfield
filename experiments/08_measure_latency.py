"""
Roadmap item 9: measure the exported classifier's inference latency.

    python experiments/08_export_onnx.py          # first, once
    python experiments/08_measure_latency.py      # writes deployment_latency.json
    python experiments/08_measure_latency.py --push

Settings are in ``configs/deployment.yaml`` under ``latency``: CPU execution
provider, batch size 1, warm-up calls discarded, then timed calls at each thread
count. Median and 95th percentile are reported, with the machine read from the
system: CPU model, core counts, RAM, operating system, whether it was on mains
power, and the onnxruntime and Python versions.

The measurement is of the model call alone, on whatever machine runs this script.
It is not a measurement of drone hardware.

``--push`` files ``deployment_latency.json`` in stage ``08_deployment`` beside the
exported model it measured. It needs a Hugging Face write token and is refused
unless that model is already filed there and the working tree is committed. The
promotion step then copies it into ``results/`` like every other result.
"""

from __future__ import annotations

import argparse

from scoutfield.app.export_onnx import DEFAULT_CONFIG, run_latency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--push", action="store_true",
                        help="upload deployment_latency.json to stage 08_deployment")
    args = parser.parse_args()

    record = run_latency(args.config, push=args.push)
    hw = record["hardware"]
    print(f"on {hw['cpu_model']}, {hw['physical_cores']} cores / {hw['logical_cores']} "
          f"threads, {hw['ram_gb']} GB RAM, {hw['os']}, mains power {hw['on_mains_power']}")
    if not args.push:
        print("not pushed to the store (run again with --push to do that)")


if __name__ == "__main__":
    main()
