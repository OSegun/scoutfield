"""
Roadmap item 9: check the deployed path on the demonstration's real photographs.

    python experiments/08_select_samples.py       # first, once (chooses the samples)
    python experiments/08_check_demo_logits.py    # writes demo_logit_check.json
    python experiments/08_check_demo_logits.py --push

Each sample photograph goes through the path the demonstration uses (NumPy and
Pillow preprocessing, then the exported classifier on ONNX Runtime's CPU provider)
and its logit is compared with the one stage 02 recorded for the same photograph
(PyTorch, from the stored logit pool). The export check compares the graph on
random inputs; this one covers decoding and preprocessing on real imagery too. It
passes when every logit is within ``export.verify.logit_atol`` of
``configs/deployment.yaml`` and every prediction agrees.

The PlantDoc samples are read from stage ``08_deployment`` of the store. The
PlantVillage samples are included only where this machine has them (their licence
keeps them out of the public store); the record says which group each came from.

``--push`` files ``demo_logit_check.json`` in stage ``08_deployment``. It needs a
Hugging Face write token and is refused unless the classifier checked is the one
filed there, the check passed, and the working tree is committed. The promotion
step then copies it into ``results/`` like every other result.
"""

from __future__ import annotations

import argparse

from scoutfield.app.demo import run_sample_check
from scoutfield.app.export_onnx import DEFAULT_CONFIG, load_deployment_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--push", action="store_true",
                        help="upload demo_logit_check.json to stage 08_deployment")
    args = parser.parse_args()

    out = run_sample_check(load_deployment_config(args.config), push=args.push)
    record = out["record"]
    for row in record["samples"]:
        print(f"{row['store_name']:22s} stage 02 {row['stage02_logit']:+10.5f}  "
              f"deployed {row['deployed_logit']:+10.5f}  |diff| {row['abs_difference']:.2e}")
    print(f"{record['n_samples']} samples, largest difference "
          f"{record['max_abs_difference']:.3g} (tolerance {record['logit_atol']:g}): "
          f"{'passed' if record['passed'] else 'FAILED'}")
    print(f"wrote {out['path']}")
    if out["pushed"]:
        print(f"pushed {out['path'].name} to stage 08_deployment of run {record['run_id']}")
    else:
        print("not pushed to the store (run again with --push to do that)")
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
