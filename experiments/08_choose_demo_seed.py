"""
Choose the demonstration's default evaluation seed by rule.

    python experiments/08_choose_demo_seed.py

For each candidate seed (``demo.seed_candidates`` in ``configs/deployment.yaml``),
Lawnmower, GreedyEntropy and PPO each fly one episode at T = 1 and at T = 4, at the
configured cluster scale and tau, exactly as the Plan panel runs them. Each
episode's detections per joule is compared with the sweep's IQM for that planner
and temperature (``summary.json``), as |episode - IQM| / IQM. The seed with the
smallest mean over the six cells is printed, with the whole table; copy it into
``demo.default_seed``.

The PPO policy is read from this machine's export when it exists (so the seed can be
chosen before the policy is pushed), otherwise from the store.
"""

from __future__ import annotations

import argparse

from scoutfield.app import demo
from scoutfield.app.export_onnx import DEFAULT_CONFIG, load_deployment_config, work_dir
from scoutfield.app.export_policy import ONNXPolicy
from scoutfield.utils.artifacts import get_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_deployment_config(args.config)
    store = get_store()
    settings = demo.load_settings()
    local = work_dir(config, store.run_id) / config.section("policy")["policy_file"]
    policy = ONNXPolicy(local) if local.is_file() else demo.load_policy(store, config)
    summary_cfg = config.section("demo")["summary"]
    summary = store.pull_json(summary_cfg["stage"], summary_cfg["name"])

    result = demo.choose_seed(
        summary, settings, demo.load_pool(settings.pool_split, store=store),
        demo.fitted_temperature(settings.reference_condition, store=store), policy,
        int(config.section("demo")["seed_candidates"]))

    cells = list(result["rows"][0]["detections_per_joule"])
    print("sweep IQM   " + "  ".join(f"{c:>20s}" for c in cells))
    print("            " + "  ".join(
        f"{summary['cells'][c]['detections_per_joule']['value']:20.4f}" for c in cells))
    for row in sorted(result["rows"], key=lambda r: r["mean_relative_difference"]):
        print(f"seed {row['seed']:3d}  mean rel. diff {row['mean_relative_difference']:.3f}  "
              + "  ".join(f"{row['detections_per_joule'][c]:.4f}" for c in cells))
    print(f"chosen seed: {result['seed']} (tau {settings.tau}, sigma {settings.sigma}); "
          f"set demo.default_seed to it in configs/deployment.yaml")


if __name__ == "__main__":
    main()
