"""
Check one stage of a pipeline run, reading only from the public artefact store.

    python scripts/verify_stage.py preflight [--resume-run]
    python scripts/verify_stage.py 01_perception
    python scripts/verify_stage.py all

Prints one PASS / FAIL line per check with the value it found, then the stage's key
numbers, and exits non-zero if any check failed. Run it after every stage and do not
start the next stage until every line says PASS.

``preflight`` needs the Hugging Face login and the Kaggle CLI; every other check
needs no token.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoutfield.config import load_config
from scoutfield.utils.artifacts import get_store, load_settings

FIGURE_NUMBERS = range(9, 16)


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def check(self, ok: bool, label: str, value="") -> bool:
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f": {value}" if value != "" else ""))
        self.failed += 0 if ok else 1
        return ok

    def key(self, label: str, value) -> None:
        print(f"      {label}: {value}")


def _driver(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  ROOT / "experiments" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _present(r: Report, store, stage: str, names: list[str]) -> bool:
    files = store.stage_entry(stage).get("files", {})
    missing = [n for n in names if n not in files]
    return r.check(not missing, f"{stage} files present",
                   "all" if not missing else f"missing {missing}")


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                         check=False)
    return out.stdout.strip()


# ------------------------------------------------------------------ checks

def _unique_runs(r: Report, rows: list[dict], jobs: list[dict]) -> None:
    ids = [row["run_id"] for row in rows]
    r.check(len(ids) == len(set(ids)), "one row per run_id",
            f"{len(ids) - len(set(ids))} duplicates")
    unknown = set(ids) - {j["run_id"] for j in jobs}
    r.check(not unknown, "every row is a job of this config",
            "yes" if not unknown else sorted(unknown)[:3])


def preflight(r: Report, store, resume_run: bool) -> None:
    settings = load_settings()
    owner = settings["hub_repo"].split("/")[0]
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        who = api.whoami().get("name")
        r.check(who == owner, "huggingface-cli whoami", who)
        info = api.repo_info(settings["hub_repo"], repo_type=settings.get("repo_type",
                                                                          "dataset"))
        r.check(not info.private, f"{settings['hub_repo']} exists and is public",
                "private" if info.private else "public")
    except Exception as exc:  # noqa: BLE001 - every failure here is a FAIL line
        r.check(False, "Hugging Face login and store repository", f"{type(exc).__name__}: {exc}")

    try:
        kaggle = subprocess.run(["kaggle", "kernels", "list", "--mine"],
                                capture_output=True, text=True, timeout=60, check=False)
        r.check(kaggle.returncode == 0, "kaggle kernels list --mine",
                "ok" if kaggle.returncode == 0 else kaggle.stderr.strip()[:200])
    except (OSError, subprocess.SubprocessError) as exc:
        r.check(False, "kaggle kernels list --mine", str(exc))

    run_id = str(settings["run_id"])
    r.check(bool(re.fullmatch(r"r\d{4}-\d{2}-\d{2}", run_id)), "run_id is set", run_id)
    used = bool(store.manifest().get("stages"))
    r.check(not used or resume_run, "run_id is unused, or resuming it on purpose",
            ("already used; pass --resume-run to continue it" if used and not resume_run
             else "resuming" if used else "unused"))

    dirty = _git("status", "--porcelain", "--untracked-files=no")
    r.check(not dirty, "working tree committed", "clean" if not dirty else dirty[:200])
    head = _git("rev-parse", "HEAD")
    branch = load_config(ROOT / "configs" / "kaggle.yaml")["branch"]
    remote = _git("ls-remote", "origin", f"refs/heads/{branch}").split("\t")[0]
    r.check(head == remote, f"HEAD pushed to origin/{branch}",
            head[:10] if head == remote else f"local {head[:10]}, remote {remote[:10]}")


def stage_01(r: Report, store) -> None:
    if not _present(r, store, "01_perception",
                    ["best.pt", "perception_summary.json", "leakage_report.json"]):
        return
    store.pull("01_perception", "best.pt")
    r.check(True, "sha256 of every file matches the manifest", "verified on download")
    leak = store.pull_json("01_perception", "leakage_report.json")
    r.check(leak["cross_split_duplicates"] == 0, "cross-split duplicates",
            leak["cross_split_duplicates"])
    summary = store.pull_json("01_perception", "perception_summary.json")
    keys = ["accuracy", "accuracy_shift", "best_epoch"]
    missing = [k for k in keys if summary.get(k) is None]
    r.check(not missing, "summary has accuracy, accuracy_shift, best_epoch",
            "yes" if not missing else f"missing {missing}")
    manifest_commit = store.stage_entry("01_perception").get("git_commit")
    r.check(summary.get("git_commit") == manifest_commit,
            "summary git_commit equals the manifest's", summary.get("git_commit"))
    r.key("PlantVillage test accuracy", summary.get("accuracy"))
    r.key("PlantDoc accuracy (all / shift_eval)",
          f"{summary.get('accuracy_shift')} / {summary.get('accuracy_shift_eval')}")
    r.key("best epoch", summary.get("best_epoch"))
    r.key("lab-fitted temperature", summary.get("lab_fitted_temperature"))
    r.key("identical files PlantVillage/PlantDoc",
          leak["plantvillage_plantdoc_identical_files"])
    r.key("PlantVillage split sizes", leak["plantvillage"]["split_sizes"])
    r.key("PlantDoc split sizes", leak["plantdoc"]["split_sizes"])


def stage_02(r: Report, store) -> None:
    from scoutfield.perception.adapter import logit_pool_filename, pool_membership
    from scoutfield.perception.calibrate import POOL_SPLITS

    pools = [logit_pool_filename(s) for s in POOL_SPLITS]
    if not _present(r, store, "02_calibration", ["calibration_sweep.json", *pools]):
        return
    cal = store.pull_json("02_calibration", "calibration_sweep.json")
    lab, shift = cal.get("lab_fitted_temperature"), cal.get("shift_fitted_temperature")
    r.check(bool(lab and lab > 0 and shift and shift > 0),
            "lab and shift fitted temperatures present and positive", f"{lab} / {shift}")

    for name, entry in cal["sweeps"].items():
        accs = {round(row["accuracy"], 6) for row in entry["sweep"].values()}
        r.check(len(accs) == 1, f"accuracy identical to 6 d.p. on {name}", sorted(accs))

    sweep = cal["sweeps"]["shift_eval"]["sweep"]
    wrong = [t for t, row in sweep.items()
             if (float(t) < 1 and not row["signed_calibration_error"] > 0)
             or (float(t) > 1 and not row["signed_calibration_error"] < 0)]
    r.check(not wrong, "shift_eval signed error > 0 below T = 1 and < 0 above",
            "yes" if not wrong else f"wrong sign at T = {wrong}")

    leak = (store.pull_json("01_perception", "leakage_report.json", required=False) or {})
    recorded = leak.get("membership", {})
    for split in POOL_SPLITS:
        stored = pool_membership(store.pull("02_calibration", logit_pool_filename(split)))
        expected = cal["pools"][split]["membership"]
        same_as_01 = recorded.get(split) in (None, expected)
        r.check(stored == expected and same_as_01,
                f"pool {split} matches its recorded split", str(stored)[:12])

    r.key("lab-fitted / shift-fitted temperature", f"{lab:.4f} / {shift:.4f}")
    for name in ("test", "shift_eval"):
        r.key(f"{name} accuracy", cal["sweeps"][name]["at_fitted_temperature"]["accuracy"])
    print("      shift_eval, relative to the shift-fitted temperature:")
    print("        T      ECE      signed error")
    for t in sorted(sweep, key=float):
        row = sweep[t]
        print(f"        {float(t):<6g} {row['ece']:.4f}   {row['signed_calibration_error']:+.4f}")
    for split in POOL_SPLITS:
        r.key(f"pool {split} by class", cal["pools"][split]["n_by_class"])


def stage_03(r: Report, store) -> None:
    seeds = load_config(ROOT / "configs" / "sweep.yaml")["ppo_train_seeds"]
    names = [f"ppo_seed{s}.zip" for s in seeds] + [f"ppo_curve_seed{s}.json" for s in seeds]
    if not _present(r, store, "03_ppo", names):
        return
    entry = store.stage_entry("03_ppo")
    recorded = {(i["stage"], i["name"]): i for i in entry.get("inputs", [])}
    cal_input = recorded.get(("02_calibration", "calibration_sweep.json"))
    cal_entry = store.stage_entry("02_calibration")["files"]["calibration_sweep.json"]
    r.check(bool(cal_input) and cal_input["sha256"] == cal_entry["sha256"],
            "stage 03 recorded the calibration file it read",
            cal_input["sha256"][:12] if cal_input else "not recorded")
    shift = store.pull_json("02_calibration", "calibration_sweep.json")["shift_fitted_temperature"]
    for s in seeds:
        curve = store.pull_json("03_ppo", f"ppo_curve_seed{s}.json")
        r.check(curve.get("complete") is True and curve.get("converged") is True,
                f"seed {s} complete and converged",
                f"complete={curve.get('complete')}, converged={curve.get('converged')}")
        r.check(abs(float(curve.get("reference_temperature", -1)) - shift) < 1e-12,
                f"seed {s} trained at the shift-fitted temperature",
                curve.get("reference_temperature"))
        eps = curve.get("episodes", [])
        tail = [e["detections_per_joule"] for e in eps[-max(len(eps) // 10, 1):]]
        r.key(f"seed {s} episodes / final-10% mean det/J",
              f"{len(eps)} / {sum(tail) / max(len(tail), 1):.4f}")


def stage_04(r: Report, store) -> None:
    if not _present(r, store, "04_sweep",
                    ["sweep_results.csv", "summary.json", "sweep_metadata.json"]):
        return
    cfg = load_config(ROOT / "configs" / "sweep.yaml")
    jobs = _driver("04_sweep.py").build_job_list(cfg)
    with store.pull("04_sweep", "sweep_results.csv").open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    r.check(len(rows) == len(jobs), "row count matches configs/sweep.yaml",
            f"{len(rows)} of {len(jobs)}")
    _unique_runs(r, rows, jobs)
    empty = sum(1 for row in rows if row.get("ece") in ("", None))
    r.check(empty == 0, "no empty ece", empty)

    cal = store.pull_json("02_calibration", "calibration_sweep.json")
    expected = {"shift_fitted": cal["shift_fitted_temperature"],
                "lab_fitted": cal["lab_fitted_temperature"]}
    for condition, value in expected.items():
        seen = {float(row["reference_temperature"]) for row in rows
                if row["reference_condition"] == condition}
        r.check(seen == {value}, f"reference_temperature on every {condition} row",
                sorted(seen))

    summary = store.pull_json("04_sweep", "summary.json")
    want = {f"{a}@T{float(t):g}" for a in cfg["agents"] for t in cfg["temperatures"]}
    missing = sorted(want - set(summary["cells"]))
    r.check(not missing, "every agent x temperature cell present",
            "yes" if not missing else missing[:5])
    for condition in cfg.get("named_conditions") or []:
        have = set(summary.get("conditions", {}).get(condition, {}))
        r.check(have == set(cfg["agents"]), f"condition {condition} present", sorted(have))
    spans = set(cfg["degradation_spans"])
    r.check(spans <= set(summary.get("degradation", {})), "all three degradation spans",
            sorted(summary.get("degradation", {})))

    print("      detections per joule degradation (ratio [95% CI], P(improvement)):")
    for name, span in summary["degradation"].items():
        print(f"        {name} T {span['low']:g} -> {span['high']:g}")
        for agent, d in sorted(span["agents"].items()):
            ci = (f"[{d['ci_low']:.2f}, {d['ci_high']:.2f}]"
                  if d.get("ci_low") is not None else "[no interval]")
            print(f"          {agent:<15} {d['ratio']:.2f}x {ci}  "
                  f"P={d['probability_of_improvement']:.2f}")


def stage_05(r: Report, store) -> None:
    names = ["tau_sensitivity.csv", "tau_sensitivity_summary.json"]
    if not _present(r, store, "05_tau", names):
        return
    cfg = load_config(ROOT / "configs" / "sweep.yaml")
    jobs = _driver("05_tau_sensitivity.py").build_tau_jobs(cfg)
    with store.pull("05_tau", "tau_sensitivity.csv").open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    r.check(len(rows) == len(jobs), "row count matches configs/sweep.yaml",
            f"{len(rows)} of {len(jobs)}")
    _unique_runs(r, rows, jobs)
    summary = store.pull_json("05_tau", "tau_sensitivity_summary.json")
    zero = [k for k, e in summary.items()
            if not e.get("ci_T1") or not e["ci_T1"][1] > e["ci_T1"][0]]
    r.check(not zero, "no zero-width confidence interval", "none" if not zero else zero[:5])
    for key in sorted(summary):
        r.key(key, f"ratio {summary[key]['ratio']:.2f}")


def stage_06(r: Report, store) -> None:
    files = store.stage_entry("06_figures").get("files", {})
    for n in FIGURE_NUMBERS:
        hits = [f for f in files if f.startswith(f"Figure{n}_") and f.endswith(".png")]
        r.check(bool(hits), f"Figure {n}", hits[0] if hits else "missing")


STAGE_CHECKS = {
    "01_perception": stage_01,
    "02_calibration": stage_02,
    "03_ppo": stage_03,
    "04_sweep": stage_04,
    "05_tau": stage_05,
    "06_figures": stage_06,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["preflight", *STAGE_CHECKS, "all"])
    parser.add_argument("--resume-run", action="store_true",
                        help="preflight: the run_id is already in use and is being resumed")
    parser.add_argument("--run", default=None, help="run_id; default: configs/artifacts.yaml")
    args = parser.parse_args()

    store = get_store(args.run)
    print(f"run {store.run_id} in {store.backend.description}")
    r = Report()
    if args.check == "preflight":
        preflight(r, store, args.resume_run)
    else:
        for name, fn in STAGE_CHECKS.items():
            if args.check in (name, "all"):
                print(f"\n== {name}")
                try:
                    fn(r, store)
                except Exception as exc:  # noqa: BLE001 - reported as a FAIL, not a crash
                    r.check(False, f"{name} could not be checked",
                            f"{type(exc).__name__}: {exc}")
    print(f"\n{'ALL PASS' if not r.failed else f'{r.failed} FAILED'}")
    raise SystemExit(1 if r.failed else 0)


if __name__ == "__main__":
    main()
