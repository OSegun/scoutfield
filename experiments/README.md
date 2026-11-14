# experiments

Runnable entry points. One file per roadmap stage, numbered in dependency order.

The split matters: `scoutfield/` is a library — importable, side-effect free, no argparse,
no file writes at import time. Everything that *runs* lives here and depends on the
library. That keeps the library testable and lets a notebook or another script reuse it
without inheriting a CLI.

| Script | Roadmap | Reads | Writes |
| --- | --- | --- | --- |
Inputs and outputs are files in the artefact store (`configs/artifacts.yaml`), by stage.

| Script | Roadmap | Reads | Writes (stage) |
| --- | --- | --- | --- |
| `00_check_leakage.py` | 1 | both datasets | `leakage_report.json` (`01_perception`) |
| `01_finetune_perception.py` | 1 | PlantVillage | `best.pt`, `last.pt`, `perception_summary.json` (`01_perception`) |
| `02_calibrate.py` | 2, 3 | `01_perception/best.pt`, both datasets | `calibration_sweep.json`, three logit pools (`02_calibration`) |
| `03_train_ppo.py` | 4, 5 | `02_calibration` | `ppo_seed<N>.zip`, resume state, `ppo_curve_seed<N>.json` (`03_ppo`) |
| `04_sweep.py` | 8 | `02_calibration`, `03_ppo` | `sweep_results.csv`, `summary.json`, `sweep_metadata.json` (`04_sweep`) |
| `05_tau_sensitivity.py` | 6 | `02_calibration` | `tau_sensitivity.csv`, `tau_sensitivity_summary.json` (`05_tau`) |
| `make_figures.py` | — | stages 02–05 | Figures 9–15 (`06_figures`) |
| `08_export_onnx.py` | 9 | `01_perception/best.pt` | `classifier.onnx`, `export_record.json` (local; `08_deployment` with `--push`) |
| `08_measure_latency.py` | 9 | the exported `classifier.onnx` | `deployment_latency.json` (local; `08_deployment` with `--push`) |
| `08_export_policy.py` | 9 | `03_ppo/ppo_seed0.zip`, `02_calibration` | `policy_ppo_seed0.onnx`, `policy_export_record.json` (local; `08_deployment` with `--push`) |
| `08_choose_demo_seed.py` | 9 | `02_calibration`, `04_sweep/summary.json`, the exported policy | prints the demonstration's default seed |
| `08_select_samples.py` | 9 | `02_calibration` pools, both datasets in `data/` | sample images and `demo_samples.json` (local; PlantDoc ones to `08_deployment` with `--push`) |
| `08_fill_demo_folder.py` | 9 | every file the demonstration reads | `.demo_offline/` (local only) |

## Order

```
00 → 01 → 02 → 03 → 04 → 05 → make_figures
```

`make_figures.py` reads generated results files and nothing else. Never regenerate a
document or a figure before re-running the sweep that feeds it — a stale summary produces
plausible-looking output that disagrees with the data, and nobody notices.

## Resumability

`01`, `03`, `04` and `05` push their resume state to the store as they go and pull it
back on start, so re-running the same command on any machine carries on from where the
store says the run is. Run `04` in bounded chunks:

```bash
while JOB_BUDGET=60 python experiments/04_sweep.py; do :; done
```

It exits 0 while jobs remain, 1 when the sweep is complete and 2 if it aborted. To start
over, set a new `run_id` in `configs/artifacts.yaml`; never delete state inside a run.
