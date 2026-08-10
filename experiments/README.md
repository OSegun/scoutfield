# experiments

Runnable entry points. One file per roadmap stage, numbered in dependency order.

The split matters: `scoutfield/` is a library — importable, side-effect free, no argparse,
no file writes at import time. Everything that *runs* lives here and depends on the
library. That keeps the library testable and lets a notebook or another script reuse it
without inheriting a CLI.

| Script | Roadmap | Reads | Writes |
| --- | --- | --- | --- |
| `01_finetune_perception.py` | 1 | `configs/perception.yaml` | `checkpoints/perception/`, `results/perception_summary.json` |
| `02_calibrate.py` | 2, 3, 7 | perception checkpoint | `results/calibration_sweep.json`, reliability data |
| `03_train_ppo.py` | 4, 5 | `configs/ppo_field32.yaml` | `checkpoints/ppo/` |
| `04_sweep.py` | — | all of the above | `results/sweep_results.csv`, `results/summary.json` |
| `05_tau_sensitivity.py` | 6 | `configs/sweep.yaml` | `results/tau_sensitivity.csv` |
| `make_figures.py` | — | `results/summary.json` | `figures/Figure*.png` |

## Order

```
01 → 02 → 03 → 04 → 05 → make_figures
```

`make_figures.py` reads `results/summary.json` and nothing else. Never regenerate a
document or a figure before re-running the sweep that feeds it — a stale summary produces
plausible-looking output that disagrees with the data, and nobody notices.

## Resumability

`03` and `04` are the long ones and both checkpoint. Run `04` in bounded chunks so a
killed Kaggle session costs one chunk rather than the whole sweep:

```bash
while JOB_BUDGET=60 python experiments/04_sweep.py; do :; done
```

To restart a sweep cleanly, delete **both** `results/_done.json` and the results CSV.
Deleting one leaves the driver resuming and appending fresh rows to stale ones with no
warning.
