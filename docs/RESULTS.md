# Results

**Results: in production.** Every value below reads `—` until the run that produces it
is complete. Each table already has the shape its result will take.

This file is written *from* `results/summary.json`, not from memory. If a number here
disagrees with that file, this file is wrong.

---

## 1. Perception

Roadmap items 1 and 3. Source: `results/perception_summary.json`,
`results/calibration_sweep.json`.

| Metric | PlantVillage (test) | PlantDoc (shift) |
| --- | --- | --- |
| Accuracy | — | — |
| ECE at fitted `T` | — | — |
| Signed calibration error | — | — |
| NLL | — | — |
| Brier | — | — |

Fitted temperature: —

BASE_ACC is the PlantDoc shift accuracy in the table above. The reasons are in
`docs/METHOD.md`, section *"2. Perception"* → *"Data"* → *"BASE_ACC"*.

### Temperature sweep

`T` is **relative to the fitted temperature** (`sweep_is_relative_to_fitted_temperature:
true`), so `T` = 1 is the calibrated point, exactly as in the pilot. Absolute temperature
is `T` × the fitted temperature.

Accuracy must be identical down each accuracy column. If it is not, the instrument is
broken (`docs/METHOD.md`, section *"1. The instrument"*).

**PlantVillage (test), n = —**

| `T` | Accuracy | ECE | Signed error | Mean confidence |
| --- | --- | --- | --- | --- |
| 0.30 | — | — | — | — |
| 0.50 | — | — | — | — |
| 1.00 | — | — | — | — |
| 2.00 | — | — | — | — |
| 3.00 | — | — | — | — |
| 4.00 | — | — | — | — |

**PlantDoc (shift), n = —**

| `T` | Accuracy | ECE | Signed error | Mean confidence |
| --- | --- | --- | --- | --- |
| 0.30 | — | — | — | — |
| 0.50 | — | — | — | — |
| 1.00 | — | — | — | — |
| 2.00 | — | — | — | — |
| 3.00 | — | — | — | — |
| 4.00 | — | — | — | — |

---

## 2. Environment

Roadmap item 4. Source: printed by `experiments/03_train_ppo.py`, `results/summary.json`.

| Quantity | Value |
| --- | --- |
| Coverage ceiling (optimistic bound) | 0.391 (analytic, from config) |
| Measured mean coverage, lawnmower | — |
| Measured mean coverage, PPO | — |
| Field prevalence at `n_parents` = 18 | — |
| Morisita index at `sigma` = 1.5 | — |

Prevalence is held at the pilot's value across the scale change — see
`docs/METHOD.md`, section *"3. Environment"*, for why that required changing `n_parents`
from 3 to 18.

---

## 3. Planners

Roadmap items 5 and 8. Source: `results/summary.json`. IQM with 95% stratified bootstrap
CI over 5 seeds.

### PPO training

Source: `results/ppo_curve_seed{0,1,2}.json`. These are **training** episodes, not the
evaluation sweep. IQM over the first and last 10% of episodes.

| Quantity | First 10% | Last 10% | Change |
| --- | --- | --- | --- |
| Detections per joule | — | — | — |
| Recall | — | — | — |
| Coverage | — | — | — |
| Time to first detection | — | — | — |
| Precision | — | — | — |
| False alarms | — | — | — |

Convergence per seed (drift in detections/joule across the final 20% of training): —

### Detections per joule

| Planner | `T`=0.3 | `T`=1.0 | `T`=4.0 | Degradation `T`=1→4 | n per cell |
| --- | --- | --- | --- | --- | --- |
| Oracle (greedy, lower bound) | — | — | — | — | — |
| Lawnmower | — | — | — | — | — |
| PPO | — | — | — | — | — |
| GreedyEntropy | — | — | — | — | — |
| Random | — | — | — | — | — |

PPO has three cells' worth of runs because three independently trained policies are
evaluated, each across five evaluation seeds and three cluster scales.

### Regret against the oracle

At `T` = 1, the calibrated point.

| Planner | Regret on detections/joule | Regret on recall |
| --- | --- | --- |
| Lawnmower | — | — |
| PPO | — | — |
| GreedyEntropy | — | — |
| Random | — | — |

The oracle is greedy nearest-diseased-cell, a **lower bound** on the true ceiling, since
optimal routing under an energy budget is NP-hard. Regret against it therefore
*understates* the real gap.

### Selectivity at `T` = 1

| At `T` = 1 | Lawnmower | PPO |
| --- | --- | --- |
| Coverage | — | — |
| Recall | — | — |
| Precision | — | — |
| False alarms | — | — |

---

## 4. τ sensitivity

Roadmap item 6. Source: `results/tau_sensitivity.csv`.

Effect size is the detections-per-joule ratio between `T` = 1 and `T` = 4. PPO runs a
reduced τ grid because its evaluation is much slower than the baselines'.

| τ | Lawnmower | GreedyEntropy | Random | Oracle | PPO |
| --- | --- | --- | --- | --- | --- |
| 0.55 | — | — | — | — | — |
| 0.60 | — | — | — | — | — |
| 0.65 | — | — | — | — | — |
| 0.70 | — | — | — | — | — |
| 0.75 | — | — | — | — | — |
| 0.85 | — | — | — | — | — |
| 0.90 | — | — | — | — | — |

---

## 5. Uncertainty methods

Roadmap item 7. Source: `results/uncertainty_comparison.json`.

| Method | ECE (in-dist) | ECE (shift) | Detections/joule | Cost |
| --- | --- | --- | --- | --- |
| Temperature scaling | — | — | — | one scalar, ~seconds |
| MC-dropout (30 samples) | — | — | — | 30× inference |
| Deep ensemble (5 members) | — | — | — | 5× training |

---

## 6. Deployment

Roadmap item 9. Source: `results/export_report.json`, written by
`python -m scoutfield.app.export_onnx`.

Properties of the exported graph and of the machine it runs on. The hardware is recorded
with every measurement.

| Metric | Value |
| --- | --- |
| ONNX vs PyTorch max logit difference | — (tolerance 1e-4) |
| Median latency per image (batch 1) | — |
| p95 latency per image (batch 1) | — |
| Throughput (batch 1) | — |
| Throughput (batch 8) | — |
| Hardware | — |
| ONNX Runtime version | — |

Re-measure on the target device before quoting any of this as a deployment figure; a
laptop CPU is not a companion computer.

A latency figure without the hardware it was measured on is not reproducible and does not
belong in a paper. Report median and p95, not the mean — tail latency determines whether a
flight controller misses its deadline, and the mean hides it.

---

## 7. What the pilot measured

For comparison. These are real, from 195 executed runs in the companion repository
(`scoutplan@v1.0.0`) — not from this repo.

| Finding | Value |
| --- | --- |
| Accuracy across all temperatures | 0.8161 (invariant, 4 d.p.) |
| ECE range | 0.0037 (`T`=1) → 0.1980 (`T`=4) |
| Lawnmower detections/joule collapse | 0.0781 → 0.0058 = 13.4× |
| Lawnmower recall collapse | 0.628 → 0.060 |
| GreedyEntropy degradation | 1.84× |
| Ranking reversal at `T`=4 | GreedyEntropy beats Lawnmower 7.3× |

Two hypotheses refuted: the advantage did not grow with spatial clustering, and
performance was **not** monotone in ECE — direction governs, not magnitude. The learned
planner lost to the lawnmower baseline at every temperature. All of that was reported
rather than hidden, and the same standard applies to whatever this repository finds.
