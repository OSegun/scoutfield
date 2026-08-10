# Results

**No scientific result in this repository has been measured yet.** Every placeholder reads
`—` and has the shape the result will take.

Three things *have* been measured, and are shown in bold where they appear: the coverage
ceiling and field prevalence in section *"2. Environment"*, and the export equivalence and
inference latency in section *"6. Deployment"*. None of them depends on a trained model —
they are properties of the environment configuration and of the exported graph — which is
why they exist before the perception model does. Everything that depends on trained
weights is still a placeholder.

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

Fitted temperature: **—**

### Temperature sweep

Accuracy must be identical down the first column. That is the instrument working.

| `T` | Accuracy | ECE | Signed error | Direction |
| --- | --- | --- | --- | --- |
| 0.30 | — | — | — | overconfident |
| 0.50 | — | — | — | overconfident |
| 1.00 | — | — | — | calibrated |
| 2.00 | — | — | — | underconfident |
| 3.00 | — | — | — | underconfident |
| 4.00 | — | — | — | underconfident |

---

## 2. Environment

Roadmap item 4. Source: printed by `experiments/03_train_ppo.py`, `results/summary.json`.

| Quantity | Value |
| --- | --- |
| Coverage ceiling (optimistic bound) | 0.391 (analytic, from config) |
| Measured mean coverage, lawnmower | **0.391** (5 seeds, matches the bound) |
| Measured mean coverage, PPO | — |
| Field prevalence at `n_parents` = 18 | **0.170** (30 seeds; pilot 12×12 measures 0.1697) |
| Morisita index at `sigma` = 1.5 | — |

The coverage bound is verified rather than assumed: a lawnmower sweep attains 0.391,
confirming the intended ≤40% regime. Prevalence is held at the pilot's value across the
scale change — see `docs/METHOD.md`, section *"3. Environment"*, for why that required
changing `n_parents` from 3 to 18 and what happens if it is not done.

---

## 3. Planners

Roadmap items 5 and 8. Source: `results/summary.json`. IQM with 95% stratified bootstrap
CI over 5 seeds.

### Detections per joule

| Planner | `T`=0.3 | `T`=1.0 | `T`=4.0 | Degradation `T`=1→4 |
| --- | --- | --- | --- | --- |
| Lawnmower | — | — | — | — |
| Random | — | — | — | — |
| GreedyEntropy | — | — | — | — |
| PPO | — | — | — | — |
| Oracle (greedy, lower bound) | — | — | — | — |

### Regret against the oracle

| Planner | Regret on detections/joule | Regret on recall |
| --- | --- | --- |
| Lawnmower | — | — |
| GreedyEntropy | — | — |
| PPO | — | — |

The oracle is greedy nearest-diseased-cell, a **lower bound** on the true ceiling, since
optimal routing under an energy budget is NP-hard. Regret against it therefore
*understates* the real gap.

---

## 4. τ sensitivity

Roadmap item 6. Source: `results/tau_sensitivity.csv`.

Until this table has numbers, the pilot's 13.4× may be quoted only as "at τ = 0.75".
(13.4×, not 13.5× — the latter comes from dividing rounded endpoints; the full-precision
values in the pilot's `summary.json` give 13.41×.)

| τ | Effect size, `T`=1→4 (IQM) | 95% CI |
| --- | --- | --- |
| 0.55 | — | — |
| 0.60 | — | — |
| 0.65 | — | — |
| 0.70 | — | — |
| 0.75 | — | — |
| 0.85 | — | — |
| 0.90 | — | — |

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

Architecture-level numbers, measured on the development laptop. These are properties of
the exported graph and the machine, not of the weights, so they are valid ahead of
training — **but the model exported to obtain them was untrained**, and no accuracy or
calibration number appears in this section for that reason.

| Metric | Value |
| --- | --- |
| ONNX vs PyTorch max logit difference | **2.2e-07** (tolerance 1e-4) |
| Median latency per image (batch 1) | **11.8 ms** |
| p95 latency per image (batch 1) | **13.5 ms** |
| Throughput (batch 1) | **84.8 images/s** |
| Throughput (batch 8) | 64.0 images/s — batching does not help on this CPU |
| Hardware | Intel64 Family 6 Model 140 (Tiger Lake), Windows 11 22621, CPU only |
| ONNX Runtime version | 1.28.0, `CPUExecutionProvider` |

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
