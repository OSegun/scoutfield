# Results

**Run `r2026-11-13`.** Every number below is read from the files in `results/`, which
are copies of that run's final outputs in the public artefact store.
`results/RUN.json` names the store revision they were taken from
(`699392734c937bcb827d99dff3244523fbf6d43d`) and lists each file with its SHA-256. If a
number here disagrees with those files, this file is wrong.

Rounding: detections per joule and calibration errors to 4 decimal places, rates
(recall, precision, F1, coverage) to 3, ratios to 2. Intervals are 95% stratified
bootstrap confidence intervals around the interquartile mean (IQM), as set out in
`docs/METHOD.md`, section *"5. Statistics"*.

`T` is **relative to a fitted temperature** throughout, so `T` = 1 is the calibrated
point. Below 1 the classifier is sharpened (overconfident); above 1 it is softened
(underconfident).

---

## 1. Perception

Roadmap items 1 and 3. Source: `results/perception_summary.json`,
`results/calibration_sweep.json`. EfficientNet-B0, best epoch 7.

| Metric | PlantVillage `test`, lab-fitted | PlantDoc `shift_eval`, shift-fitted | PlantDoc `shift_eval`, lab-fitted |
| --- | --- | --- | --- |
| n | 8150 | 1461 | 1461 |
| Accuracy | 0.9995 | 0.8070 | 0.8070 |
| ECE | 0.0006 | 0.0145 | 0.1277 |
| Signed calibration error | −0.0005 | −0.0020 | +0.1249 |
| NLL | 0.0030 | 0.4242 | 0.7229 |
| Brier | 0.0005 | 0.1361 | 0.1577 |
| Mean confidence | 0.9990 | 0.8050 | 0.9319 |

Fitted temperatures: **lab-fitted 0.9949** (on PlantVillage `val`), **shift-fitted
3.3843** (on PlantDoc `shift_fit`). The third column is the field imagery scored with the
temperature fitted on laboratory imagery: the condition named `lab_fitted` in section 3.

BASE_ACC is the PlantDoc `shift_eval` accuracy, **0.8070**. The reasons are in
`docs/METHOD.md`, section *"2. Perception"* → *"Data"* → *"BASE_ACC"*.

### Temperature sweep

Accuracy must be identical down each accuracy column. If it is not, the instrument is
broken (`docs/METHOD.md`, section *"1. The instrument"*). It is identical to six decimal
places on every split.

**PlantVillage `test`, relative to the lab-fitted temperature, n = 8150**

| `T` | Accuracy | ECE | Signed error | Mean confidence |
| --- | --- | --- | --- | --- |
| 0.30 | 0.999509 | 0.0005 | +0.0003 | 0.9998 |
| 0.50 | 0.999509 | 0.0005 | +0.0001 | 0.9996 |
| 1.00 | 0.999509 | 0.0006 | −0.0005 | 0.9990 |
| 2.00 | 0.999509 | 0.0056 | −0.0056 | 0.9939 |
| 3.00 | 0.999509 | 0.0208 | −0.0208 | 0.9787 |
| 4.00 | 0.999509 | 0.0454 | −0.0454 | 0.9541 |

**PlantDoc `shift_eval`, relative to the shift-fitted temperature, n = 1461.** Every
planner in section 3 runs on this pool.

| `T` | Accuracy | ECE | Signed error | Mean confidence |
| --- | --- | --- | --- | --- |
| 0.30 | 0.806982 | 0.1265 | +0.1237 | 0.9307 |
| 0.50 | 0.806982 | 0.0862 | +0.0838 | 0.8907 |
| 1.00 | 0.806982 | 0.0145 | −0.0020 | 0.8050 |
| 2.00 | 0.806982 | 0.1095 | −0.1095 | 0.6975 |
| 3.00 | 0.806982 | 0.1652 | −0.1652 | 0.6417 |
| 4.00 | 0.806982 | 0.1975 | −0.1975 | 0.6095 |

ECE is lowest at `T` = 1 and rises in both directions, and the signed error is positive
below `T` = 1 and negative above it. The two halves are not the same size: the
underconfident end (`T` = 4, ECE 0.1975) is further from calibrated than the
overconfident end (`T` = 0.3, ECE 0.1265).

The same pool swept relative to the lab-fitted temperature is in
`results/calibration_sweep.json` (`sweeps.shift_eval_lab`) and drawn in Figure 10.

### Data and leakage

Source: `results/leakage_report.json`.

| | PlantVillage | PlantDoc |
| --- | --- | --- |
| Images | 54305 | 2922 |
| Distinct contents | 54284 | 2916 |
| Split sizes | train 38011 · val 8144 · test 8150 | shift_fit 1461 · shift_eval 1461 |
| Source keys in more than one split | 0 | 0 |
| Identical files in more than one split | 0 | 0 |

42 PlantVillage files have a byte-identical copy under another name. Grouped by source
key alone, 12 identical contents would have fallen on both sides of a split; grouping by
content as well (`docs/METHOD.md`, section *"2. Perception"* → *"Data"*) keeps them
together. No PlantDoc file is byte-identical to a PlantVillage file.

---

## 2. Environment

Roadmap item 4. Source: `configs/ppo_field32.yaml`, `results/summary.json`
(`environment`, `cells`). 32 × 32 field, energy budget 640.

| Quantity | Value |
| --- | --- |
| Coverage ceiling (optimistic bound) | 0.391 (analytic, from config) |
| Measured coverage at `T` = 1 (IQM), Lawnmower | 0.391 |
| Measured coverage at `T` = 1 (IQM), PPO | 0.252 |
| Field prevalence at `n_parents` = 18, `sigma` = 1.5 | 0.170 |
| Morisita index at `sigma` = 1.5 | 1.480 (sd 0.215, 30 fields) |

Morisita index and prevalence at each cluster scale, 30 fields each:

| `sigma` | Morisita index (sd) | Prevalence |
| --- | --- | --- |
| 0.75 | 1.613 (0.280) | 0.118 |
| 1.5 | 1.480 (0.215) | 0.170 |
| 3 | 1.284 (0.129) | 0.183 |

A Morisita index above 1 means the disease is aggregated rather than uniformly spread;
it falls towards 1 as `sigma` widens the clusters.

---

## 3. Planners

Roadmap items 5 and 8. Source: `results/summary.json`, `results/sweep_results.csv`.
735 runs on the `shift_eval` pool at τ = 0.75, 20 episodes each. IQM with 95% stratified
bootstrap CI over 5 evaluation seeds and 3 cluster scales; 15 runs per cell for each
baseline and 45 for PPO, whose three independently trained policies are each evaluated
on every seed and scale.

### PPO training

Source: `results/ppo_curve_seed{0,1,2}.json`. These are **training** episodes on the
`shift_fit` pool, not the evaluation sweep. IQM over the first and last 10% of each
seed's episodes, pooled across the three seeds.

| Quantity | First 10% | Last 10% | Change |
| --- | --- | --- | --- |
| Detections per joule | 0.0656 | 0.0927 | +41.2% |
| Recall | 0.234 | 0.328 | +39.9% |
| Coverage | 0.227 | 0.248 | +9.1% |
| Time to first detection | 2.87 | 2.20 | −23.4% |
| Precision | 0.361 | 0.434 | +20.1% |
| False alarms | 73.9 | 77.3 | +4.6% |

Convergence per seed, as the change in mean detections per joule between the last fifth
of training and the fifth before it (converged if under 5%): seed 0 −0.94%, seed 1
+1.56%, seed 2 +1.22%. All three converged. Seed 1 was interrupted and resumed from its
checkpoint at timestep 800008; its curve file records this.

### Detections per joule, with the calibration each column ran at

The first two rows are the classifier's calibration on the `shift_eval` pool at that
`T`, as recorded on every run in that column.

| | `T`=0.3 | `T`=0.5 | `T`=1 | `T`=2 | `T`=3 | `T`=4 |
| --- | --- | --- | --- | --- | --- | --- |
| ECE | 0.1265 | 0.0862 | 0.0145 | 0.1095 | 0.1652 | 0.1975 |
| Signed error | +0.1237 | +0.0838 | −0.0020 | −0.1095 | −0.1652 | −0.1975 |
| Oracle | 0.2505 | 0.2489 | 0.2435 | 0.2182 | 0.1937 | 0.1807 |
| Lawnmower | 0.0922 | 0.0879 | 0.0774 | 0.0474 | 0.0202 | 0.0050 |
| PPO | 0.0801 | 0.0766 | 0.0695 | 0.0539 | 0.0430 | 0.0361 |
| GreedyEntropy | 0.0624 | 0.0544 | 0.0448 | 0.0402 | 0.0416 | 0.0416 |
| Random | 0.0396 | 0.0387 | 0.0369 | 0.0308 | 0.0262 | 0.0233 |

Intervals for every cell are in `results/summary.json` (`cells`) and drawn in Figure 13.
Ranking by detections per joule, best first:

| `T` | Ranking |
| --- | --- |
| 0.3, 0.5, 1 | Oracle, Lawnmower, PPO, GreedyEntropy, Random |
| 2 | Oracle, PPO, Lawnmower, GreedyEntropy, Random |
| 3 | Oracle, PPO, GreedyEntropy, Random, Lawnmower |
| 4 | Oracle, GreedyEntropy, PPO, Random, Lawnmower |

### Degradation over the three spans

Ratio of detections per joule at the less-miscalibrated end of the span to the
more-miscalibrated end, with its 95% CI and the probability of improvement. The spans are
defined in `docs/METHOD.md`, section *"5. Statistics"* → *"Degradation spans"*.

| Planner | Full, `T` 0.3 → 4 | Overconfident, `T` 0.3 → 1 | Underconfident, `T` 1 → 4 |
| --- | --- | --- | --- |
| Oracle | 1.39 [1.19, 1.52], P = 0.91 | 1.03 [0.87, 1.20], P = 0.64 | 1.35 [1.18, 1.46], P = 0.91 |
| Lawnmower | 18.52 [15.00, 23.09], P = 1.00 | 1.19 [0.98, 1.45], P = 0.73 | 15.54 [12.54, 19.39], P = 1.00 |
| PPO | 2.22 [2.02, 2.45], P = 1.00 | 1.15 [1.05, 1.27], P = 0.76 | 1.93 [1.74, 2.13], P = 1.00 |
| GreedyEntropy | 1.50 [1.23, 1.83], P = 0.88 | 1.39 [1.15, 1.75], P = 0.82 | 1.08 [0.86, 1.31], P = 0.60 |
| Random | 1.70 [1.39, 2.05], P = 0.93 | 1.07 [0.88, 1.30], P = 0.60 | 1.59 [1.30, 1.92], P = 0.89 |

An interval that contains 1.00 does not rule out no change over that span: Lawnmower,
Oracle and Random on the overconfident half, and GreedyEntropy on the underconfident
half.

### Precision, recall, F1 and false alarms

| Planner | Precision `T` 0.3 / 1 / 4 | Recall `T` 0.3 / 1 / 4 | F1 `T` 0.3 / 1 / 4 | False alarms `T` 0.3 / 1 / 4 |
| --- | --- | --- | --- | --- |
| Oracle | 0.772 / 0.838 / 0.915 | 0.990 / 0.970 / 0.751 | 0.867 / 0.898 / 0.817 | 44.6 / 28.8 / 10.8 |
| Lawnmower | 0.309 / 0.393 / 0.944 | 0.360 / 0.303 / 0.019 | 0.329 / 0.338 / 0.037 | 128.8 / 73.6 / 0.0 |
| PPO | 0.324 / 0.333 / 0.220 | 0.317 / 0.275 / 0.142 | 0.312 / 0.294 / 0.168 | 106.8 / 87.5 / 82.8 |
| GreedyEntropy | 0.239 / 0.184 / 0.163 | 0.250 / 0.172 / 0.166 | 0.240 / 0.175 / 0.161 | 124.8 / 124.2 / 135.7 |
| Random | 0.239 / 0.236 / 0.166 | 0.159 / 0.148 / 0.094 | 0.186 / 0.178 / 0.116 | 78.3 / 74.5 / 74.3 |

False alarms are per episode, averaged over each run's 20 episodes. GreedyEntropy is the only planner whose false
alarms are higher at `T` = 4 than at `T` = 0.3. False alarms per joule for every cell are
in `results/summary.json`.

### The lab-fitted condition

The same pool scored with the temperature fitted on laboratory imagery (0.9949) instead
of the one fitted on field imagery (3.3843): ECE 0.1277, signed error +0.1249. Relative
to the shift-fitted scale this is a factor of 0.9949 / 3.3843 = 0.294, close to `T` =
0.3.

| Planner | Detections per joule | Precision | Recall | F1 | Precision at `T` = 1, shift-fitted |
| --- | --- | --- | --- | --- | --- |
| Oracle | 0.2505 [0.2152, 0.2719] | 0.771 [0.763, 0.780] | 0.991 | 0.866 | 0.838 |
| Lawnmower | 0.0922 [0.0787, 0.1021] | 0.307 [0.270, 0.336] | 0.360 | 0.328 | 0.393 |
| PPO | 0.0811 [0.0754, 0.0853] | 0.323 [0.303, 0.337] | 0.320 | 0.314 | 0.333 |
| GreedyEntropy | 0.0628 [0.0554, 0.0708] | 0.241 [0.216, 0.267] | 0.252 | 0.242 | 0.184 |
| Random | 0.0396 [0.0338, 0.0440] | 0.238 [0.207, 0.263] | 0.159 | 0.186 | 0.236 |

### Regret against the oracle

At `T` = 1, the calibrated point. Regret is 1 − planner / oracle.

| Planner | Regret on detections/joule | Regret on recall |
| --- | --- | --- |
| Lawnmower | 0.682 | 0.688 |
| PPO | 0.715 | 0.717 |
| GreedyEntropy | 0.816 | 0.823 |
| Random | 0.848 | 0.847 |

The oracle is greedy nearest-diseased-cell, a **lower bound** on the true ceiling, since
optimal routing under an energy budget is NP-hard. Regret against it therefore
*understates* the real gap. Figure 15 shows regret at every `T`.

### PPO against Lawnmower at `T` = 1

| At `T` = 1 | Lawnmower | PPO |
| --- | --- | --- |
| Detections per joule | 0.0774 [0.0657, 0.0862] | 0.0695 [0.0641, 0.0735] |
| Coverage | 0.391 | 0.252 |
| Recall | 0.303 | 0.275 |
| Precision | 0.393 | 0.333 |
| False alarms | 73.6 | 87.5 |

The two detections-per-joule intervals overlap.

---

## 4. τ sensitivity

Roadmap item 6. Source: `results/tau_sensitivity.csv`,
`results/tau_sensitivity_summary.json`.

Effect size is the detections-per-joule ratio between `T` = 1 and `T` = 4 (the
underconfident span) at each confirmation threshold τ. Baselines run all three cluster
scales (15 runs per cell). PPO runs a reduced τ grid with the seed 0 policy at the median
scale only (`sigma` = 1.5, 5 runs per cell), because its evaluation is much slower; its
ratio at τ = 0.75 (1.82) therefore differs from the main sweep's (1.93).

| τ | Lawnmower | GreedyEntropy | Random | Oracle | PPO |
| --- | --- | --- | --- | --- | --- |
| 0.55 | 1.13 | 1.07 | 1.06 | 1.02 | — |
| 0.60 | 1.44 | 1.07 | 1.15 | 1.07 | 1.31 |
| 0.65 | 2.23 | 1.07 | 1.31 | 1.18 | — |
| 0.70 | 4.57 | 1.07 | 1.47 | 1.29 | — |
| 0.75 | 15.54 | 1.08 | 1.59 | 1.35 | 1.82 |
| 0.85 | no detections at `T` = 4 | 1.06 | 1.50 | 1.31 | — |
| 0.90 | no detections at `T` = 4 | 1.04 | 1.44 | 1.30 | 1.69 |

At τ = 0.85 and 0.90 Lawnmower's detections per joule at `T` = 4 is 0, so the ratio has
no finite value; its `T` = 1 values there are 0.0613 and 0.0474. `—` marks a τ not in
PPO's grid. Figure 14 draws this table.

---

## 5. Pre-registered predictions

Recorded in `docs/METHOD.md`, section *"Pre-registered predictions"*, before this run.
Each is judged by the rule fixed there, on the values in section 3.

| | Rule | Measured | Outcome |
| --- | --- | --- | --- |
| P1 | Overconfident half: both ratios above 1, and GreedyEntropy's above Lawnmower's | GreedyEntropy 1.39, Lawnmower 1.19 | **Held** |
| P2 | Underconfident half: Lawnmower's interval lower bound above 1, and its ratio above its overconfident ratio | lower bound 12.54; 15.54 against 1.19 | **Held** |
| P3 | Underconfident half: GreedyEntropy's ratio below Lawnmower's | 1.08 against 15.54 | **Held** |
| P4 | At `T` = 1, PPO's interval not entirely above Lawnmower's | [0.0641, 0.0735] against [0.0657, 0.0862] | **Held** |
| P5 | For every planner, precision under `lab_fitted` below precision at `T` = 1 under `shift_fitted` | below for Oracle, Lawnmower and PPO; above for GreedyEntropy (0.241 against 0.184) and Random (0.238 against 0.236) | **Not held** |

P1's rule compares the ratios; Lawnmower's overconfident interval [0.98, 1.45] contains
1. For Random under P5 the two precisions differ by 0.002 and their intervals
overlap; for GreedyEntropy the intervals do not overlap.

---

## 6. Comparison with the pilot, on matched spans

The pilot (`scoutplan@v1.0.0`, `results/summary.json` in that repository) used a
temperature-scaled surrogate classifier on a 12 × 12 field. The same three spans are
compared here; the magnitudes are not expected to match.

| Planner | Span | Pilot | This run |
| --- | --- | --- | --- |
| Lawnmower | Full, `T` 0.3 → 4 | 13.41 | 18.52 [15.00, 23.09] |
| Lawnmower | Overconfident, `T` 0.3 → 1 | 1.23 | 1.19 [0.98, 1.45] |
| Lawnmower | Underconfident, `T` 1 → 4 | 10.86 | 15.54 [12.54, 19.39] |
| GreedyEntropy | Full, `T` 0.3 → 4 | 1.84 | 1.50 [1.23, 1.83] |
| GreedyEntropy | Overconfident, `T` 0.3 → 1 | 1.59 | 1.39 [1.15, 1.75] |
| GreedyEntropy | Underconfident, `T` 1 → 4 | 1.16 | 1.08 [0.86, 1.31] |

---

## 7. Uncertainty methods

Roadmap item 7. **Not run in this release.** MC-dropout and deep ensembles were not
evaluated for run `r2026-11-13`, so there is no comparison with temperature scaling and
Figure 16 is not drawn. Reliability data at the fitted temperature for every split is
in `results/calibration_sweep.json` (`reliability_at_fitted`); Figure 9 draws it for
PlantDoc `shift_eval` under both temperatures.

---

## 8. Deployment

Roadmap item 9. Read from `results/export_record.json` and
`results/deployment_latency.json`; the method is in `docs/METHOD.md`, section
*"6. Deployment"*.

**Export.** The run's `best.pt` was exported to a single ONNX file (opset 18). On 65
seeded random inputs, ONNX Runtime and PyTorch gave identical predictions, and their
logits differed by at most 1.96e-05, within the 1e-4 tolerance.

**Latency.** Measured on a laptop, not on drone hardware: an 11th Gen Intel Core
i7-1165G7 at 2.80 GHz (4 cores, 8 threads), 15.7 GB RAM, Windows 11, on mains power;
ONNX Runtime 1.28.0 on the CPU, Python 3.13.2. One image per call (1 × 3 × 224 × 224),
the model call only, without decoding or resizing the image. Each figure is the middle
of three repeats of 500 timed calls after 50 warm-up calls; the range across the
repeats is in brackets. Milliseconds to 1 decimal place.

| Threads | Median (ms) | p95 (ms) | Images per second (at the median) |
| --- | --- | --- | --- |
| ONNX Runtime default | 8.8 [8.0, 8.8] | 10.2 [9.6, 10.7] | 114.1 |
| 1 | 14.5 [14.5, 15.0] | 18.5 [18.2, 19.9] | 68.8 |

Even on a single core, 95% of classifications finish within 20 ms, so on this machine
the classifier would not hold up a scout that classifies a few images per second. An
on-board computer is slower than this laptop, and its latency has not been measured.

---

## 9. Figures

All in `figures/`, drawn by `experiments/make_figures.py` from the files above.

| Figure | Shows |
| --- | --- |
| 9 | Reliability on PlantDoc `shift_eval`, lab-fitted against shift-fitted temperature |
| 10 | Accuracy and ECE against `T`: PlantVillage `test`, and `shift_eval` under both temperatures |
| 11 | Coverage ceiling against energy budget on the 32 × 32 field |
| 12 | PPO learning curves, three seeds |
| 13 | Detections per joule against `T`, every planner, with intervals |
| 14 | τ sensitivity of the underconfident-span ratio |
| 15 | Regret against the oracle at every `T` |

---

## 10. What the pilot measured

For comparison. These are real, from 195 executed runs in the companion repository
(`scoutplan@v1.0.0`) — not from this repo.

| Finding | Value |
| --- | --- |
| Accuracy across all temperatures | 0.8161 (invariant, 4 d.p.) |
| ECE range | 0.0037 (`T`=1) → 0.1980 (`T`=4) |
| Lawnmower detections/joule collapse, `T` 0.3 → 4 | 0.0781 → 0.0058 = 13.4× |
| Lawnmower recall collapse, `T` 0.3 → 4 | 0.628 → 0.060 |
| GreedyEntropy degradation, `T` 0.3 → 4 | 1.84× |
| Ranking reversal at `T`=4 | GreedyEntropy beats Lawnmower 7.3× |

Two hypotheses refuted: the advantage did not grow with spatial clustering, and
performance was **not** monotone in ECE — direction governs, not magnitude. The learned
planner lost to the lawnmower baseline at every temperature. All of that was reported
rather than hidden, and the same standard applies to whatever this repository finds.
