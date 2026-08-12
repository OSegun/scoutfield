# Results

**Perception and calibration are measured; the planner work downstream of them is not.**
Section *"1. Perception"* now holds real numbers from a trained EfficientNet-B0 and its
temperature sweep. Every remaining placeholder reads `—` and has the shape the result
will take.

Also measured, and shown in bold where they appear: the coverage ceiling and field
prevalence in section *"2. Environment"*, and the export equivalence and inference
latency in section *"6. Deployment"*. Neither depends on trained weights — they are
properties of the environment configuration and of the exported graph — which is why they
predate the perception model.

This file is written *from* `results/summary.json`, not from memory. If a number here
disagrees with that file, this file is wrong.

---

## 1. Perception

Roadmap items 1 and 3. Source: `results/perception_summary.json`,
`results/calibration_sweep.json`.

Measured 10 August 2026 from commit `34e1023`, seed 0, EfficientNet-B0, 12 epochs, best
epoch 6. Splits: 38,013 train / 8,146 val / 8,146 test (38 fine-grained classes, ~72%
diseased); shift 2,922 across 28 classes.

| Metric | PlantVillage (test) | PlantDoc (shift) |
| --- | --- | --- |
| Accuracy | **0.9998** | **0.8036** |
| ECE at fitted `T` | **0.0004** | **0.1376** |
| Signed calibration error | **−0.0003** | **+0.1376** |
| NLL | — | — |
| Brier | — | — |

Fitted temperature: **0.7723** (fitted on validation, applied to both columns)

Three things in that table matter more than the headline accuracy.

**The shift column is where the experiment now lives.** ECE rises 344× from in-distribution
to shift, and the sign turns positive — overconfidence. The pilot had to *impose*
miscalibration with a temperature knob; here it arises on its own from the lab-to-field
gap, which is the stronger version of the same claim.

**Under shift, the signed error equals the ECE exactly** (both 0.13760501).
That identity holds only when every reliability bin errs in the same direction, so the
shift miscalibration is uniformly overconfident rather than a mix that partially cancels.
Worth stating, because the pilot's refuted H2 found that *direction* governs planner
performance, not magnitude.

**The specified BASE_ACC is 0.9998, and that choice now needs revisiting.** The split
table in `docs/METHOD.md`, section *"2. Perception"* → *"Data"*, assigns the `test` split
"early stopping, model selection, and the reported in-distribution accuracy that becomes
BASE_ACC". Measured, that is 0.9998 — replacing the pilot's cited 0.816 (Ahmad et al.,
2023) with a value at the ceiling. The alternative was not anticipated when that was
written:

| Candidate | Value | Argument for it |
| --- | --- | --- |
| PlantVillage test | 0.9998 | The model's accuracy on the data it was trained for |
| PlantDoc shift | 0.8036 | The accuracy a deployed scout would actually see; near the pilot's 0.816 |

At 0.9998 accuracy and 0.0004 ECE there is almost no uncertainty left for a planner to
consume, so an in-distribution planner sweep risks measuring nothing. This is an open
decision, recorded here rather than settled silently; it must be resolved and justified
in `docs/METHOD.md` before the planner sweep is interpreted.

> ⚠️ **Not yet checked: augmented-duplicate leakage.** 0.9998 is consistent with the
> literature on PlantVillage, which is lab imagery and famously easy, so it is not on its
> own evidence of a bug. But the only thing preventing an augmented copy of one leaf from
> straddling train and test is `_source_key` in `scoutfield/perception/datasets.py`, and
> that has not been verified against this particular mirror of the dataset. Verify before
> the number appears in a paper.

NLL and Brier at the fitted temperature: **0.0010** / **0.0002** in-distribution,
**0.8057** / **0.1625** under shift.

### Temperature sweep

`T` is **relative to the fitted temperature** (`sweep_is_relative_to_fitted_temperature:
true`), so `T` = 1 is the calibrated point, exactly as in the pilot. Absolute temperature
is `T` × 0.7723.

Accuracy must be identical down each accuracy column. It is, to six decimal places, on
all three splits — the instrument works on a real CNN.

**PlantVillage (test), n = 8,146**

| `T` | Accuracy | ECE | Signed error | Mean confidence |
| --- | --- | --- | --- | --- |
| 0.30 | 0.999754 | 0.0002 | +0.0002 | 0.9999 |
| 0.50 | 0.999754 | 0.0002 | +0.0001 | 0.9999 |
| 1.00 | 0.999754 | 0.0004 | −0.0003 | 0.9995 |
| 2.00 | 0.999754 | 0.0031 | −0.0029 | 0.9968 |
| 3.00 | 0.999754 | 0.0111 | −0.0110 | 0.9888 |
| 4.00 | 0.999754 | 0.0258 | −0.0257 | 0.9740 |

**PlantDoc (shift), n = 2,922**

| `T` | Accuracy | ECE | Signed error | Mean confidence |
| --- | --- | --- | --- | --- |
| 0.30 | 0.803559 | 0.1771 | +0.1771 | 0.9806 |
| 0.50 | 0.803559 | 0.1658 | +0.1658 | 0.9693 |
| 1.00 | 0.803559 | 0.1376 | +0.1376 | 0.9412 |
| 2.00 | 0.803559 | 0.0863 | +0.0862 | 0.8898 |
| 3.00 | 0.803559 | 0.0414 | +0.0414 | 0.8450 |
| 4.00 | 0.803559 | 0.0080 | +0.0028 | 0.8064 |

**The two splits move in opposite directions, and that is the finding.** In-distribution,
ECE is near-minimal at `T` = 1 and grows as the model is softened — the ordinary picture,
and the one the pilot's invariant *"ECE is minimised at `T` = 1 and rises in both
directions"* describes. Under shift, ECE **falls monotonically** across the sweep and is
minimised at `T` = 4, a 17× reduction from `T` = 0.3.

That is not the invariant breaking. It is the temperature fitted on in-distribution
validation data failing to transfer: under shift the model is overconfident at every
temperature tested (signed error positive throughout), and softening it by a further
factor of ~4 — absolute `T` ≈ 3.09 — is what brings mean confidence, 0.8064, into line
with accuracy, 0.8036.

Two consequences worth stating before the planner sweep:

- **A single fitted temperature does not serve both regimes.** Any deployed system that
  calibrates on lab data and flies over a field carries roughly the miscalibration in
  the `T` = 1 row of the shift table, not the test table.
- **The in-distribution sweep spans an ECE range of 0.0002 to 0.0258.** The pilot's
  planner effect was driven by a range of 0.0037 to 0.1980. The in-distribution
  condition may therefore be too well-calibrated to move a planner at all, and the shift
  condition is where the effect has room to appear. This bears directly on the open
  BASE_ACC question above.

Not yet measured: MC-dropout and deep ensembles (section *"5. Uncertainty methods"*),
and reliability diagrams per condition. `reliability_at_fitted` and `per_class_at_fitted`
are present in `results/calibration_sweep.json` but not yet plotted.

---

## 2. Environment

Roadmap item 4. Source: printed by `experiments/03_train_ppo.py`, `results/summary.json`.

| Quantity | Value |
| --- | --- |
| Coverage ceiling (optimistic bound) | 0.391 (analytic, from config) |
| Measured mean coverage, lawnmower | **0.391** (5 seeds, matches the bound) |
| Measured mean coverage, PPO | **0.350** (3 seeds, final 10% of training) |
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

### PPO training

Source: `results/ppo_curve_seed{0,1,2}.json`, 2,000,000 timesteps per seed at `T` = 1.
These are **training** episodes, not the evaluation sweep — the tables below are still
unmeasured. IQM over the first and last 10% of episodes.

| Quantity | First 10% | Last 10% | Change |
| --- | --- | --- | --- |
| Detections per joule | 0.0950 | 0.1492 | **1.57×** |
| Recall | 0.354 | 0.549 | 1.55× |
| Coverage | 0.307 | 0.350 | 1.14× |
| Time to first detection | 1.48 | 0.99 | 0.67× |
| Precision | 1.000 | 1.000 | — |
| False alarms | 0 | 0 | — |

All three seeds report `converged: true`. Drift in detections/joule across the final 20%
of training is +2.0%, +0.14% and +0.32% — flat, by the standard this phase committed to.

**The agent learned selectivity, not just more flying.** Detections per joule rose 1.57×
while coverage rose 1.14×, so the gain comes from *which* cells were visited rather than
how many. That is the behaviour the 32×32 scale change and the 0.391 coverage ceiling
were introduced to make possible, and it is the regime the pilot's 12×12 field could not
reach. Seed-to-seed spread is small: 1.55×, 1.56×, 1.60×.

> ⚠️ **Precision is exactly 1.000 and false alarms exactly 0 — in every episode, on every
> seed.** This is not a bug, and it is a problem for the experiment. At `T` = 1 the
> classifier is 0.9998-accurate in-distribution with ECE 0.0004, so at τ = 0.75 a false
> positive essentially cannot occur. The precision/recall trade-off is therefore
> unmeasurable in this condition.
>
> The pilot's central mechanism was that overconfidence *trades precision for recall*,
> and its refuted H2 concluded that the direction of miscalibration governs planner
> performance rather than its magnitude. Neither claim can be tested against a planner
> whose precision is pinned at 1.0 by construction. Together with the in-distribution ECE
> range of 0.0002–0.0258 against the pilot's 0.0037–0.1980, this is the second
> independent indication that the in-distribution condition is degenerate for this
> study's purpose, and that the shift condition is where the effect has room to appear.
> This must be settled before the sweep in section *"3. Planners"* is interpreted — see
> the BASE_ACC decision in section *"1. Perception"*.

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
