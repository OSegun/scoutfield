# Method

Every parameter, its value, and the citation or measurement behind it — so the
implementation can be checked against the method without reading the code.

Values here mirror `configs/*.yaml`. **The YAML is the source of truth**; if the two
disagree, the YAML is right and this file is stale. Keep them in step in the same change.

---

## 1. The instrument

The independent variable is **calibration error at fixed accuracy**.

Temperature scaling divides the logits by a scalar `T` before the link function. Because
the link is strictly monotone and the decision threshold sits at logit 0, dividing by any
`T > 0` leaves the arg-max — and therefore the accuracy — exactly unchanged, while moving
the reported confidence. Frenkel and Goldberger (2022) state this property explicitly.

```
T = 1   calibrated (relative to the fitted reference temperature)
T < 1   overconfident
T > 1   underconfident
```

**The sweep is applied relative to a fitted temperature.** The pilot's surrogate
reported the exactly calibrated posterior at `T = 1` by construction. A real network's
raw logits are not calibrated, so its ECE bottoms out at the fitted `T*`, not at 1.
Sweeping the raw temperature would therefore put the calibrated point somewhere other
than 1, and "T = 4" would denote a different manipulation in each phase. The effective
divisor is `T* × T`, so sweep `T = 1` is the calibrated point exactly as in the pilot.
`CNNClassifier` takes `reference_temperature = T*` and applies this internally; the
`observe()` signature is unchanged.

**Two reference temperatures are fitted, and the planner sweep uses the shift one.**
A temperature fitted on laboratory validation data does not transfer to field imagery:
under the PlantVillage-to-PlantDoc shift the network is overconfident at the
lab-fitted point, so a grid centred there never reaches underconfidence, the half of
the pilot's grid where most of its effect lay. Stage 02 therefore fits:

| Reference | Fitted on | Used for |
| --- | --- | --- |
| `lab_fitted` | PlantVillage `val` | the in-distribution sweeps, and the lab-calibrated condition below |
| `shift_fitted` | PlantDoc `shift_fit` | the planner sweep, PPO training and the τ sweep (`reference_condition: shift_fitted` in `configs/sweep.yaml` and `configs/ppo_field32.yaml`) |

Relative to `shift_fitted`, `T = 1` is calibrated on field imagery, so the grid
`0.3 … 4` straddles it as the pilot's did. Stage 02 asserts this: on `shift_eval` the
signed calibration error must be positive at every `T < 1` and negative at every
`T > 1`, or the run stops.

**The lab-calibrated condition.** One extra condition (`named_conditions: [lab_fitted]`)
runs every planner at effective divisor = the lab-fitted temperature on the same
`shift_eval` pool: a system calibrated in the lab and flown over a field. Its rows are
labelled `reference_condition = lab_fitted` in the CSV and are kept apart from the
temperature grid in the summary.

This is why the study is possible with no budget: a real CNN cannot give this
counterfactual, because accuracy and calibration co-vary under retraining. Only a
post-hoc scalar separates them.

**The invariant, asserted in code, not assumed:**

```python
accs = {evaluate(model, T=t)["accuracy"] for t in (0.3, 1.0, 4.0)}
assert len(accs) == 1, f"instrument broken: {accs}"
```

A failure means the temperature is applied after a softmax, or the hard prediction is
recomputed from already-scaled values. It is not a tolerance issue — the invariance is
exact, to floating-point equality of the arg-max.

---

## 2. Perception

`configs/perception.yaml`

| Parameter | Value | Why this value |
| --- | --- | --- |
| Backbone | EfficientNet-B0, ImageNet-pretrained | smallest member of a family with strong transfer; fine-tunes in minutes on a free-tier GPU; exports cleanly to ONNX. The binding constraint is a weekly Kaggle quota, so a backbone needing a day of training is out of scope regardless of accuracy. |
| Head | `Dropout(0.3) → Linear(1280, 1)` | one logit, decision boundary at 0, matching the pilot's scalar instrument exactly. Dropout must be a real `nn.Module` so it can be reactivated at inference for MC-dropout. |
| Output | single raw logit, no sigmoid in the model | the calibration code needs the raw logit; a sigmoid buried in the model is the most likely cause of the invariance failing. |
| Image size | 224 | standard EfficientNet-B0 input; changing it changes the pretrained receptive field. |
| Optimizer | AdamW, head 3e-4 / backbone 3e-5 | discriminative rates: a randomly initialised head at full LR against a pretrained backbone destroys pretrained features in the first steps. |
| Backbone freeze | 1 epoch | lets the head reach a sane initialisation before the backbone moves. |
| Schedule | cosine, 12 epochs | fits inside one Kaggle session with margin for a restart. |
| Weight decay | 1e-4 | standard; not tuned. |
| Label smoothing | **0.0** | it is a standard accuracy trick that directly alters confidence calibration — the dependent variable. Enabling it would confound the measurement. If ever enabled it must be swept and reported. |
| Batch size | 64 | fits a T4 at 224px with AMP. |
| AMP | on when CUDA available | quota, not accuracy. Verify it does not shift ECE before trusting a calibration number from an AMP run. |

### Data

| | Dataset | Role |
| --- | --- | --- |
| Train / val / test | PlantVillage | lab-condition imagery: uniform background, controlled lighting, single centred leaf |
| Shift test | PlantDoc | field imagery: cluttered backgrounds, variable lighting, occlusion |

Training on one and testing on the other is the distribution shift the study needs. Deep
classifiers are miscalibrated in-distribution and degrade *further* under shift, so the
PlantDoc evaluation is where the calibration story becomes non-trivial.

**Never train on PlantDoc.** It is the held-out test and contaminating it destroys the
only shift measurement this project has.

Both datasets are multi-class over (crop, condition) pairs and are collapsed to
healthy / diseased to match the pilot's binary environment. Keep the fine-grained label
alongside the binary one: per-class calibration explains *why* the model is miscalibrated
in a way an aggregate ECE cannot.

**The two datasets name their classes incompatibly, and the collapse is not a regex.**
PlantVillage uses `<Crop>___<Condition>` and names its healthy class explicitly
(`Tomato___healthy`). PlantDoc never uses the word "healthy" at all: its healthy classes
are `<Crop> leaf` (`Apple leaf`, `Tomato leaf`, `grape leaf`) and its diseased classes
carry the condition in the name (`Apple Scab Leaf`, `Tomato mold leaf`, `grape leaf black
rot`). Collapsing on the string "healthy" alone would mark every PlantDoc image diseased,
turning accuracy-under-shift into the diseased prevalence and producing a number credible
enough to publish. PlantDoc's 28 classes are therefore enumerated explicitly in
`scoutfield/perception/datasets.py` (`PLANTDOC_HEALTHY`, `PLANTDOC_DISEASED`), verified
against the canonical release (Singh et al., 2020) — 28 classes in `train/`, 27 in
`test/`, which omits *Tomato two spotted spider mites leaf*. An unrecognised directory
name raises instead of defaulting, and `build_dataloaders` refuses to return a shift split
that is entirely one class.

**One PlantVillage rendering, not three.** PlantVillage is distributed as three parallel
copies of the same photographs — `color`, `grayscale`, `segmented`. `data.plantvillage_variant`
selects one, default `color`. Using all three counts every leaf three times and lets one
photograph's renderings straddle the train/val boundary. `color` specifically, because it
is the only rendering consistent with both PlantDoc's RGB field imagery and an
ImageNet-pretrained backbone, and because `segmented` removes the background — the very
cue that fails under shift, so training on it would flatter the model.

**PlantDoc's own train/test split is merged, then halved for this design.** PlantDoc's
internal split carries no meaning here. All of it is pooled and then split into two
halves, stratified by binary label so both keep PlantDoc's class balance
(`calibration.shift_fit_fraction` = 0.5, `calibration.shift_split_seed` = 0 in
`configs/perception.yaml`):

| Split | Role |
| --- | --- |
| `shift_fit` | fits the shift temperature; PPO trains on its logit pool |
| `shift_eval` | every planner is scored on its logit pool; BASE_ACC is its accuracy |

No image both fits the calibration point and is evaluated against it, and no policy is
scored on the logits it trained on.

**Split by source image and by content, not by file.** Augmented PlantVillage variants
circulate widely; the same leaf appearing in train and val inflates accuracy and makes the
model look better calibrated than it is. Images are grouped before splitting, and two
images join a group when they share a source key (the filename with augmentation
suffixes removed) **or** have byte-identical contents (SHA-1 of the file). Links chain, so
a group is a connected component. The same grouping is applied to the PlantDoc halves.
`experiments/00_check_leakage.py` runs before fine-tuning and counts source keys and
identical files found in more than one split (both must be zero), identical files shared
between PlantVillage and PlantDoc, and what a split by source key alone would have left
on both sides. Its report is `leakage_report.json` in stage `01_perception`.

**Validation fits the temperature, so it cannot also drive early stopping.** Doing both
leaks the calibration fit into model selection. PlantVillage is therefore split three
ways, not two:

| Split | Fraction | Role |
| --- | --- | --- |
| `train` | remainder | fine-tuning |
| `val` | `val_fraction` = 0.15 | fits the temperature, and nothing else |
| `test` | `test_fraction` = 0.15 | early stopping, model selection, and the reported in-distribution accuracy |

#### BASE_ACC

**BASE_ACC is the PlantDoc `shift_eval` accuracy**, read from
`results/perception_summary.json` (`accuracy_shift_eval`), not the accuracy on the
PlantVillage `test` split. Three reasons, in order of weight:

1. **It is the accuracy the sweep actually runs at.** `configs/sweep.yaml` sets
   `pool_split: shift_eval`, so every planner number in `docs/RESULTS.md` is produced against a
   classifier operating at the shift accuracy. A BASE_ACC drawn from a split the
   experiment does not use would describe nothing that was measured.
2. **The in-distribution condition is degenerate for this study's question.** PlantVillage
   is controlled laboratory imagery and a fine-tuned network sits near the accuracy
   ceiling on it, with near-zero calibration error. A false positive then essentially
   cannot occur at τ = 0.75, so the precision-for-recall trade this study exists to
   measure has no room to appear.
3. **It keeps the two phases comparable.** The pilot pinned BASE_ACC at 0.816 from Ahmad et
   al. (2023), the accuracy of a classifier in deployment rather than in the lab. The shift
   accuracy is the matching operating point here, so a difference between the phases is
   attributable to the classifier rather than to a change of operating point.

The `test` split is therefore the **control** condition: it establishes that the
temperature instrument works on a real network — accuracy invariant across the sweep —
and that the model is well calibrated where it was trained. It is not the experimental
condition, and its accuracy is not BASE_ACC.

### Calibration

| Parameter | Value | Source |
| --- | --- | --- |
| Temperature fit | LBFGS on log-T, NLL; lab on PlantVillage `val`, shift on PlantDoc `shift_fit` | Guo et al. (2017) |
| Shift split | `shift_fit_fraction` = 0.5, `shift_split_seed` = 0, stratified by binary label | the fit needs one scalar, the evaluation needs a pool per class; half each way |
| Temperature sweep | 0.30, 0.50, 1.00, 2.00, 3.00, 4.00 | matches the pilot exactly, so the two phases are directly comparable |
| Sweeps written | `val`, `test` relative to lab; `shift_eval` relative to shift; `shift_eval` relative to lab | the third is the planner condition, the fourth the lab-calibrated one |
| ECE bins | 15, equal-width | matches the pilot's convention. Two ECE implementations with different binning give different numbers; comparing this phase's ECE to the pilot's 0.0037 / 0.1980 only means something if both were computed the same way — so import the pilot's implementation rather than rewriting it. |
| MC-dropout samples | 30 | diminishing returns beyond ~30 for a single dropout layer |
| Ensemble members | 5 | ~10 GPU-minutes each is affordable within a weekly quota; more is not. State the constraint rather than leaving the choice unexplained. Members must differ in initialisation *and* data order — ensembling checkpoints that differ only in data order understates real ensemble diversity. |

Beyond ECE, report **signed** calibration error. ECE takes an absolute value and discards
the sign, and the pilot's refuted hypothesis H2 is precisely a finding about direction
mattering more than magnitude: overconfidence trades precision for recall, while
underconfidence loses both.

### The logit pool, and its one limitation

The environment cannot run a network per observation: a scouting episode makes hundreds of
observations and the sweep runs thousands of episodes. Instead the network is run once over
a split, the logits are cached by true label, and `CNNClassifier.observe(label)` draws from
that cache. Temperature acts only on logits, so caching is exact rather than approximate —
no fidelity is lost relative to running the network each time.

Which split the cache comes from **is the experimental condition**. Stage 02 builds three
pools: `shift_fit` (PPO trains on it), `shift_eval` (every planner is scored on it) and
`test` (in-distribution PlantVillage, the control). Training and scoring use the two
halves of one field condition, so PPO is not scored on the logits it trained on, and is
not the only planner evaluated away from the distribution it was fitted on.

Each pool file carries a hash of its split's sorted member list, and
`calibration_sweep.json` records the hash each split should have. A pool whose hash does
not match is rebuilt from the stage 01 checkpoint, and the rebuild must then match the
recorded hash, or the run stops: the data on that machine is not the data the run used.
The pool and condition are written into every row of `results/sweep_results.csv`.

**Limitation: the pool is finite, and small on the shift split.** PlantDoc holds a few
thousand images, with healthy leaves the minority class, while a single episode can
observe up to about 400 cells (the coverage ceiling, 0.391 × 1024, section
*"3. Environment"*). Draws are with
replacement, so an episode reuses a meaningful fraction of the healthy pool and two
episodes are not fully independent in the logits they see.

This is a bootstrap from the empirical logit distribution rather than draws from a
generative model, and it is a departure from the pilot, whose Gaussian surrogate could
supply unlimited independent draws. The trade is deliberate — the empirical distribution
is the real classifier's, including whatever asymmetry and heavy tails it has, which is
the point of this phase — but it means confidence intervals over episodes understate
variance slightly, and a result resting on differences smaller than the pool's granularity
should not be trusted. Report the pool sizes alongside any such result; they are in
`calibration_sweep.json` under `pools`.

---

## 3. Environment

`configs/ppo_field32.yaml`. Inherited from the pilot unless noted.

| Parameter | Pilot | Here | Why the change |
| --- | --- | --- | --- |
| Grid | 12×12 | **32×32** | the pilot's budget permitted near-total coverage, so there was nothing for selectivity to buy and a fixed lawnmower sweep was near-optimal by construction |
| Budget | 190.0 | **640.0** | 640 / (1.0 + 0.6) = 400 steps over 1024 cells → optimistic ceiling 0.391. Diagonal moves cost `translate × √2` and revisits are unavoidable, so measured coverage is lower. Verify empirically. |
| Prior | 0.15 | 0.15 | unchanged |
| Patch | 5×5 | 5×5 | unchanged |
| τ (confirmation) | 0.75 | 0.75, **swept** | see the τ note below |
| Observation | 5×5 patch + 3 scalars | **+ 8×8 global belief map** | an agent that cannot see past its neighbourhood cannot plan a route to a distant high-belief region; it can only hill-climb. Fixed 8×8 so obs dimension does not grow with field size. Downsample by **mean** pooling, not max — max lets one confident cell dominate a region and hides the uncertainty structure the planner exists to exploit. Pad with the prior, not zeros: zero asserts "certainly healthy", a claim with no evidence. |
| Energy | `hover 1.0`, `translate 0.6` | unchanged | changing them would break comparability with the pilot |
| Reward | `α=1.0, λ=6.0, μ=0.15` | unchanged | information gain + detection bonus − energy cost. **Freeze this.** Reward shaping consuming weeks is the project's top identified risk. |
| Disease field | Neyman–Scott cluster process | unchanged | field epidemiology reports aggregated, not uniform, patterns (Heck et al., 2021). `sigma` is the aggregation dial, and aggregation is *measured* via the Morisita index rather than asserted. |
| `n_parents` | 3 | **18** | holds prevalence constant across the scale change — see below |
| `offspring_mean` | 14 | 14 | unchanged |

**Prevalence is held constant across the scale change, and this is not cosmetic.** The
Neyman–Scott parameters generate an absolute *count* of diseased cells, not a density.
Inheriting the pilot's `n_parents = 3` at 32×32 would drop prevalence to a small fraction
of the pilot's, a change arriving silently alongside the grid change, so that two things
move at once and neither result is attributable. `n_parents = 18` holds the pilot's
prevalence (0.1697); `tests/test_env_parity.py`
(`test_scaled_field_holds_the_pilot_prevalence`) measures it over 30 seeds on every test
run.

Note 18, not the 21 that naive area scaling (1024 / 144 = 7.1×) suggests: clusters overlap
more at higher density, so the value is measured rather than derived.

This matters because prevalence moves planner performance on its own: a prevalence
change can reorder the planners, and would then be reported as a calibration effect.

**Coverage ceiling, verified empirically rather than assumed:** the analytic bound for
32×32 at budget 640 is 0.3906. Measured Lawnmower coverage is read from
`results/summary.json` and must sit at or just below that bound for the intended ≤40%
regime to hold.

**Belief update.** Bayesian, via likelihood ratio. This is what makes miscalibration a
*systematic bias in belief dynamics* rather than additive noise — and that mechanism is
the paper's contribution, not an implementation detail.

**Termination.** Episodes end when the energy budget cannot fund another move. That is a
genuine terminal state: there is no future return to bootstrap. In the Gymnasium wrapper
it is `terminated=True, truncated=False`. Getting this backwards makes SB3 bootstrap a
value that does not exist — a bug that still trains, still improves, and quietly caps
final performance.

### The τ caveat

τ interacts with `T` **by construction**. A detection is confirmed when posterior belief
exceeds τ; temperature scaling moves reported confidence, so it moves how often belief
crosses any fixed threshold. Raising `T` pushes probabilities toward 0.5 and makes
crossing τ = 0.75 rarer; lowering it makes crossing commoner.

Some portion of the pilot's 13.4× collapse is therefore attributable to the threshold
rather than to miscalibration. Until `experiments/05_tau_sensitivity.py` runs over
τ ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.85, 0.90}, **13.4× may be quoted only as "at
τ = 0.75"**.

---

## 4. Planners

| Planner | Source | Role |
| --- | --- | --- |
| Lawnmower | pilot, imported | fixed boustrophedon sweep; the baseline that beat everything in the pilot |
| Random | pilot, imported | uniform random move; the pilot ships no Spiral agent |
| GreedyEntropy | pilot, imported | one-step myopic information gain; the robust performer (1.84× degradation) |
| PPO | here | Stable-Baselines3, `MlpPolicy`, `[256, 256]` |
| Oracle | here | sees true labels; defines regret |

Baselines are **imported unchanged** from the pilot. Reimplementing them would mean
comparing against this project's version of the baseline rather than the published one.

### PPO

| Parameter | Value | Note |
| --- | --- | --- |
| `total_timesteps` | 2,000,000 | "converged" means the evaluation curve is flat over the last ~20% on **every** seed, not that a step count was reached |
| `n_envs` | 8 | `SubprocVecEnv`, each seeded distinctly — sharing an env across subprocesses correlates runs and silently narrows confidence intervals |
| `n_steps` / `batch_size` | 512 / 512 | |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | |
| `clip_range` | 0.2 | Schulman et al. (2017) default |
| `ent_coef` | 0.01 | detection events are rare; premature entropy collapse is the failure to watch for |
| `learning_rate` | 3e-4 | |
| Seeds | ≥3 for the learning curve, 5 for evaluation | a single seed's curve is not evidence of convergence |
| Inference device | CPU | evaluation is thousands of short episodes where per-step transfer latency dominates; GPU is slower here and CPU keeps the sweep runnable without quota |

### Oracle, and the honesty about it

Optimal routing under an energy budget is a variant of the orienteering problem and is
NP-hard, so a genuinely optimal oracle is out of reach at 32×32. The implemented oracle is
greedy nearest-diseased-cell, which is a **lower bound** on the true ceiling and must be
labelled as such wherever it appears. Overstating it as "optimal" would understate every
planner's regret — the flattering direction, and reason enough to be careful.

The oracle still confirms detections *through the classifier*, so it inherits the same
miscalibration the planners face. That isolates the routing ceiling from the perception
ceiling.

---

## 5. Statistics

`configs/sweep.yaml`

| Choice | Value | Why |
| --- | --- | --- |
| Estimator | interquartile mean | RL returns are heavy-tailed and multimodal; one lucky seed drags a mean. Agarwal et al. (2021) |
| Interval | stratified bootstrap, 10,000 resamples, 95% | stratify by seed: runs sharing a seed share a field layout and classifier draws, so treating them as independent gives intervals that are too narrow — the direction that manufactures significance |
| Also report | probability of improvement | "GreedyEntropy beats Lawnmower 7.4×" is a ratio of central tendencies and says nothing about how often it wins on an individual field. A practitioner choosing a planner cares about the latter. Agarwal et al. (2021) |
| Seeds | 5 | Colas et al. (2018, 2019) |

**Never mean ± std over three seeds.** It is the commonest reason RL results fail to
replicate.

### Metrics

Declared 13 November 2026, before the full re-run.

| Metric | Role | Definition, per episode, then averaged over a run's episodes |
| --- | --- | --- |
| Detections per joule | **primary**, as in the pilot | confirmed true positives / energy spent |
| F1 | secondary | 2 × precision × recall / (precision + recall); 0 when both are 0 |
| False alarms per joule | secondary | confirmed false positives / energy spent |

Detections per joule counts true positives only, so an overconfident classifier, which
confirms more cells, can raise it while confirming more healthy cells as diseased. The two
secondary metrics expose that cost: F1 brings precision back in, and false alarms per joule
prices the false confirmations on the same energy scale as the primary metric. Both are
reported beside the primary metric; neither replaces it.

### Degradation spans

The detections-per-joule ratio is reported over three spans of the temperature grid
(`degradation_spans` in `configs/sweep.yaml`), each as the IQM at the lower temperature
over the IQM at the higher, with a stratified bootstrap interval on the ratio and the
probability that a run at the lower temperature beats a run at the higher:

| Span | Temperatures | Pilot Lawnmower | Pilot GreedyEntropy |
| --- | --- | --- | --- |
| full | 0.3 → 4 | 13.41× | 1.84× |
| overconfident | 0.3 → 1 | 1.23× | 1.59× |
| underconfident | 1 → 4 | 10.86× | 1.16× |

The pilot values are from `scoutplan/results/summary.json` at `v1.0.0` (`dpj_iqm`). Most of
the pilot's collapse lies on the underconfident half, so a comparison between the phases is
like-for-like only span by span.

### Pre-registered predictions

Recorded 13 November 2026, before the run that tests them. Each is judged on the
`shift_eval` pool, relative to the shift-fitted temperature, at τ = 0.75, from the
detections-per-joule IQM in `summary.json` (`degradation` for P1–P3, `cells` for P4,
`conditions` for P5). The decision rule for each is fixed here, and each will be reported
as held or not held, whichever it turns out to be.

| | Prediction | Pilot value | Held if |
| --- | --- | --- | --- |
| P1 | On the overconfident half (T 0.3 → 1), Lawnmower detections per joule falls modestly, and GreedyEntropy falls more. | Lawnmower 1.23×, GreedyEntropy 1.59× | both ratios are above 1, and GreedyEntropy's ratio is above Lawnmower's |
| P2 | On the underconfident half (T 1 → 4), Lawnmower detections per joule falls sharply. How sharply, with a real classifier, is the question this phase answers. | Lawnmower 10.86× | the lower bound of Lawnmower's 95% interval on the underconfident ratio is above 1, and its ratio is above its overconfident ratio |
| P3 | On the underconfident half, GreedyEntropy is less sensitive than Lawnmower. | GreedyEntropy 1.16× against Lawnmower 10.86× | GreedyEntropy's underconfident ratio is below Lawnmower's |
| P4 | PPO, trained at the shift-calibrated point, does not beat Lawnmower on detections per joule. | the pilot's REINFORCE agent lost at every temperature | at T = 1, PPO's 95% interval does not lie entirely above Lawnmower's |
| P5 | The lab-calibrated condition has lower precision than the shift-calibrated condition, for every planner. | not measured in the pilot | for every planner, precision under `lab_fitted` is below precision at T = 1 under `shift_fitted` |

P2 predicts a direction, not a size: the pilot's 10.86× came from a Gaussian surrogate, and
nothing here predicts that a real classifier reproduces it. Every ratio is reported with
its interval whether or not its prediction held.

---

## 6. Where results live

Every stage reads its inputs from, and writes its outputs to, one public Hugging Face
dataset repository named in `configs/artifacts.yaml`, under `runs/<run_id>/<stage>/`. Each
run's `manifest.json` records every file's SHA-256, the commit of this repository that
produced it, and the files each stage read, so every number traces to its exact inputs.
`scripts/pull_artifacts.py --run <run_id>` mirrors a run locally;
`scripts/verify_stage.py <stage>` checks one.

---

## 7. References

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2021).
  Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS 34*.
- Ahmad, A., Saraswat, D., & El Gamal, A. (2023). A survey on using deep learning
  techniques for plant disease diagnosis and recommendations for development of
  appropriate tools. *Smart Agricultural Technology, 3*, 100083.
- Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2018). How many random seeds? Statistical
  power analysis in deep reinforcement learning experiments. *arXiv:1806.08295*.
- Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2019). A hitchhiker's guide to statistical
  comparisons of reinforcement learning algorithms. *arXiv:1904.06979*.
- Frenkel, L., & Goldberger, J. (2022). Calibration of medical imaging classification
  systems with weight scaling. *MICCAI 2022*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern
  neural networks. *ICML 34*.
- Heck, D. W., et al. (2021). Spatial pattern analysis of plant disease epidemics.
  *Phytopathology*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal
  policy optimization algorithms. *arXiv:1707.06347*.
- Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A
  dataset for visual plant disease detection. *Proceedings of the 7th ACM IKDD CoDS and
  25th COMAD*, 249–253. https://doi.org/10.1145/3371158.3371196
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional
  neural networks. *ICML 36*.

The pilot's full reference list (58 entries) is in the companion repository.
