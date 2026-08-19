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
T = 1   perfectly calibrated (fitted on validation)
T < 1   overconfident
T > 1   underconfident
```

**The sweep is applied relative to the fitted temperature.** The pilot's surrogate
reported the exactly calibrated posterior at `T = 1` by construction. A real network's
raw logits are not calibrated, so its ECE bottoms out at the fitted `T*`, not at 1.
Sweeping the raw temperature would therefore put the calibrated point somewhere other
than 1, and "T = 4" would denote a different manipulation in each phase. The effective
divisor is `T* × T`, so sweep `T = 1` is the calibrated point exactly as in the pilot.
`CNNClassifier` takes `reference_temperature = T*` and applies this internally; the
`observe()` signature is unchanged.

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

**PlantDoc's own train/test split is merged.** PlantDoc is the held-out shift set in its
entirety, so its internal split carries no meaning in this design, and using half of it
would discard data for nothing.

**Split by source image, not by file.** Augmented PlantVillage variants circulate widely;
the same leaf appearing in train and val inflates accuracy and makes the model look better
calibrated than it is.

**Validation fits the temperature, so it cannot also drive early stopping.** Doing both
leaks the calibration fit into model selection. PlantVillage is therefore split three
ways, not two:

| Split | Fraction | Role |
| --- | --- | --- |
| `train` | remainder | fine-tuning |
| `val` | `val_fraction` = 0.15 | fits the temperature, and nothing else |
| `test` | `test_fraction` = 0.15 | early stopping, model selection, and the reported in-distribution accuracy |

#### BASE_ACC — settled 19 August 2026

**BASE_ACC = 0.8036, the PlantDoc shift accuracy.** Not the 0.9998 measured on the
PlantVillage `test` split, which an earlier draft of the row above assigned to the role.
Three reasons, in order of weight:

1. **It is the accuracy the sweep actually runs at.** `configs/sweep.yaml` sets
   `pool_split: shift`, so every planner number in `docs/RESULTS.md` was produced against a
   classifier operating at 0.8036. A BASE_ACC drawn from a split the experiment does not
   use would describe nothing that was measured.
2. **The in-distribution condition is degenerate for this study's question.** At 0.9998
   accuracy and ECE 0.0004, a false positive essentially cannot occur at τ = 0.75 —
   measured PPO precision is exactly 1.000 and false alarms exactly 0, on every episode of
   every seed. The precision-for-recall trade this study exists to measure is unmeasurable
   there. Its swept ECE range, 0.0002–0.0258, is also an order of magnitude below the
   pilot's 0.0037–0.1980.
3. **It keeps the two phases comparable.** The pilot pinned BASE_ACC at 0.816 from Ahmad et
   al. (2023). 0.8036 sits beside it, so a difference between the phases is attributable to
   the classifier rather than to a change of operating point.

The `test` split is therefore the **control** condition: it establishes that the
temperature instrument works on a real network — accuracy invariant to six decimal places —
and that the model is well calibrated where it was trained. It is not the experimental
condition, and its accuracy is not BASE_ACC.

### Calibration

| Parameter | Value | Source |
| --- | --- | --- |
| Temperature fit | LBFGS on log-T, NLL on validation | Guo et al. (2017) |
| Temperature sweep | 0.30, 0.50, 1.00, 2.00, 3.00, 4.00 | matches the pilot exactly, so the two phases are directly comparable |
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

Which split the cache comes from **is the experimental condition**, and the two are not
interchangeable: `test` is in-distribution PlantVillage (8,146 logits, accuracy 0.9998),
`shift` is PlantDoc field imagery (2,922 logits, accuracy 0.8036). Pools are stored under
split-keyed filenames so one cannot silently stand in for the other, the condition is
written into every row of `results/sweep_results.csv`, and training and evaluation must
use the same one — see `docs/RESULTS.md`, section *"3. Planners"*, for what happened when
they did not.

**Limitation: the pool is finite, and small on the shift split.** It holds 823 healthy and
2,099 diseased logits, while a single episode at ~0.35 coverage makes roughly 358
observations. Draws are with replacement, so an episode reuses a meaningful fraction of
the healthy pool and two episodes are not fully independent in the logits they see.

This is a bootstrap from the empirical logit distribution rather than draws from a
generative model, and it is a departure from the pilot, whose Gaussian surrogate could
supply unlimited independent draws. The trade is deliberate — the empirical distribution
is the real classifier's, including whatever asymmetry and heavy tails it has, which is
the point of this phase — but it means confidence intervals over episodes understate
variance slightly, and a result resting on differences smaller than the pool's granularity
should not be trusted. Report the pool sizes alongside any such result.

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
Inheriting the pilot's `n_parents = 3` at 32×32 drops prevalence from 0.170 to 0.030 — a
5.7× change arriving silently alongside the grid change, so that two things move at once
and neither result is attributable. Measured over 30 seeds, `n_parents = 18` reproduces
the pilot's prevalence (0.1700 against 0.1697).

Note 18, not the 21 that naive area scaling (1024 / 144 = 7.1×) suggests: clusters overlap
more at higher density, so the value is measured rather than derived.

This mattered. At the uncorrected prevalence the planner ranking inverts — GreedyEntropy
beat Lawnmower on detections/joule (0.0084 against 0.0069), while at held prevalence
Lawnmower wins (0.0564 against 0.0325). A prevalence confound would have been reported as
a calibration effect.

**Coverage ceiling, verified empirically rather than assumed:** the analytic bound for
32×32 at budget 640 is 0.3906, and a Lawnmower sweep measures 0.391 over 5 seeds. The
intended ≤40% regime holds.

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

---

## 6. References

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
