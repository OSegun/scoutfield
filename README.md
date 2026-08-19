# scoutfield

**Does a classifier's miscalibration change where a drone should fly?** The implementation
phase of a completed pilot study — replacing the pilot's synthetic confidence instrument
with a fine-tuned EfficientNet-B0 and its REINFORCE agent with PPO, to find out whether the
pilot's effect was real or an artefact of the surrogate.

[![tests](https://github.com/OSegun/scoutfield/actions/workflows/ci.yml/badge.svg)](https://github.com/OSegun/scoutfield/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![pilot](https://img.shields.io/badge/pilot-scoutplan%20v1.0.0-2C5F2D)](https://github.com/OSegun/scoutplan)

> **Status: research in progress, not a finished study.** Seven of the nine roadmap items
> below have produced measured numbers; two have not. Nothing here has been peer-reviewed,
> and the headline finding — that the pilot's effect size does **not** reproduce — is a
> result from one perception model on one dataset pair and should be read that way.
>
> The **completed** work is the pilot study: 195 executed runs, 8 figures, a full write-up,
> frozen in the companion repository [**scoutplan**](https://github.com/OSegun/scoutplan)
> and pinned here at `v1.0.0`. This repository extends it; it does not replace it.

---

## The idea in one paragraph

Informative path planning decides where to look next by maximising expected information
gain. It assumes the information signal it consumes is calibrated. Deep classifiers are
systematically miscalibrated, and degrade further under distribution shift. Nobody has
quantified how that calibration error propagates into planner performance — the question
falls between two literatures, because path planning assumes good uncertainty and the
calibration community never closes a control loop around it.

Temperature scaling makes the question answerable. Dividing logits by a scalar `T` is
strictly monotone, and the decision boundary sits at logit 0, so **accuracy cannot move
while confidence does.** Sweep `T`, hold accuracy fixed, and any change in planner
performance is attributable to calibration alone. That is the whole experimental design,
and it needs no budget.

## Where this repository stands

| Roadmap item | State | Evidence |
| --- | --- | --- |
| 1. Fine-tune EfficientNet-B0; hold out PlantDoc | measured | 0.9998 test / 0.8036 shift accuracy |
| 2. `CNNClassifier` behind the pilot's `observe()` | measured | contract tests pass with the CNN swapped in |
| 3. Temperature sweep on the real classifier | measured | accuracy invariant to 6 d.p. on all splits |
| 4. `FieldScoutEnv`: 32×32, ≤40% coverage | measured | coverage ceiling 0.391, attained by lawnmower |
| 5. PPO to convergence | measured | 3 seeds × 2M steps, flat over the final 20% |
| 6. τ sensitivity | measured | 7 thresholds; see the caveat below |
| 7. MC-dropout, deep ensembles, reliability diagrams | **not started** | — |
| 8. Oracle planner; regret | measured | greedy oracle as a lower bound on the ceiling |
| 9. ONNX export, latency, deployed interface | **partial** | graph equivalence and latency measured on *untrained* weights; no deployed interface |

Full numbers, with their confidence intervals and their caveats, are in
[`docs/RESULTS.md`](docs/RESULTS.md). Every number there is read from a generated summary
file; if the two disagree, the document is wrong.

---

## What changed when the surrogate was replaced

This is the reason the repository exists, so it goes first — including the part that is
unflattering to the pilot.

### The 13.4× collapse does not survive a real classifier

| | Pilot (surrogate) | This phase (EfficientNet-B0) |
| --- | --- | --- |
| Lawnmower degradation, `T`=1 → `T`=4 | **13.4×** | **1.27×** |
| GreedyEntropy degradation | 1.84× | 1.49× |
| Ranking reversal at `T`=4 | GreedyEntropy beats Lawnmower 7.3× | **no reversal at any `T`** |
| More robust planner | GreedyEntropy | Lawnmower |

The effect is real and in the same direction — miscalibration costs the planner detections
per joule at fixed accuracy — but an order of magnitude smaller, and the robustness
ordering **reverses**. The ECE range swept is comparable (pilot 0.0037–0.1980; here
0.0080–0.1771 on the shift split), so the difference is attributable to the perception
model rather than to the manipulation.

That is the finding. It was the outcome the pilot's own README named as acceptable —
*"should reproduce the qualitative result. If it does not, that is the finding"* — and it
is reported here without softening.

### The measured sweep

630 jobs: 5 planners × 6 temperatures × 3 cluster scales × 5 evaluation seeds, on the
PlantDoc shift split. IQM with 95% stratified bootstrap CIs, per Agarwal et al. (2021).

| Planner | `T`=0.3 | `T`=1.0 | `T`=4.0 | Degradation `T`=1→4 | n |
| --- | --- | --- | --- | --- | --- |
| Oracle (greedy; lower bound) | 0.2497 [0.2149, 0.2700] | 0.2478 [0.2142, 0.2671] | 0.2364 [0.2094, 0.2501] | 1.05× | 15 |
| Lawnmower | 0.0910 [0.0780, 0.1009] | 0.0870 [0.0746, 0.0964] | 0.0683 [0.0589, 0.0756] | 1.27× | 15 |
| PPO | 0.0872 [0.0799, 0.0920] | 0.0848 [0.0781, 0.0891] | 0.0663 [0.0616, 0.0693] | 1.28× | 45 |
| GreedyEntropy | 0.0738 [0.0616, 0.0817] | 0.0612 [0.0505, 0.0693] | 0.0410 [0.0343, 0.0467] | 1.49× | 15 |
| Random | 0.0391 [0.0333, 0.0437] | 0.0384 [0.0328, 0.0429] | 0.0350 [0.0299, 0.0393] | 1.10× | 15 |

<sub>Detections per joule. Each cell pools three cluster scales (σ ∈ {0.75, 1.5, 3.0});
PPO has three times the runs because three independently trained policies were evaluated.
Source: `results/summary.json`.</sub>

**PPO and the lawnmower are statistically indistinguishable.** The intervals overlap
heavily at every temperature, so the honest statement is a tie, not a loss. PPO reaches
within 3% of the lawnmower's recall while visiting 17% fewer cells (coverage 0.325 against
0.391), with higher precision and fewer false alarms — it is the more selective planner
without converting selectivity into an efficiency win. The pilot's REINFORCE agent *lost*
outright; changing algorithm, observation and field scale moved that to a tie.

A confound was found and closed rather than argued about: the first sweep trained PPO
in-distribution and evaluated it under shift. The policies were retrained on the shift
split and the sweep re-run in full. PPO moved from 0.0857 to 0.0848 detections per joule —
the mismatch was not the reason for the gap.

### The pilot's τ caveat is confirmed

The pilot flagged that its 13.4× was a magnitude *at* τ = 0.75, not a general one. Measured
across seven thresholds, Lawnmower's degradation runs from 1.02× at τ = 0.55 to 2.32× at
τ = 0.90 — it more than doubles. **No single effect size is quotable without its threshold
attached.** The dependence is also planner-specific, which the pilot did not predict:
GreedyEntropy is flat below τ = 0.85 while Lawnmower climbs monotonically, and the two
curves cross between τ = 0.75 and τ = 0.85.

### Known limitations of what is measured

Stated here rather than left for a reviewer to find.

- **The in-distribution condition is degenerate for this question.** At 0.9998 accuracy and
  0.0004 ECE, precision is pinned at 1.000 and false alarms at 0 for every PPO episode on
  every seed, so the precision–recall trade the pilot's mechanism rests on is unmeasurable
  there. The sweep therefore runs on the shift split, and in-distribution is the control.
- **PlantVillage leakage is unverified.** 0.9998 is consistent with the literature on that
  dataset, but the augmented-duplicate guard in `perception/datasets.py` has not been
  checked against this mirror. Until it is, treat the in-distribution accuracy as
  provisional.
- **The τ confidence intervals are currently degenerate.** With one run per seed at each
  τ, the stratified bootstrap resamples strata of size 1 and returns a zero-width interval.
  The τ ratios in `docs/RESULTS.md` are point estimates; the intervals in
  `results/tau_sensitivity_summary.json` should not be quoted until this is fixed.
- **The headline table pools three cluster scales.** The pilot found planner performance
  varies strongly with σ. Aggregating over σ is a deliberate choice for the temperature
  comparison, not a claim that σ does not matter.
- **The deployment numbers are architecture-level.** Latency and ONNX equivalence were
  measured on an untrained export, on a laptop CPU. They say nothing about the trained
  model's field behaviour.
- **No drone flew.** The field is simulated throughout, as in the pilot.

---

## Install

```bash
git clone https://github.com/OSegun/scoutfield.git
cd scoutfield
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install-dev
make test
```

`make test` runs the contract tests against the pinned pilot. If they fail on a fresh clone
the dependency pin is wrong — fix that before writing anything else.

## Usage

The library is importable and holds no CLI. Runnable entry points live in `experiments/`.

```python
from scoutfield.perception import CNNClassifier
from scoutfield.envs import FieldScoutEnv
from scoutfield.planners import PPOPlanner, GreedyEntropyAgent  # the latter is the pilot's

clf = CNNClassifier(checkpoint="checkpoints/perception/best.pt", temperature=4.0)
env = FieldScoutEnv(field, clf, budget=640.0, global_map_size=8)
```

```bash
python experiments/01_finetune_perception.py --config configs/perception.yaml
python experiments/02_calibrate.py           --config configs/perception.yaml
python experiments/03_train_ppo.py           --config configs/ppo_field32.yaml
python experiments/04_sweep.py               --config configs/sweep.yaml
python experiments/05_tau_sensitivity.py     --config configs/sweep.yaml
python experiments/make_figures.py
```

Same commands via `make train-perception`, `make calibrate`, `make train-ppo`, `make
sweep`, `make tau`, `make figures`. Reproducing any specific number:
[`docs/REPRODUCING.md`](docs/REPRODUCING.md).

---

## Architecture

```
                 scoutplan @ v1.0.0  (frozen pilot, pip-installed)
                 ┌────────────────────────────────────┐
                 │ field.py    Neyman-Scott field      │
                 │ env.py      ScoutEnv, Bayesian      │
                 │             belief, energy model    │
                 │ agents.py   Lawnmower, Random,      │
                 │             GreedyEntropy, REINFORCE│
                 │ perception.py  CalibratedClassifier │
                 └──────────────┬─────────────────────┘
                                │ imported, never copied
                 ┌──────────────▼─────────────────────┐
                 │ scoutfield                          │
                 │  perception/  EfficientNet-B0,      │
                 │               temperature scaling   │
                 │               (MC-dropout, ensembles│
                 │                still to come)       │
                 │  envs/        FieldScoutEnv(ScoutEnv)│
                 │  planners/    PPO (SB3), oracle     │
                 │  analysis/    IQM + bootstrap CIs   │
                 │  app/         ONNX, latency, UI     │
                 └─────────────────────────────────────┘
```

### The one interface that matters

The pilot's environment talks to perception through exactly one method:

```python
CalibratedClassifier.observe(true_label: int) -> tuple[float, int]
#                                                (probability, hard_prediction)
```

`scoutfield.perception.CNNClassifier` satisfies that same signature while backing it with
a real network. Everything downstream — belief update, energy accounting, reward, the four
baseline planners — is untouched. That is deliberate: if the result changes, the classifier
is the only thing that changed, so the comparison is clean. It is also what licenses the
comparison table above.
`tests/test_pilot_contract.py` fails if the signature drifts.

### Depend, don't fork

```
scoutplan @ git+https://github.com/OSegun/scoutplan.git@v1.0.0
```

Two rules follow, and both are load-bearing:

- **Never copy a file out of the pilot into this repo.** Two diverging copies of `env.py`
  with nobody able to say which is authoritative is the failure mode this structure
  prevents.
- **Never edit the pilot to make something here work.** Subclass instead.
  `FieldScoutEnv(ScoutEnv)` overrides `_obs()`; the published baseline stays exactly as
  described, and the override is visibly *this project's* contribution rather than a
  silent mutation.

> **Module names.** The pilot ships flat top-level modules, so after installation
> `import env`, `import agents` and `import perception` resolve to the *pilot*. Anything
> here is `scoutfield.<...>`. Keep that distinction visible in every import line.

---

## Layout

```
scoutfield/
├── scoutfield/                 the library — importable, small, no CLI
│   ├── perception/             datasets · model · train · metrics · calibrate · adapter
│   ├── envs/                   field_env · gym_wrapper
│   ├── planners/               ppo · oracle (baselines imported from the pilot)
│   ├── analysis/               stats — IQM, stratified bootstrap, P(improvement)
│   ├── app/                    export_onnx · inference · streamlit_app
│   └── utils/                  paths · seeding · checkpoint
│
├── experiments/                runnable entry points, one per roadmap stage
├── configs/                    YAML — no hyperparameter lives in code
├── notebooks/                  Kaggle kernels, git-tracked, synced via the API
├── docs/                       METHOD.md · REPRODUCING.md · RESULTS.md
├── scripts/                    kaggle_sync.py · make_notebooks.py
├── tests/                      contract tests against the pinned pilot
└── results/  figures/  checkpoints/       generated
```

`docs/METHOD.md` lists every hyperparameter and the citation behind it, so the
implementation can be checked against the method without reading the code.
`docs/REPRODUCING.md` maps each reported number to the exact command that produces it.
`docs/RESULTS.md` holds the numbers themselves and is written *from* the generated summary
files.

---

## Compute

Everything runs on free-tier Kaggle. No budget, no paid data, no purchased hardware.

Kaggle supplies the GPU and nothing else — the code lives here and is installed into the
notebook, never pasted into it. A notebook that contains logic is a bug.

Each notebook's first cell clones this repository and installs from the clone. Cloning
rather than only pip-installing is deliberate: `experiments/` and `configs/` are not
packaged, so a pip install alone would leave the drivers and the YAML configs missing.

```python
subprocess.run(["git", "clone", "--depth", "1", REPO, "/kaggle/working/scoutfield"])
# then, from inside the clone:
#   pip install -r requirements-kaggle.txt
#   pip install -e . --no-deps
```

The cell is generated, not hand-written — see `notebooks/_bootstrap.py` and
`configs/kaggle.yaml`.

Internet **on**, accelerator **GPU**, persistence **variables and files**. Sessions are
killed at nine hours and much sooner when idle, so every long job checkpoints per epoch or
per sweep job and resumes from disk. See `notebooks/README.md`.

```bash
pip install kaggle          # credentials at ~/.kaggle/kaggle.json
make notebooks-push         # local -> Kaggle
make notebooks-pull         # Kaggle -> local, executed outputs included
```

---

## Roadmap

Ordered by dependency. Each item names how you know it is done. ✅ measured · ◐ partial ·
☐ not started.

| # | Work | Done when | State |
| --- | --- | --- | --- |
| 1 | Fine-tune EfficientNet-B0 on PlantVillage; hold out PlantDoc | accuracy and per-class ECE reported on both sets; checkpoint reproducible from a seed | ✅ (leakage check outstanding) |
| 2 | `CNNClassifier` behind `observe()` | `make test` passes with the CNN swapped in; pilot baselines run unmodified | ✅ |
| 3 | Temperature sweep on the real classifier | accuracy invariant to 4 d.p. across `T` — or a documented explanation of why not | ✅ (invariant to 6 d.p.) |
| 4 | `FieldScoutEnv`: 32×32, ≤40% coverage, global belief map | coverage ceiling verified empirically; obs dimension asserted in a test | ✅ (0.391 attained) |
| 5 | PPO to convergence | curve plateaus over ≥3 seeds; beats or loses to lawnmower with a stated reason | ✅ (ties; reason stated) |
| 6 | τ sensitivity | effect size reported as a curve over τ, not a single number | ✅ (intervals need fixing) |
| 7 | MC-dropout + deep ensembles; reliability diagrams | three uncertainty methods compared on one axis | ☐ |
| 8 | Oracle planner; regret | every planner's gap to the ground-truth ceiling quantified | ✅ |
| 9 | ONNX export, measured latency, deployed interface | latency measured on stated hardware, not estimated | ◐ (untrained export only) |

**Standing risks.** Reward shaping can consume weeks — the reward is frozen and swept only
if a result demands it. Simulator realism will be challenged in review, so every physical
constant traces to a citation and every invented value gets swept.

---

## Ground rules

Inherited from the pilot, and the reason its numbers held up.

1. **Seed everything.** A run that does not reproduce from `(agent, seed, T, τ, σ)` is
   not a result.
2. **Baselines get the same observation and pay the same energy.** Never advantage the
   proposed method through the interface.
3. **No number is hard-coded into a document.** Results are read from generated summary
   files, so text and data cannot drift apart.
4. **Negative results are reported.** The pilot published two refuted hypotheses and an
   underperforming agent. That standard holds here — which is why the failure to reproduce
   its own headline effect size is the first result on this page.
5. **Generated artefacts are never hand-edited.** A wrong figure means wrong code or
   wrong data.

---

## Citing

This phase's results are provisional and not yet written up. **Cite the pilot study's
paper** — see [`CITATION.cff`](CITATION.cff). If you refer to a number from this
repository, cite the repository and the commit it was measured at.

```bibtex
@mastersthesis{odusina2026calibration,
  author = {Odusina, Oluwasegun Ibrahim},
  title  = {Development and Evaluation of a Calibration-Aware {REINFORCE} Agent
            for Energy-Efficient Aerial Agricultural Surveillance},
  school = {Miva Open University},
  year   = {2026},
  type   = {Seminar paper},
  note   = {Reproduction package: \url{https://github.com/OSegun/scoutplan}}
}
```

MIT licensed. See [`LICENSE`](LICENSE).

## References

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2021). Deep
reinforcement learning at the edge of the statistical precipice. *NeurIPS 34*.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural
networks. *ICML 34*.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy
optimization algorithms. *arXiv:1707.06347*.

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural
networks. *ICML 36*.

---

**Suggested GitHub topics:** `reinforcement-learning` `computer-vision`
`uncertainty-quantification` `calibration` `informative-path-planning` `pytorch`
`stable-baselines3` `precision-agriculture` `paper-implementations`
`reproducible-research`
