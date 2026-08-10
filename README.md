# scoutfield

**Does a classifier's miscalibration change where a drone should fly?** A PyTorch
implementation of calibration-aware informative path planning for energy-constrained
aerial crop scouting.

[![tests](https://github.com/OSegun/scoutfield/actions/workflows/ci.yml/badge.svg)](https://github.com/OSegun/scoutfield/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![pilot](https://img.shields.io/badge/pilot-scoutplan%20v1.0.0-2C5F2D)](https://github.com/OSegun/scoutplan)

> **Status: work in progress. No result in this repository has been measured yet.**
> The completed pilot study — 195 executed runs, 8 figures, the full write-up — lives
> in the companion repository [**scoutplan**](https://github.com/OSegun/scoutplan),
> pinned here at `v1.0.0`.

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

## What the pilot found, and what this repo tests

| Pilot result | Value |
| --- | --- |
| Accuracy across all temperatures | 0.8161, invariant to 4 d.p. |
| ECE swept | 0.0037 (`T`=1) → 0.1980 (`T`=4) |
| Lawnmower detections/joule collapse | 0.0781 → 0.0058 = **13.4×** |
| GreedyEntropy degradation | 1.84× |
| Ranking reversal at `T`=4 | GreedyEntropy beats Lawnmower 7.3× |

Two hypotheses were refuted and reported as such: the advantage did not grow with spatial
clustering, and performance was **not** monotone in ECE — the *direction* of
miscalibration governs, not its magnitude. The learned planner also lost to the lawnmower
baseline at every temperature.

Every one of those numbers came from an instrument that was, by design, not a classifier.
Removing that limitation is what this repository is for.

**Three claims to earn.**

1. *The effect is not an artefact of the surrogate.* A fine-tuned EfficientNet-B0,
   temperature-scaled the same way, should reproduce the qualitative result. If it does
   not, that is the finding.
2. *The learned planner can win when the regime allows it.* The pilot ran a 12×12 field
   with a budget permitting near-total coverage, leaving nothing for selectivity to buy.
   At 32×32 with ≤40% reachable coverage, adaptive planning has room to win — or to fail
   for a reason we can name.
3. *The effect size is not an artefact of one threshold.* The confirmation threshold
   τ = 0.75 interacts with the temperature manipulation by construction. Until τ is
   swept, 13.4× is a magnitude *at that threshold*, not a general one.

---

## Install

```bash
git clone https://github.com/OSegun/scoutfield.git
cd scoutfield
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install-dev
make test
```

`make test` runs the contract tests against the pinned pilot. If they fail on a fresh
clone the dependency pin is wrong — fix that before writing anything else.

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
sweep`, `make figures`.

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
                 │               temperature scaling,  │
                 │               MC-dropout, ensembles │
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
is the only thing that changed, so the comparison is clean.
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
└── results/  figures/  checkpoints/       generated; gitignored
```

`docs/METHOD.md` lists every hyperparameter and the citation behind it, so the
implementation can be checked against the method without reading the code.
`docs/REPRODUCING.md` maps each reported number to the exact command that produces it.

---

## Compute

Everything runs on free-tier Kaggle. No budget, no paid data, no purchased hardware.

Kaggle supplies the GPU and nothing else — the code lives here and is installed into the
notebook, never pasted into it. A notebook that contains logic is a bug.

```python
!pip install -q -r https://raw.githubusercontent.com/OSegun/scoutfield/main/requirements-kaggle.txt
```

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

Ordered by dependency. Each item names how you know it is done.

| # | Work | Done when |
| --- | --- | --- |
| 1 | Fine-tune EfficientNet-B0 on PlantVillage; hold out PlantDoc | accuracy and per-class ECE reported on both sets; checkpoint reproducible from a seed |
| 2 | `CNNClassifier` behind `observe()` | `make test` passes with the CNN swapped in; pilot baselines run unmodified |
| 3 | Temperature sweep on the real classifier | accuracy invariant to 4 d.p. across `T` — or a documented explanation of why not |
| 4 | `FieldScoutEnv`: 32×32, ≤40% coverage, global belief map | coverage ceiling verified empirically; obs dimension asserted in a test |
| 5 | PPO to convergence | curve plateaus over ≥3 seeds; beats or loses to lawnmower with a stated reason |
| 6 | τ sensitivity | effect size reported as a curve over τ, not a single number |
| 7 | MC-dropout + deep ensembles; reliability diagrams | three uncertainty methods compared on one axis |
| 8 | Oracle planner; regret | every planner's gap to the ground-truth ceiling quantified |
| 9 | ONNX export, measured latency, deployed interface | latency measured on stated hardware, not estimated |

**Standing risks.** Reward shaping can consume weeks — freeze the reward early and sweep
it only if a result demands it. Simulator realism will be challenged in review, so every
physical constant traces to a citation and every invented value gets swept.

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
   underperforming agent. That standard holds here.
5. **Generated artefacts are never hand-edited.** A wrong figure means wrong code or
   wrong data.

---

## Citing

Until this phase produces results, cite the pilot study's paper — see
[`CITATION.cff`](CITATION.cff).

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

**Suggested GitHub topics:** `reinforcement-learning` `computer-vision`
`uncertainty-quantification` `calibration` `informative-path-planning` `pytorch`
`stable-baselines3` `precision-agriculture` `paper-implementations`
`reproducible-research`
