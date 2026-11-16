# scoutfield

**Does a classifier's miscalibration change where a drone should fly?** The implementation
phase of a completed pilot study — replacing the pilot's synthetic confidence instrument
with a fine-tuned EfficientNet-B0 and its REINFORCE agent with PPO, to find out whether the
pilot's effect was real or an artefact of the surrogate.

| | |
| --- | --- |
| **Project** | Development of a Calibration-Aware UAV Path Planning Model for Efficient Crop Disease Detection Under Limited Battery Capacity |
| **Type** | Research engineering project, Miva Open University |
| **Author** | Oluwasegun Ibrahim Odusina |
| **Supervisor** | Dr. Chukwuemeka Odi Agwu |

[![tests](https://github.com/OSegun/scoutfield/actions/workflows/ci.yml/badge.svg)](https://github.com/OSegun/scoutfield/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![pilot](https://img.shields.io/badge/pilot-scoutplan%20v1.0.0-2C5F2D)](https://github.com/OSegun/scoutplan)

> **Status: research in progress, not a finished study.** The pipeline is built and has run
> end to end. **Results: run `r2026-11-13`, reported in [`docs/RESULTS.md`](docs/RESULTS.md).**
> The classifier's ONNX latency is measured on a laptop; uncertainty methods are not yet
> run. Nothing here has been peer-reviewed.
>
> **Live demonstration: <https://scoutfield.streamlit.app/>** (Streamlit Community Cloud).
> It sleeps after 12 hours without visitors; the first visit after that wakes it, which
> takes up to a minute.
>
> The **completed** work is the pilot study: 195 executed runs and 8 figures, frozen in the
> companion repository [**scoutplan**](https://github.com/OSegun/scoutplan)
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

| Roadmap item | State |
| --- | --- |
| 1. Fine-tune EfficientNet-B0; hold out PlantDoc | implemented |
| 2. `CNNClassifier` behind the pilot's `observe()` | implemented; contract tests pass with the CNN swapped in |
| 3. Temperature sweep on the real classifier | implemented |
| 4. `FieldScoutEnv`: 32×32, ≤40% coverage | implemented |
| 5. PPO to convergence | implemented |
| 6. τ sensitivity | implemented |
| 7. MC-dropout, deep ensembles, reliability diagrams | **partial**: reliability diagrams drawn (Figure 9); MC-dropout and deep ensembles not run |
| 8. Oracle planner; regret | implemented |
| 9. ONNX export, latency, deployed interface | **partial**: classifier and PPO policy exported and verified; latency measured on a laptop CPU; demonstration live on Streamlit Community Cloud at <https://scoutfield.streamlit.app/> and runs offline on a laptop; Google Cloud Run copy not yet deployed |

Results are in [`docs/RESULTS.md`](docs/RESULTS.md). Every number there is read from the
files in `results/`, which are copies of one run's outputs; if the two disagree, the
document is wrong.

---

## What this run measured

Run `r2026-11-13`: 735 evaluation runs and a 2610-run τ sweep, on PlantDoc field images
the classifier was never trained on, with `T` relative to the temperature fitted on field
images. Detections per joule degrades by the ratios below across three spans of `T`
(IQM, 95% stratified bootstrap CI). The pilot's values for the same spans are beside
them.

| Planner | Full, `T` 0.3 → 4 | Overconfident, `T` 0.3 → 1 | Underconfident, `T` 1 → 4 |
| --- | --- | --- | --- |
| Lawnmower | 18.52 [15.00, 23.09] (pilot 13.41) | 1.19 [0.98, 1.45] (pilot 1.23) | 15.54 [12.54, 19.39] (pilot 10.86) |
| GreedyEntropy | 1.50 [1.23, 1.83] (pilot 1.84) | 1.39 [1.15, 1.75] (pilot 1.59) | 1.08 [0.86, 1.31] (pilot 1.16) |
| PPO | 2.22 [2.02, 2.45] | 1.15 [1.05, 1.27] | 1.93 [1.74, 2.13] |
| Random | 1.70 [1.39, 2.05] | 1.07 [0.88, 1.30] | 1.59 [1.30, 1.92] |
| Oracle | 1.39 [1.19, 1.52] | 1.03 [0.87, 1.20] | 1.35 [1.18, 1.46] |

The pilot used a surrogate classifier on a 12 × 12 field, so the spans are matched and
the magnitudes are not expected to be. All at τ = 0.75; across the τ grid Lawnmower's
underconfident ratio runs from 1.13 at τ = 0.55 to no detections at all at `T` = 4 for
τ ≥ 0.85. Of the five predictions registered before the run, four held and one (P5) did
not; the rules and values are in `docs/RESULTS.md`, section *"5. Pre-registered
predictions"*.

**Checkpoints and every intermediate file** (classifier weights, logit pools, PPO
policies, resume state and the run manifest) are in the public artefact store, pinned to
the revision the results were taken from:
[`runs/r2026-11-13`](https://huggingface.co/datasets/OSegun/scoutfield-artifacts/tree/699392734c937bcb827d99dff3244523fbf6d43d/runs/r2026-11-13).
`results/RUN.json` lists every file promoted into this repository with its SHA-256;
`docs/REPRODUCING.md` shows how to download the run and check it.

---

## What the pilot measured

The pilot's results, from 195 executed runs, read from `results/summary.json` in
[scoutplan](https://github.com/OSegun/scoutplan) at `v1.0.0`. Detections per joule, IQM.

| Pilot finding | Span | Value |
| --- | --- | --- |
| Lawnmower detections/joule | `T` = 0.3 → `T` = 4 | 0.0781 → 0.0058 = 13.4× |
| GreedyEntropy degradation | `T` = 0.3 → `T` = 4 | 1.84× |
| Ranking reversal | at `T` = 4 | GreedyEntropy beats Lawnmower 7.3× |

The pilot measured that magnitude at a single confirmation threshold, τ = 0.75, and its
own threats-to-validity section required a τ sensitivity analysis before quoting it as a
general effect size. This repository runs that analysis.

### Known limitations

Stated here rather than left for a reviewer to find.

- **The headline table pools three cluster scales.** The pilot found planner performance
  varies strongly with σ. Aggregating over σ is a deliberate choice for the temperature
  comparison, not a claim that σ does not matter.
- **No single effect size exists without its threshold.** The degradation depends
  strongly on the confirmation threshold τ, and differently for each planner; every ratio
  is quoted at its τ.
- **The τ sweep evaluates one PPO policy.** For cost, PPO's τ runs use the seed 0 policy
  at the median cluster scale only, so its τ curve has 5 runs per point against 15 for
  each baseline.
- **Uncertainty methods are not run in this release.** MC-dropout and deep ensembles are
  not compared.
- **Deployment numbers are machine-specific.** The exported classifier's latency is
  measured on a laptop CPU, stated in `docs/RESULTS.md`, section *"8. Deployment"*; no
  drone or embedded computer was measured, and latency says nothing about the model's
  field behaviour.
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
sweep`, `make tau`, `make figures`.

The demonstration, three panels showing the coupling under the conditions the experiment
measured, is live at <https://scoutfield.streamlit.app/>. It reads the public artefact
store and offers only the PlantDoc sample images. To run it yourself (settings in
`configs/deployment.yaml`):

```bash
make app            # reads the public artefact store
make demo-fill      # once, online: copies what the app reads into .demo_offline/
make app-offline    # reads only that folder; runs with the network off
```

Reproducing any specific number: [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

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
is the only thing that changed, so the comparison is clean. It is also what licenses
comparing this phase's results with the pilot's.
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
├── tests/                      contract tests against the pinned pilot
├── results/  figures/          one run's final outputs, copied from the artefact store
└── checkpoints/                created locally, untracked
```

`docs/METHOD.md` lists every hyperparameter and the citation behind it, so the
implementation can be checked against the method without reading the code.
`docs/REPRODUCING.md` maps each reported number to the exact command that produces it.
`docs/RESULTS.md` holds the numbers themselves and is written *from* the summary
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

The notebooks are in `notebooks/`; the cell above is `notebooks/_bootstrap.py`, with its
settings in `configs/kaggle.yaml`.

Internet **on**, accelerator **GPU T4 x2** (None for the CPU-only sweep notebook), and the
`HF_TOKEN` secret attached to each notebook. Sessions are
killed at nine hours and much sooner when idle, so every long job pushes its progress to
the artefact store and resumes from there when the notebook is run again. See
`notebooks/README.md`.

```bash
pip install kaggle                                   # credentials at ~/.kaggle/kaggle.json
kaggle kernels push -p notebooks/01_perception_finetune   # local -> Kaggle, one notebook
```

---

## Roadmap

Ordered by dependency. Each item names how you know it is done. ✅ done, results in
`docs/RESULTS.md` · ◐ partial · ☐ not started.

| # | Work | Done when | State |
| --- | --- | --- | --- |
| 1 | Fine-tune EfficientNet-B0 on PlantVillage; hold out PlantDoc | accuracy and per-class ECE reported on both sets; checkpoint reproducible from a seed | ✅ |
| 2 | `CNNClassifier` behind `observe()` | `make test` passes with the CNN swapped in; pilot baselines run unmodified | ✅ |
| 3 | Temperature sweep on the real classifier | accuracy invariant to 4 d.p. across `T` — or a documented explanation of why not | ✅ |
| 4 | `FieldScoutEnv`: 32×32, ≤40% coverage, global belief map | coverage ceiling verified empirically; obs dimension asserted in a test | ✅ |
| 5 | PPO to convergence | curve plateaus over ≥3 seeds; beats or loses to lawnmower with a stated reason | ✅ |
| 6 | τ sensitivity | effect size reported as a curve over τ, not a single number | ✅ |
| 7 | MC-dropout + deep ensembles; reliability diagrams | three uncertainty methods compared on one axis | ◐ |
| 8 | Oracle planner; regret | every planner's gap to the ground-truth ceiling quantified | ✅ |
| 9 | ONNX export, measured latency, deployed interface | latency measured on stated hardware, not estimated | ◐ demonstration live at <https://scoutfield.streamlit.app/>; Cloud Run copy not yet deployed |

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
3. **No number is hard-coded into a document.** Results are read from summary
   files, so text and data cannot drift apart.
4. **Negative results are reported.** The pilot published two refuted hypotheses and an
   underperforming agent. That standard holds here.
5. **Results and figures are written only by the code that produces them, never by
   hand.** A wrong figure means wrong code or wrong data.

---

## Citing

Cite this repository, the release a number appears in, and the run it comes from — see
[`CITATION.cff`](CITATION.cff).

```bibtex
@software{odusina2026scoutfield,
  author  = {Odusina, Oluwasegun Ibrahim},
  title   = {Development of a Calibration-Aware {UAV} Path Planning Model for Efficient
             Crop Disease Detection Under Limited Battery Capacity},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/OSegun/scoutfield},
  note    = {Research engineering project, Miva Open University.
             Supervisor: Dr. Chukwuemeka Odi Agwu. Results: run r2026-11-13
             in the Hugging Face artefact store}
}
```

The pilot it extends:

```bibtex
@software{odusina2026scoutplan,
  author  = {Odusina, Oluwasegun Ibrahim},
  title   = {{ScoutPlan}: calibration-aware active sensing for energy-constrained aerial
             crop scouting},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/OSegun/scoutplan}
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
`stable-baselines3` `precision-agriculture` `uav` `reproducible-research`
