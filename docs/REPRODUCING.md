# Reproducing

Every reported number maps to one command. If a number in a document has no row here, it
should not be in the document.

---

## 0. Prerequisites

```bash
git clone https://github.com/OSegun/scoutfield.git
cd scoutfield
python -m venv .venv && source .venv/bin/activate
make install-dev
make test          # must pass on a fresh clone
```

`make test` runs the contract tests against the pinned pilot (`scoutplan@v1.0.0`). If they
fail here, the dependency pin is wrong — fix that before anything else.

Datasets are not vendored. PlantVillage and PlantDoc are attached to the Kaggle notebooks
(mounted at `/kaggle/input/<slug>`) or downloaded to `./data/<slug>` locally. Slugs are in
`configs/perception.yaml` under `data.kaggle_slugs`.

---

## 1. Determinism

A run is fully described by `(agent, seed, T, tau, sigma)` plus the config file and the
git commit. `scoutfield.utils.seeding.run_id` builds that identifier, and it is what the
sweep's done-registry keys on.

`seed_everything(seed)` seeds Python, NumPy and torch, and puts cuDNN in deterministic
mode. Deterministic cuDNN costs throughput, which matters on a limited quota — turn it off
only for exploratory runs whose numbers will never be reported, and say so if you do.

If a parameter is added that changes results, it must be added to `run_id` too. Otherwise
a resumed sweep treats two different runs as the same completed job and skips one.

---

## 2. The pipeline

| Step | Command | Runtime | Produces |
| --- | --- | --- | --- |
| 1 | `python experiments/01_finetune_perception.py --config configs/perception.yaml` | ~40 min, GPU | `checkpoints/perception/best.pt`, `results/perception_summary.json` |
| 2 | `python experiments/02_calibrate.py --config configs/perception.yaml` | ~10 min, GPU | `results/calibration_sweep.json` |
| 3 | `python experiments/03_train_ppo.py --config configs/ppo_field32.yaml --seed 0` | ~2 h/seed, GPU | `checkpoints/ppo/seed0/` |
| 4 | `while JOB_BUDGET=60 python experiments/04_sweep.py; do :; done` | ~1 h, CPU | `results/sweep_results.csv`, `results/summary.json` |
| 5 | `python experiments/05_tau_sensitivity.py --config configs/sweep.yaml` | ~3 h, CPU | `results/tau_sensitivity.csv` |
| 6 | `python experiments/make_figures.py` | seconds | `figures/Figure9..16_*.png` |

Runtimes are estimates on a Kaggle T4 and will be replaced with measurements once the
steps exist.

Repeat step 3 for seeds 0–4. A single seed is not evidence of convergence.

**Ordering rule.** Never regenerate a figure or a document before re-running the sweep
that feeds it. `make_figures.py` reads `results/summary.json`; if that file is stale the
figures and the data disagree and nobody notices.

---

## 3. Number → command

| Number | Source file | Command |
| --- | --- | --- |
| Test accuracy (BASE_ACC for this phase) | `results/perception_summary.json` → `accuracy` | step 1 |
| Accuracy under shift | `results/perception_summary.json` → `accuracy_shift` | step 1 |
| Fitted temperature | `results/calibration_sweep.json` → `fitted_temperature` | step 2 |
| ECE per temperature | `results/calibration_sweep.json` | step 2 |
| Signed calibration error | `results/calibration_sweep.json` | step 2 |
| Coverage ceiling | printed by step 3; `results/summary.json` → `coverage` | step 3 |
| Detections per joule, per (agent, T) | `results/summary.json` | step 4 |
| Recall / precision / false alarms | `results/summary.json` | step 4 |
| IQM + 95% bootstrap CI | `results/summary.json` | step 4 |
| Effect size vs τ | `results/tau_sensitivity.csv` | step 5 |
| Regret against oracle | `results/summary.json` → `regret` | step 4 |
| ONNX latency (median, p95) | `results/latency.json` | `python -m scoutfield.app.export_onnx --checkpoint ...` |

---

## 4. Restarting cleanly

The sweep is resumable, which means it will happily resume into a mess.

```bash
rm results/_done.json results/sweep_results.csv     # BOTH, always
python experiments/04_sweep.py --reset
```

Deleting only one leaves the driver resuming and appending fresh rows to stale ones, with
no warning that the analysis now mixes two code versions. This has already cost time once.

PPO resumes the same way, from `checkpoints/ppo/ppo_seed<N>_resume.zip`. Delete that file
to force a fresh run — otherwise `train_ppo` continues the previous one, which is correct
after a killed Kaggle session and wrong after a code change.

### The `__main__` guard is not optional

`SubprocVecEnv` uses the *spawn* start method on Windows and macOS: each worker
re-imports the launching module. A script that calls `train_ppo` at module scope therefore
re-executes itself in every subprocess and spawns recursively, and the symptom is not a
crash — it is a run that appears to hang while consuming every core.

```python
if __name__ == "__main__":   # required in anything that trains PPO
    main()
```

Every driver in `experiments/` already has this. Add it to any ad-hoc script too. For
reference, training runs at roughly 390 fps with `n_envs=2` on a CPU laptop, and the
environment alone steps at ~3,500 steps/s — so a run that is not producing output within a
minute is hung, not slow.

---

## 5. Verifying an implementation, not just running it

Four checks catch the failures that produce plausible-looking wrong numbers.

**The instrument.** Accuracy must be invariant across the temperature sweep to four
decimal places. Asserted in `experiments/02_calibrate.py`; a failure means temperature is
applied after a softmax or the prediction is recomputed from scaled values.

**ECE shape.** ECE is minimised at `T = 1` and rises in both directions. If it becomes
monotone in `T`, the parameterisation is broken and the pilot's H2 result is no longer
testable. Covered by `tests/test_pilot_contract.py`.

**The interface.** `CNNClassifier.observe` matches `CalibratedClassifier.observe` exactly.
Covered by `tests/test_pilot_contract.py`.

**Environment parity.** `FieldScoutEnv` must not override `step`, `info`, `_bayes_update`
or `_observe_current` — those carry the energy model and the reward, and overriding them
would make these numbers incomparable to the pilot's. Covered by
`tests/test_env_parity.py`.

---

## 6. On Kaggle

Notebooks are version-controlled here and synced with the Kaggle API, so the repository
stays the single source of truth:

```bash
pip install kaggle          # credentials at ~/.kaggle/kaggle.json
make notebooks-push
make notebooks-pull
```

Each notebook installs this repo rather than containing logic:

```python
!pip install -q -r https://raw.githubusercontent.com/OSegun/scoutfield/main/requirements-kaggle.txt
```

Internet **on**, accelerator **GPU**, persistence **variables and files**. Only
`/kaggle/working` survives to the notebook output; anything written elsewhere is lost when
the session ends. Sessions are killed at nine hours and much sooner when idle, so every
long job checkpoints per epoch or per sweep job.
