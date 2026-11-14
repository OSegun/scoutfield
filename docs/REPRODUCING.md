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

| Step | Command | Runtime (run `r2026-11-13`) | Produces, in the store |
| --- | --- | --- | --- |
| 0 | `python experiments/00_check_leakage.py --config configs/perception.yaml` | ~27 min including notebook setup, GPU session | `leakage_report.json` (stage `01_perception`) |
| 1 | `python experiments/01_finetune_perception.py --config configs/perception.yaml` | ~40 min, GPU | `best.pt`, `last.pt`, `perception_summary.json` (stage `01_perception`) |
| 2 | `python experiments/02_calibrate.py --config configs/perception.yaml` | ~8 min, GPU | `calibration_sweep.json` and three logit pools (stage `02_calibration`) |
| 3 | `python experiments/03_train_ppo.py --config configs/ppo_field32.yaml --seed <N>` | ~25 min per seed, GPU | `ppo_seed<N>.zip`, `ppo_seed<N>_resume.zip`, `ppo_curve_seed<N>.json` (stage `03_ppo`) |
| 4 | `while JOB_BUDGET=60 python experiments/04_sweep.py --config configs/sweep.yaml; do :; done` | ~81 min for 735 jobs, CPU | `sweep_results.csv`, `summary.json`, `sweep_metadata.json` (stage `04_sweep`) |
| 5 | `python experiments/05_tau_sensitivity.py --config configs/sweep.yaml` | ~2.5 h for 2610 jobs, CPU | `tau_sensitivity.csv`, `tau_sensitivity_summary.json` (stage `05_tau`) |
| 6 | `python experiments/make_figures.py` | under a minute | Figures 9–15 (stage `06_figures`); Figure 16 needs the uncertainty comparison and is not drawn |
| 8 | `python experiments/08_export_onnx.py` | under a minute, CPU, laptop | `classifier.onnx`, `export_record.json` in `.artifact_cache/deployment/<run_id>/`; stage `08_deployment` only with `--push` |

Runtimes are wall-clock times from run `r2026-11-13` on Kaggle: GPU T4 x2 for steps 0–3,
CPU for steps 4–6. Steps 0 and 1 run in one notebook.

Step 8 reads `best.pt` from stage `01_perception` of the store, exports it to ONNX, and
runs ONNX and PyTorch on the same inputs: the hard predictions must match and the logits
must agree within the tolerance in `configs/deployment.yaml`, or the script exits
non-zero. `--checkpoint <path>` exports a local copy instead, with no network. The
`.onnx` file is never written into the repository. `--push` files it in the store and is
refused for a checkpoint the run did not record, or from a working tree with uncommitted
changes.

Run step 3 for every seed in `ppo_train_seeds` in `configs/sweep.yaml` (0, 1 and 2). A
single seed is not evidence of convergence.

**Ordering rule.** Never regenerate a figure or a document before re-running the sweep
that feeds it. `make_figures.py` reads `results/summary.json`; if that file is stale the
figures and the data disagree and nobody notices.

### 2.1 Carrying artefacts between stages

On Kaggle the steps are four separate kernels with no shared filesystem:
`/kaggle/working` is the only writable location and it belongs to a single session. So no
stage reads another's output from disk. Every stage reads its inputs from, and writes its
outputs to, the artefact store named in `configs/artifacts.yaml`, a public Hugging Face
dataset repository:

```
runs/<run_id>/manifest.json
runs/<run_id>/<stage>/<file>
```

A missing input raises and names the store path it looked for; an input whose SHA-256
differs from the manifest raises too. Writing needs a Hugging Face write token: on Kaggle,
attach the `HF_TOKEN` secret to each notebook (Add-ons → Secrets); on a laptop, run
`hf auth login` (`huggingface-cli login` in `huggingface_hub` versions before 1.0).
Reading needs nothing. Setting
`SCOUTFIELD_ARTIFACT_STORE=<dir>` swaps the hub for a local directory, which runs the
pipeline fully offline.

Long stages push their resume state as they go (fine-tuning every epoch, PPO every
`checkpoint_freq` steps, the sweeps every `push_every_jobs` jobs), so a session that ends
early is resumed by running the same notebook again. Each push is two commits to the
store (the files, then the manifest), and Hugging Face accepts at most 128 commits an
hour per repository, so these intervals are chosen to stay well below that rate; a push
refused by the limit stops the stage, which is then resumed the same way once the hour
has passed. Notebook 01 alone needs
`RESUME = True` set in its run cell, because resuming fine-tuning is refused unless asked
for.

The store is public, so a run can be downloaded without an account. Pin the store
revision recorded in `results/RUN.json` (`store_revision`) so the files are exactly the
ones the tracked results came from:

```python
from huggingface_hub import snapshot_download

snapshot_download("OSegun/scoutfield-artifacts", repo_type="dataset",
                  revision="<store_revision from results/RUN.json>",
                  allow_patterns="runs/<run_id>/*", local_dir="artifacts")
```

`runs/<run_id>/manifest.json` lists every file with its SHA-256. Each file under
`results/` and `figures/` is listed in `results/RUN.json` with the SHA-256 the manifest
recorded for it, so the tracked results can be checked against the run byte for byte.

---

## 3. Number → command

Every file named here is in `results/` for the promoted run, and in the store under the
stage that produced it.

| Number | Source file | Command |
| --- | --- | --- |
| In-distribution accuracy (control) | `perception_summary.json` → `accuracy` | step 1 |
| Accuracy under shift (BASE_ACC: `shift_eval`) | `perception_summary.json` → `accuracy_shift`, `accuracy_shift_eval` | step 1 |
| Leakage counts | `leakage_report.json` | step 0 |
| Lab- and shift-fitted temperatures | `calibration_sweep.json` → `lab_fitted_temperature`, `shift_fitted_temperature` | step 2 |
| ECE, signed error, NLL, Brier per split and temperature | `calibration_sweep.json` → `sweeps` | step 2 |
| PPO learning curves and convergence | `ppo_curve_seed<N>.json` → `episodes`, `converged` | step 3 |
| Coverage ceiling | printed by step 3; measured coverage in `summary.json` → `cells` | step 3 |
| Morisita index and prevalence | `summary.json` → `environment` | step 4 |
| Detections per joule, recall, precision, F1, false alarms, per (agent, T), with IQM and 95% CI | `summary.json` → `cells` | step 4 |
| Degradation over the three spans | `summary.json` → `degradation` | step 4 |
| Lab-fitted condition | `summary.json` → `conditions` | step 4 |
| Regret against the oracle | computed from `summary.json` → `cells` as 1 − planner / oracle; Figure 15 | steps 4 and 6 |
| Effect size against τ | `tau_sensitivity_summary.json`, `tau_sensitivity.csv` | step 5 |
| Pre-registered predictions, held or not | judged from `summary.json` by the rules in `docs/METHOD.md`, section *"Pre-registered predictions"* | step 4 |
| ONNX export check (max logit difference, predictions identical) | `export_record.json` → `verification` | step 8 |
| ONNX latency (median, p95) | not measured in this release | step 8 |

---

## 4. Restarting cleanly

Every stage is resumable through the store, which means it will happily resume into a
mess. Resume state is kept per run: the sweeps' done-registries and partial CSVs, and
PPO's `ppo_seed<N>_resume.zip`, all live under `runs/<run_id>/` in the store, and a stage
re-run under the same `run_id` continues from them. That is correct after a killed Kaggle
session and wrong after a code change, because the results would then mix two code
versions without warning.

**To start fresh, set a new `run_id` in `configs/artifacts.yaml`, commit and push it.**
Never reuse a `run_id`, and never delete files from a run in the store to reset it: the
manifest of the old run is the record of what it produced.

Do not delete files from `results/` or `figures/` either. They are the promoted outputs of
one run, listed in `results/RUN.json`, not resume state.

**Running steps 4–6 on a laptop writes into this clone's `results/` and `figures/`**, over
the promoted files (the sweep's working files are set by `runner` in
`configs/sweep.yaml`). On Kaggle this is harmless: each session writes into its own clone.
Locally, run them in a separate clone, or restore the promoted files afterwards with
`git checkout -- results figures`.

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

Notebooks are version-controlled here and uploaded to Kaggle with the Kaggle API
(`kaggle kernels push -p notebooks/<notebook>`), so the repository stays the single
source of truth.

Each notebook clones and installs this repo rather than containing logic. It clones
because `experiments/` and `configs/` are not packaged, so a pip install alone leaves the
drivers and the configs absent; it installs *from the clone* so that nothing depends on
`raw.githubusercontent.com` and no second copy of the package can shadow the cloned one.
The cell is generated from `configs/kaggle.yaml` — a username or a fork is changed there
and the notebooks regenerated.

Every run prints the commit it cloned. Quote that commit beside any number the run
produced: the notebooks track `main`, and `main` moves.

Internet **on**; accelerator **GPU T4 x2** for notebooks 01–03 (the first cell stops on a
GPU this PyTorch cannot use) and **None** for notebook 04; the `HF_TOKEN` secret attached
to each notebook. Uploading a notebook again with `kaggle kernels push` can detach its
secret; the first cell then stops within seconds and says so. Only `/kaggle/working`
survives a session, so every long job pushes its progress to the artefact store as it
goes and resumes from there, as described above.
