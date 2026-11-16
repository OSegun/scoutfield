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
| 6 | `python experiments/make_figures.py` | under a minute | Figures 9–15 (stage `06_figures`); Figure 16 once step 7 has run |
| 7 | `python experiments/07_uncertainty.py train --member <k>` for k = 1 to 4, then `python experiments/07_uncertainty.py evaluate`, then `python experiments/make_figures.py --only 16` | about 40 min per member (step 1's time), GPU; not yet run | `member<k>_best.pt`, `member<k>_last.pt`, `member<k>_summary.json`, `uncertainty_comparison.json` (stage `07_uncertainty`); Figure 16 (stage `06_figures`) |
| 8 | `python experiments/08_export_onnx.py` | under a minute, CPU, laptop | `classifier.onnx`, `export_record.json` in `.artifact_cache/deployment/<run_id>/`; stage `08_deployment` only with `--push` |
| 8 | `python experiments/08_measure_latency.py` | about a minute, CPU, laptop | `deployment_latency.json` beside the export; stage `08_deployment` only with `--push` |
| 8 | `python experiments/08_check_demo_logits.py` | under a minute, CPU, laptop | `demo_logit_check.json` beside the export; stage `08_deployment` only with `--push` |
| 8 | `python experiments/08_export_policy.py` | about a minute, CPU, laptop | `policy_ppo_seed0.onnx`, `policy_export_record.json` beside the export; stage `08_deployment` only with `--push` |

Runtimes are wall-clock times from run `r2026-11-13` on Kaggle: GPU T4 x2 for steps 0–3,
CPU for steps 4–6. Steps 0 and 1 run in one notebook.

Step 8 reads `best.pt` from stage `01_perception` of the store, exports it to ONNX, and
runs ONNX and PyTorch on the same inputs: the hard predictions must match and the logits
must agree within the tolerance in `configs/deployment.yaml`, or the script exits
non-zero. `--checkpoint <path>` exports a local copy instead, with no network. The
`.onnx` file is never written into the repository. `--push` files it in the store and is
refused for a checkpoint the run did not record, or from a working tree with uncommitted
changes. `experiments/08_measure_latency.py` then times the exported file on this
machine; its `--push` is refused unless that file is the `classifier.onnx` already filed
in the store. Once both are pushed, the promotion step copies `export_record.json` and
`deployment_latency.json` into `results/` and lists them in `results/RUN.json`, like
every other result. A latency belongs to the machine it was measured on: re-running step
8 elsewhere gives that machine's figure, not this one.

`experiments/08_check_demo_logits.py` runs the demonstration's sample photographs
through the deployed path and compares each logit with the one stage 02 recorded for the
same photograph; it exits non-zero if any differs by more than the export tolerance or
changes sign. The PlantVillage samples are included only on a machine that has them in
`.artifact_cache/deployment/<run_id>/samples_local/`. Its `--push` is refused unless the
classifier checked is the one filed in the store, the check passed, and the working tree
is committed.

`experiments/08_export_policy.py` exports the PPO policy trained with seed 0 for the
demonstration, which runs it with ONNX Runtime alone. It flies PPO's own episodes in the
real environment and, at every step, asks both the ONNX policy and the Stable-Baselines3
policy for an action; at least 1,000 observations are compared, and any differing action
makes it exit non-zero. Its `--push` follows the same rules as the classifier's.

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

The scripts that operate the store on the author's accounts (uploading, promoting,
verifying a stage) are kept outside this repository, and none is needed to check a run.
From the repository root, this compares every promoted file with the SHA-256 that
`results/RUN.json` recorded for it:

```python
import hashlib, json, pathlib

run = json.loads(pathlib.Path("results/RUN.json").read_text(encoding="utf-8"))
for f in run["files"]:
    digest = hashlib.sha256(pathlib.Path(f["path"]).read_bytes()).hexdigest()
    print("ok      " if digest == f["sha256"] else "MISMATCH", f["path"])
```

Every line should read `ok`. To check the run in the store as well, download it with
`snapshot_download` above and compare each entry of `results/RUN.json` with the
`sha256` that `runs/<run_id>/manifest.json` records for the same stage and file.

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
| ONNX latency (median, p95, range over repeats) and the machine it ran on | `deployment_latency.json` → `measurements`, `hardware` | step 8 |

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

**A stage run on a laptop never writes into `results/` or `figures/`.** Its working files
go to `local_work_dir` in `configs/artifacts.yaml` (`.artifact_cache/work/`, gitignored):
the sweep's done-registry, CSV and summary under `results/` there (file names set by
`runner` in `configs/sweep.yaml`), figures under `figures/` there. On Kaggle they go to
`/kaggle/working`. The repository's own `results/` and `figures/` are written by the
promotion step only, from the store.

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

---

## 7. Running the demonstration

`scoutfield/app/streamlit_app.py` is one app with four panels: Classify, Plan, Compare
and Analyse your leaves. The first three show what the experiment measured, under the
conditions it measured. The fourth runs a visitor's own photographs through the same
deployed classifier and offers a report (CSV and Markdown) to download; uploads stay in
memory and are never saved, and its limits are under `demo.upload`. Every
setting is in `configs/deployment.yaml` under `demo`, and every file it reads comes
from the artefact store, by one code path. Where that store is depends on one setting:
`demo.source`, overridden by the environment variable `SCOUTFIELD_DEMO_SOURCE`.
`store` means the public store, read-only, with no token. `offline` means a local
folder. The app imports no PyTorch.

### On a laptop, offline

Once, while online:

```bash
python experiments/08_select_samples.py      # optional: needs the two datasets in data/
python experiments/08_fill_demo_folder.py
```

The first command picks sample images by rule from the stage 02 records (see
`demo.samples`). It needs PlantDoc (Kaggle `nirmalsankalana/plantdoc-dataset`) and
PlantVillage (Kaggle `abdallahalidev/plantvillage-dataset`) unpacked under `data/`. The
second copies every file the app reads, and the laptop-only samples, into
`.demo_offline/` (gitignored). Each file is checked against the run's manifest when it
is copied and again whenever it is read.

Then, with the network switched off:

```bash
make app-offline
```

In PowerShell, which has no `make`:

```powershell
$env:SCOUTFIELD_DEMO_SOURCE = "offline"; python -m streamlit run scoutfield/app/streamlit_app.py
```

If the folder is empty, the page says so and names the fill command. The default
evaluation seed of the Plan panel is chosen by `experiments/08_choose_demo_seed.py`,
which prints the table it chose from.

### Hosted: Streamlit Community Cloud

Live at <https://scoutfield.streamlit.app/>. Community Cloud deploys
`scoutfield/app/streamlit_app.py` from the GitHub repository, branch `main`, on Python
3.12, and redeploys after every push to `main`. It installs
`scoutfield/app/requirements.txt`, which sits beside the app and so takes precedence
over the root `requirements.txt`. It reads the public store with no token or secret.
Only the PlantDoc sample images are offered there. Limits: about 1 GB of memory (the app
holds about 220 MB once loaded, and peaked at 524 MB with the largest upload batch the
limits allow, measured on a laptop; `docs/METHOD.md`, section "6. Deployment",
subsection "The demonstration"); the app sleeps after 12 hours without visitors and wakes on the next
visit, which takes up to a minute.

### Hosted: Google Cloud Run

The fallback, from the container defined by `Dockerfile` (with `.dockerignore`, which
admits only `scoutfield/`, `configs/` and `LICENSE`). Built and run locally first:

```bash
docker build -t scoutfield-demo .
docker run --rm -p 8080:8080 scoutfield-demo
```

The app is then at `http://localhost:8080`. On Cloud Run it listens on the port in the
`PORT` environment variable. A deployment from source:

```bash
gcloud run deploy scoutfield-demo --source . --region <region> --memory 1Gi --cpu 1 --allow-unauthenticated --min-instances 0 --session-affinity --timeout 3600
```

`--min-instances 0` lets the service scale to nothing when idle, so it costs nothing
then; the first visit afterwards waits while a container starts (a cold start).
`--session-affinity` keeps a viewer's Streamlit connection on one container, and
`--timeout 3600` keeps that connection open for up to an hour. Cloud Run's free tier
still requires a billing account.

### Considered and not used

An always-free virtual machine on Oracle Cloud was considered and not used: it is a
self-managed server, which is more upkeep than a demonstration needs.
