# Notebooks

Kaggle supplies the GPU. It supplies nothing else — the code lives in this
repository and is installed into the notebook, never pasted into it.

**A notebook that contains logic is a bug.** Logic belongs in `scoutfield/`
where it can be tested, reviewed and reused. A notebook orchestrates: install,
configure, call, display. If you find yourself defining a training loop in a cell,
that loop belongs in a module.

## Layout

Each notebook is a directory containing the `.ipynb` and a `kernel-metadata.json`
that the Kaggle API uses to push and pull it.

**Both files in every directory are generated.** `kernel-metadata.json` is rewritten on
every run of `scripts/make_notebooks.py`, from `configs/kaggle.yaml` (username, fork,
branch) and `configs/perception.yaml` (dataset slugs, shared with the training code).
Kaggle requires one metadata file per kernel — each carries its own id, code file,
accelerator and attached datasets — but they are maintained in one place, not four. Edit
a config and re-run the script; hand-edits here are meant to be lost.

```
notebooks/
├── _bootstrap.py                first cell of every notebook, kept in one place
├── 01_perception_finetune/      EfficientNet-B0 on PlantVillage        [GPU]
├── 02_calibration_shift/        temperature fit, sweep, PlantDoc shift [GPU]
├── 03_ppo_training/             PPO on the 32x32 field                 [GPU]
└── 04_evaluation_sweep/         full sweep + figures                   [CPU]
```

## Sync

```bash
pip install kaggle                  # credentials at ~/.kaggle/kaggle.json
make notebooks-push                 # local -> Kaggle
make notebooks-pull                 # Kaggle -> local, with executed outputs
```

Notebooks are version-controlled, so the repository stays the single source of
truth and a notebook edited in the Kaggle UI does not quietly become the real one.

Strip outputs before committing unless the output *is* the artefact — committed
plot images inflate diffs and make review painful.

## Required notebook settings

Set these in the right-hand panel of the Kaggle editor. They are also encoded in
each `kernel-metadata.json`, but the UI wins if the two disagree.

| Setting | Value | Why it matters |
| --- | --- | --- |
| Internet | **On** | `pip install git+...` fails without it, often with a confusing SSL error rather than a clear network one |
| Accelerator | **GPU T4 ×2** / **P100** | fine-tuning is impractical on CPU |
| Persistence | **Variables and files** | sessions die; see below |
| Datasets | PlantVillage, PlantDoc | mounted read-only at `/kaggle/input/<slug>` |

## Sessions get killed — design for it

Kaggle enforces a nine-hour ceiling and disconnects idle sessions far sooner. Only
`/kaggle/working` survives to the notebook's output; anything written elsewhere is
gone when the session ends.

Every long job checkpoints per epoch or per sweep job via
`scoutfield.utils.checkpoint` and resumes from disk. Run the sweep in bounded
chunks so a kill costs one chunk:

```python
!while JOB_BUDGET=60 python experiments/04_sweep.py; do :; done
```

Note that persistence covers the *interactive* session. A committed run
(**Save Version → Save & Run All**) starts in a clean container and keeps only what it
writes to `/kaggle/working` during that run, so resume state has to be fed back in as an
attached input or a Dataset. Verify that on your first pair of commits rather than
assuming it.

## GPU quota

The free tier gives a limited number of GPU hours per week. Two habits protect it:

- Develop on a CPU session with a tiny subset, switch to GPU only for the real run.
- Never reinstall torch, torchvision, numpy, pandas, sklearn or pillow — Kaggle
  ships them, reinstalling costs minutes and risks a CUDA mismatch. That is why
  `requirements-kaggle.txt` is separate from `requirements.txt`.
