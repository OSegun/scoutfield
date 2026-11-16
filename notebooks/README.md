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

**Both files in every directory are generated**, from `configs/kaggle.yaml` (username,
fork, branch) and `configs/perception.yaml` (dataset slugs, shared with the training
code). Kaggle requires one metadata file per kernel — each carries its own id, code file,
accelerator and attached datasets — but they are maintained in one place, not four. Edit
a config and regenerate; hand-edits here are meant to be lost.

```
notebooks/
├── _bootstrap.py                first cell of every notebook, kept in one place
├── 01_perception_finetune/      EfficientNet-B0 on PlantVillage        [GPU]
├── 02_calibration_shift/        temperature fit, sweep, PlantDoc shift [GPU]
├── 03_ppo_training/             PPO on the 32x32 field                 [GPU]
├── 04_evaluation_sweep/         full sweep + figures                   [CPU]
└── 07_uncertainty_methods/      ensemble members, MC-dropout, Fig. 16  [GPU]
```

## Sync

```bash
pip install kaggle                                   # credentials at ~/.kaggle/kaggle.json
kaggle kernels push -p notebooks/01_perception_finetune   # local -> Kaggle, one notebook
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
| Accelerator | **GPU T4 ×2** for 01–03, **None** for 04 | the shipped PyTorch cannot run on the P100; the first cell stops if it sees one |
| Secrets | **HF_TOKEN** attached (Add-ons → Secrets) | every stage writes to the artefact store; the first cell stops without it |
| Datasets | PlantVillage and PlantDoc, on every notebook | mounted read-only at `/kaggle/input/<slug>` |

## Sessions get killed — design for it

Kaggle enforces a nine-hour ceiling and disconnects idle sessions far sooner. Only
`/kaggle/working` survives to the notebook's output; anything written elsewhere is
gone when the session ends.

A committed run (**Save Version → Save & Run All**) starts in a clean container, so
resume state is not kept on Kaggle at all. It lives in the artefact store
(`configs/artifacts.yaml`): fine-tuning pushes `last.pt` every epoch, PPO pushes its
resume state every `checkpoint_freq` steps, and the sweeps push their CSV and
done-registry every `push_every_jobs` jobs. To resume after a session ends, run the same
notebook again. Notebook 01 also needs `RESUME = True` in its run cell, because
fine-tuning refuses to resume unless asked to.

No notebook attaches another's output (`kernel_sources` is empty by design): every input
comes from the store.

## GPU quota

The free tier gives a limited number of GPU hours per week. Two habits protect it:

- Develop on a CPU session with a tiny subset, switch to GPU only for the real run.
- Never reinstall torch, torchvision, numpy, pandas, sklearn or pillow — Kaggle
  ships them, reinstalling costs minutes and risks a CUDA mismatch. That is why
  `requirements-kaggle.txt` is separate from `requirements.txt`.
