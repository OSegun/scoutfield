.PHONY: install install-dev test lint fmt clean \
        train-perception calibrate train-ppo sweep tau figures export-onnx app

PY ?= python

# ------------------------------------------------------------------ setup
install:            ## install the library and the pinned pilot
	$(PY) -m pip install -e ".[perception,planning,figures]"

install-dev: install
	$(PY) -m pip install -e ".[dev,app]"

# ------------------------------------------------------------------ checks
test:               ## contract tests — must pass before any commit
	$(PY) -m pytest

lint:
	$(PY) -m ruff check scoutfield experiments tests

fmt:
	$(PY) -m ruff format scoutfield experiments tests

# ----------------------------------------------- experiments (see docs/REPRODUCING.md)
train-perception:   ## 1. fine-tune EfficientNet-B0 on PlantVillage
	$(PY) experiments/01_finetune_perception.py --config configs/perception.yaml

calibrate:          ## 2. fit T on val; sweep T; evaluate the PlantDoc shift
	$(PY) experiments/02_calibrate.py --config configs/perception.yaml

train-ppo:          ## 3. PPO on the 32x32 field
	$(PY) experiments/03_train_ppo.py --config configs/ppo_field32.yaml

sweep:              ## 4. full evaluation sweep, resumable in bounded chunks
	while JOB_BUDGET=60 $(PY) experiments/04_sweep.py; do :; done

tau:                ## 5. tau sensitivity — unblocks quoting the effect size
	$(PY) experiments/05_tau_sensitivity.py --config configs/sweep.yaml

figures:            ## 6. rebuild every figure from results/summary.json
	$(PY) experiments/make_figures.py

export-onnx:        ## 8. export the trained classifier to ONNX and verify it (no push)
	$(PY) experiments/08_export_onnx.py --config configs/deployment.yaml

# -------------------------------------------------------------------- app
app:
	$(PY) -m streamlit run scoutfield/app/streamlit_app.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
