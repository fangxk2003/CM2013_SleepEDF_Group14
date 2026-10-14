# Development

Run commands from `CM2013_SleepEDF_Group14/`. For your first experiment, follow [Getting started](GETTING_STARTED.md); for code interfaces, read [Architecture](ARCHITECTURE.md).

## Shared environment

The team uses Python **3.11**, matching the course CI, and `uv` with the committed lockfile.

| File/folder | Purpose |
| --- | --- |
| `.python-version` | Shared Python minor version, 3.11 |
| `pyproject.toml` | Project metadata and direct dependency requirements |
| `uv.lock` | Exact resolved versions, including transitive dependencies |
| `.venv/` | Each teammate's installed local environment; ignored by Git |
| `requirements-lock.txt`, `requirements-real.txt` | Course scientific and real-data dependency pins |

Use this after cloning or pulling dependency changes:

```bash
uv sync --locked
uv run --locked python --version
uv run --locked python -c "import numpy, scipy, sklearn, mne; print(numpy.__version__, scipy.__version__, sklearn.__version__, mne.__version__)"
```

The checks should show Python 3.11.x and `2.2.6 1.15.3 1.7.2 1.10.1`.

The remaining direct pins are Matplotlib 3.10.9, PyWavelets 1.8.0, Pillow 12.2.0, imbalanced-learn 0.14.2 and WFDB 4.3.0. Commit `.python-version`, `pyproject.toml` and `uv.lock` together when they change, along with the course requirements files if those pins change.

`uv sync --locked` installs from the lockfile and errors if it needs updating. Plain `uv sync` or `uv run` can update an absent or outdated lockfile before installing; using `--locked` keeps an accidental dependency change visible. After an intentional manifest change:

```bash
uv lock
uv sync --locked
```

To add a runtime dependency, use `uv add <package>`; for an editor/notebook tool, use `uv add --dev <package>`. To change a pin intentionally, use `uv add "mne==<new-version>"`. These update both manifest and lockfile. Review version changes and run the checks below before sharing them. Individual `pip install` changes bypass the shared lock.

`uv run --locked python ...` uses `.venv` without activation. You can also activate once per terminal session:

```bash
source .venv/bin/activate
python -c "import sys; print(sys.executable); print(sys.version)"
python scripts/smoke_test.py
```

The executable should be inside `.venv/bin/`.

## Verification and notebooks

```bash
uv run --locked python scripts/smoke_test.py
uv run --locked python -m unittest discover -s tests -v
```

The smoke check exercises synthetic data offline. Regression tests compare supplied and project behavior and verify experiment selection, model factories and saved-result plotting.

Use this repository's `.venv` as the notebook kernel, restart it after syncing dependencies, clear old outputs and rerun from the top. The supplied notebook is [src/sleepedf/reference/track_sleep_edf.ipynb](../src/sleepedf/reference/track_sleep_edf.ipynb). Team experiments import `SleepEDFTrack` from `sleepedf`; supplied reference recipes import it from `sleepedf.reference.sleep_edf`. Shared types live in `sleepedf.reference.adapter`; reporting helpers are available as `from sleepedf.reference import report as R`.

## Experiment commands

Use `scripts/run_experiment.py` as the main command, including for the baseline:

```bash
# EEG bandpass and RBF-SVM on the first three cached subjects, both nights
uv run --locked python scripts/run_experiment.py --preprocess bandpass --model rbf_svm --n-subjects 3 --nights 1 2 --seed 0

# Wavelet and logistic regression on explicitly selected subjects, first nights
uv run --locked python scripts/run_experiment.py --preprocess wavelet --model logistic_regression --subjects 0 1 2 --nights 1 --seed 0

# Show all runner options
uv run --locked python scripts/run_experiment.py --help
```

Preprocessing choices are `none` (P0), `bandpass` (P1), `wavelet` (P2), `broken_segments` (P3), and `denoise_clipping` (P5). P3 excludes detected near-flat epochs; P5 adds suspected-clipping flags and leaves signals unchanged. P4 signal normalisation remains a planned interface and is not a runner choice.

Models are `random_forest`, `logistic_regression`, `rbf_svm` and optional `xgboost`; the default is `random_forest`, with seed 0. The runner uses existing cached EDFs, no cropping, the supplied 11 features and subject-wise LOSO. It performs no download or hyperparameter tuning. Python experiments can use `sleepedf.machine_learning.make_model(name, **kwargs)` and pass a fresh estimator through `clf=` to `evaluate_loso`; see [Architecture](ARCHITECTURE.md) for parameter and tuning guidance.

Use `--subjects` or `--n-subjects`, together with `--nights`, to keep the cohort explicit. Selection happens before EDF loading and hashing. Each selected PSG must have one matching hypnogram. `--cache-dir` changes the cache location and `--output-dir <new-directory>` selects a fresh output folder; existing output directories are never overwritten.

Shared loading and evaluation code lives in `scripts/internal/run_baseline.py`; the import helper lives in `scripts/internal/_bootstrap.py`. The public experiment command uses both. For reproducing a historical baseline's output naming, the shared runner can also be invoked directly:

```bash
uv run --locked python scripts/internal/run_baseline.py --subjects 0 1 2 --nights 1 2
```

Use `scripts/run_experiment.py` for new experiments.

XGBoost is optional:

```bash
uv sync --locked --extra xgboost
uv run --locked --extra xgboost python scripts/run_experiment.py --model xgboost --subjects 0 1 2 --nights 1 2 --seed 0
```

For a separately managed pip environment, the extra is installed with `pip install -e '.[xgboost]'`.

## Inspect signals and saved runs

```bash
# Offline synthetic signals
uv run --locked python scripts/visualisation.py

# A real recording and epoch
uv run --locked python scripts/visualisation.py --cache-dir sleep_edf_data --record 0 --epoch 10

# Two saved runs; replace the folder names with your chosen runs
uv run --locked python scripts/visualisation_results.py results/<first-run> results/<second-run>
```

Saved-result plots require no EDF loading or model training. The comparison index is normally `results/visualisations/index.html`, with per-run dashboards and `summary.csv`. It shows confusion counts/percentages, stage precision/recall/F1, class distributions and subject variation. `--report` and `--provenance` accept an explicit file pair; `--results-dir` chooses another results root and `--output-dir` chooses a plot destination. PNG is the default; use `--format pdf`, `--format svg` or `--show` as needed.

Plot outputs are regenerated on each invocation. Use a separate output directory to retain a comparison. Without a run selection, the script searches recursively through all saved results. Cohort IDs help expose different input files, labels, groups, folds and epoch counts, but still compare configuration, source and package versions. Error bars are sample standard deviations across finite subject scores, not confidence intervals; undefined scores or spread from fewer than two subjects are N/A. Saved predictions do not preserve enough timing information to reconstruct whole-night timelines safely.

Detailed P3/P5 investigations belong in `scripts/diagnostics/`; use its [script guide](../scripts/diagnostics/README.md) after learning the main workflow.

## Preserve experiment evidence

Runs save `report.json` and `provenance.json`, not a fitted final model or submission CSV. Keep those original files together; they document inputs, settings, source hashes, environment and Git state. Log decisions and metrics with subject spread in [RESULTS.md](../RESULTS.md).

Older records include Python 3.13.3 and earlier package versions. Keep them as historical evidence. After an environment change, rerun the smoke check and baseline and record the new seed, source revision and environment before comparing new experiments. The [smoke-test comparison](SMOKE_TEST_COMPARISON.md) explains the historical environment investigation; the [results guide](../results/README.md) explains saved runs and provenance.
