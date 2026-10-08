# CM2013 Sleep-EDF — Group 14

A three-stage pipeline: **preprocessing → feature extraction → machine learning**.
The runnable default remains P0 (no preprocessing), the supplied 11 features,
and a StandardScaler + 200-tree random forest (balanced class weights, seed 0).
Evaluation uses leave-one-subject-out (LOSO) validation, with scaling and feature
selection fitted separately inside each training fold.

## Repository layout

```text
src/
  sleepedf/
    preprocessing/        # P0–P3/P5 transforms and planned P4 interface
      base.py             # Recording transform contract and no-op baseline
      experiments.py      # Recording-level filters and quality-control transforms
    feature_extraction/
      baseline.py         # BaselineFeatureExtractor: 11 values per epoch
    machine_learning/
      baseline.py         # Random-forest factory using supplied implementation
      models.py           # Model factory registry
      logistic_regression.py  # Scaled logistic regression
      svm.py              # Scaled RBF-SVM
      xgboost.py          # Optional XGBoost with fold-local label encoding/weights
      evaluation.py       # LOSO entry point using supplied evaluation harness
    track.py              # Connects the stages; inherits loading/reporting
    reference/            # Supplied course adapter and Sleep-EDF track
  bsp/                    # Supplied numerical helpers and synthetic recordings
run_experiment.py         # Repository-root experiment entrypoint
scripts/                  # Experiment/baseline evaluation, download, plots, smoke test
tests/                   # Offline baseline-regression checks
docs/                    # Dataset card, course instructions, background, architecture
results/                 # Saved reports and provenance
sleep_edf_data/           # Local EDF cache (ignored by Git)
RESULTS.md                # Existing experiment/results log
pyproject.toml            # Package metadata and direct dependencies
uv.lock                   # Exact dependency versions shared by the team
.python-version           # Shared Python minor version (3.11)
requirements-lock.txt      # Course scientific-package pins
requirements-real.txt      # Course real-data dependencies
```

Import the project track with `from sleepedf import SleepEDFTrack` and run the
baseline through `scripts/run_baseline.py`. Shared adapter types and helpers
live in `sleepedf.reference.adapter`; import reporting helpers with
`from sleepedf.reference import report as R`.
The project track provides EEG bandpass and wavelet filtering, near-flat epoch
exclusion, and suspected-clipping flags. P4 signal normalisation remains a
planned experiment. The supplied reference preprocessing recipes also remain
available through `sleepedf.reference.sleep_edf.SleepEDFTrack`.

## Run

All teammates use **Python 3.11**, matching the course CI, and the committed
`uv.lock`. The course guide does not prescribe a Python patch version. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) (version 0.9.26
or newer) once, then run this from the repository root after cloning or pulling:

```bash
uv sync --locked
```

This creates or synchronizes `.venv`, installs the project in editable mode,
and includes notebook kernel support. uv can download a Python 3.11 interpreter
if it is missing. Use the same command on macOS, Linux, and Windows; activation
is unnecessary when running commands through `uv run`. Systems without a
compatible prebuilt package may need build tools.

The direct dependency versions match the course requirements files: NumPy 2.2.6,
SciPy 1.15.3, scikit-learn 1.7.2, Matplotlib 3.10.9, PyWavelets 1.8.0,
Pillow 12.2.0, imbalanced-learn 0.14.2, MNE 1.10.1 and WFDB 4.3.0.
PyWavelets and imbalanced-learn support the supplied wavelet and SMOTE/ADASYN
options; MNE and WFDB are the course real-data dependencies. `pyproject.toml`
mirrors these pins, while `uv.lock` also fixes their transitive dependencies and
the project’s notebook tools. Platform-specific supporting packages may differ
by operating system. Select this repository's `.venv` as your editor's Python
interpreter and notebook kernel.

Check the installed environment:

```bash
uv run --locked python --version
uv run --locked python -c "import numpy, scipy, sklearn, mne; print(numpy.__version__, scipy.__version__, sklearn.__version__, mne.__version__)"
```

These should show Python 3.11.x and `2.2.6 1.15.3 1.7.2 1.10.1`.

Commit and share **`.python-version`, `pyproject.toml`, and `uv.lock` together**,
along with the course requirements files when those pins change.
Each teammate creates their own local `.venv` (already ignored by Git). Use
`uv sync --locked` after pulling dependency changes. `--locked` refuses an
outdated lockfile instead of silently changing the team's versions. See
[uv's locking and syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

Run the project with the shared environment:

```bash
# Offline synthetic check and regression tests
uv run --locked python -B scripts/smoke_test.py
uv run --locked python -B -m unittest discover -s tests -v

# Evaluate every PSG/hypnogram pair already in the local cache
uv run --locked python -B scripts/run_baseline.py --cache-dir sleep_edf_data

# Evaluate the first 3 cached subjects, both nights (6 recordings)
uv run --locked python -B scripts/run_baseline.py --n-subjects 3 --nights 1 2

# Choose specific subject IDs and only their first night
uv run --locked python -B scripts/run_baseline.py --subjects 0 1 2 --nights 1

# Optional: download the original subjects 0, 1, 2 (both nights)
uv run --locked python -B scripts/download.py

# Plot synthetic data, or a selected real epoch
uv run --locked python -B scripts/visualisation.py
uv run --locked python -B scripts/visualisation.py --cache-dir sleep_edf_data --record 0 --epoch 10
```

Omitting subject/night options evaluates all cached recordings. `--n-subjects N`
chooses the lowest N cached Sleep-Cassette subject IDs; `--subjects` chooses
explicit IDs instead. `--nights 1`, `--nights 2`, or `--nights 1 2` selects nights
per subject. Requested subjects and explicitly requested subject/night pairs must
be cached, and LOSO requires at least two subjects. Selection happens before EDF
loading; only selected recordings are loaded, hashed, and evaluated.

Baseline evaluation checks that each selected PSG has one matching hypnogram.
By default it performs no downloads, optional preprocessing, cropping, feature
selection or tuning.
Each run creates `results/real__P0_<UTC timestamp>/` containing:

- `report.json`: pooled metrics, confusion matrix, per-subject/fold metrics,
  spread, and aligned out-of-fold predictions and labels.
- `provenance.json`: requested selection, EDF filenames/hashes, recordings,
  settings, feature names, package versions, source hashes (including the new
  modules), and Git state.

Use `--output-dir <new-directory>` to choose another location. Existing output
directories are never overwritten. These are evaluation reports, not a saved
final model or submission CSV. Existing result artifacts retain their original
provenance and the downloaded EDFs remain in their original cache.

Results recorded before this environment switch used Python 3.13.3 and the
earlier package versions listed in their provenance. Keep those records as
historical evidence. Rerun the smoke check and baseline after switching to the
course environment, and log the new results with their seed, code revision and
environment before comparing them with subsequent experiments.

## Visualise saved results

Create plots from the `report.json` and `provenance.json` pairs already saved
under `results/`; no EDF loading or model training is needed:

```bash
# Compare every saved iteration
uv run --locked python -B scripts/visualisation_results.py

# Compare two selected saved iterations
uv run --locked python -B scripts/visualisation_results.py results/real__P0_20261008T115921097052Z results/real__P1_20261008T120846779566Z

# Read an explicit pair and choose the output directory
uv run --locked python -B scripts/visualisation_results.py --report results/real__P1_20261008T120846779566Z/report.json --provenance results/real__P1_20261008T120846779566Z/provenance.json --output-dir results/plots
```

Open `results/visualisations/index.html` to browse the comparison and each run's
dashboard: confusion percentages/counts, stage precision/recall/F1, true versus
predicted class distribution, pooled metrics, and subject/fold variation.
The index lists configuration, source and environment changes between iterations,
preserves full provenance, and links to `summary.csv`. Figures are PNG by default;
use `--format pdf` or `--format svg` for export, or `--show` to open figures.
`--results-dir <directory>` chooses another results root. Outputs are regenerated
on each invocation; use separate output directories to keep different selections.

Iterations are ordered by their saved UTC start times. Cohort IDs distinguish
input files, recordings, evaluated labels/groups/folds and epoch counts so runs
using different populations are visible. Compare settings and source/package
changes even within one cohort. Error bars are sample SD across finite group
scores, not confidence intervals; undefined scores and SD with fewer than two
finite groups are shown as N/A. Prediction arrays do not contain enough timing
information to reconstruct whole-night recording timelines safely.

## Run experiments

From the repository root, choose preprocessing and a model for cached real EDFs:

```bash
# EEG bandpass + RBF-SVM, first 3 cached subjects
.venv/bin/python run_experiment.py --preprocess bandpass --model rbf_svm --n-subjects 3

# Wavelet + logistic regression, selected subjects and first nights
.venv/bin/python run_experiment.py --preprocess wavelet --model logistic_regression --subjects 0 1 2 --nights 1 --seed 0

# Defaults: no preprocessing, random forest, all cached recordings
.venv/bin/python run_experiment.py --cache-dir sleep_edf_data
```

Preprocessing choices are `none`, `bandpass`, `wavelet`, `broken_segments`, and
`denoise_clipping`. `broken_segments` excludes detected near-flat epochs;
`denoise_clipping` only adds suspected-clipping flags and leaves signals unchanged.
Model choices are `random_forest`, `logistic_regression`, `rbf_svm`, and `xgboost`.
The seed defaults to 0. The existing `--subjects`/`--n-subjects`, `--nights`,
`--cache-dir`, and `--output-dir` selection options are available.

The experiment runner loads real cached EDFs and evaluates them with the existing
subject-wise LOSO harness. It performs no downloads or hyperparameter tuning.
Each run writes `report.json` and `provenance.json` to a new directory such as
`results/real__P1_rbf_svm_<UTC timestamp>/`. Provenance includes the selected model,
factory settings, preprocessing, inputs, source hashes, seed and package versions.
`--output-dir` chooses another new directory. The existing baseline script keeps
its behavior and output naming.

XGBoost is an optional dependency. Install the extra with:

```bash
uv sync --locked --extra xgboost
# Alternative for an environment managed with pip:
pip install -e '.[xgboost]'
```

Then run with `--model xgboost`. For Python experiments, use
`sleepedf.machine_learning.make_model(name, **kwargs)` and pass it through `clf=`
to `evaluate_loso`; see [the factory API and tuning guidance](docs/ARCHITECTURE.md).
Compare the same subjects and nights under identical features and LOSO folds.

## Development

Add a runtime dependency with `uv add <package>`, or an editor/notebook tool with
`uv add --dev <package>`. To change a pinned package deliberately, use, for
example, `uv add "mne==<new-version>"`. These commands update both the manifest
and lockfile. Run the smoke check and regression tests, review the version
changes, and commit both files together. Avoid individual `pip install` changes
in the shared project environment, since they bypass the lock.

The supplied reference notebook uses the installed `sleepedf.reference` package
and this repository's project kernel. For team experiments, import
`SleepEDFTrack` from `sleepedf` using the project kernel so runs use this team's
code and locked dependencies.

See [the architecture and experiment interfaces](docs/ARCHITECTURE.md) for
contracts, feature order, and where to implement each stage. The original
[dataset card](docs/sleep_edf_card.md), [course instructions](docs/sleep_edf_instructions.md),
[background map](docs/BACKGROUND_MAP.md), and [results log](RESULTS.md) are retained.
