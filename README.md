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
    preprocessing/        # P0 implementation and P1–P5 experiment interfaces
      base.py             # Recording transform contract and no-op baseline
      experiments.py      # P1–P5 recording-level preprocessing placeholders
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
scripts/                  # Baseline evaluation, download, visualisation, smoke test
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
The supplied reference preprocessing recipes remain available through
`sleepedf.reference.sleep_edf.SleepEDFTrack`; the project track deliberately
exposes P1–P5 as **unimplemented experiments**, not completed preprocessing.

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

## Compare models

`sleepedf.machine_learning` exposes `make_model(name, **kwargs)` and the named
factories `make_random_forest`, `make_logistic_regression`, `make_rbf_svm`, and
`make_xgboost`. The track and baseline script still use the supplied random forest
by default. Compare classifiers by passing a fresh estimator through `clf=`:

```python
from sleepedf import SleepEDFTrack
from sleepedf.machine_learning import evaluate_loso, make_model

track = SleepEDFTrack()
recordings = track.smoke(n_subjects=3, n_epochs=40, seed=0)
X, y, groups = track.build_dataset(recordings)

for name in ("random_forest", "logistic_regression", "rbf_svm"):
    report = evaluate_loso(track, X, y, groups, clf=make_model(name))
    print(name, report["summary"], "macro-F1:", report["macro_f1"])
```

The synthetic example checks functionality; it does not establish performance on
Sleep-EDF. For a real comparison, build the dataset once from the same selected
recordings and use identical preprocessing, features and subject folds for every
model. Both nights of a subject share a group. Scaling and feature selection are
fitted within each training fold.

Logistic regression uses L2 regularization and RBF-SVM uses an RBF kernel. Both
include `StandardScaler` and default to `class_weight="balanced"`. Tune `C` for
logistic regression and `C`/`gamma` for SVM through factory arguments. SVM defaults
to `probability=False`. Random forest retains the supplied defaults and
`imbalance=` options; the other factories configure weighting with `class_weight=`.
`cfg["imbalance"]` does not change a classifier passed through `clf=`.

XGBoost is an optional dependency. Install the extra with:

```bash
uv sync --locked --extra xgboost
# Alternative for an environment managed with pip:
pip install -e '.[xgboost]'
```

Then include `"xgboost"` in the loop. Its classifier encodes stage labels and
computes balanced sample weights from each training fold. It supports scikit-learn
cloning and exposes the underlying XGBoost estimator's nested parameters for
tuning.

Choose hyperparameters using subject-aware validation inside the development
training data. If reporting tuned LOSO scores, use inner subject-wise validation
inside each outer training fold; do not choose settings from the outer held-out
subject. Report Cohen's kappa and its subject spread, macro-F1, per-stage scores,
and confusion matrices. These factories use each epoch's existing features;
temporal context is a separate experiment.

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
