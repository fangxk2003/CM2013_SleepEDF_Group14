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
.python-version           # Shared Python version (3.13.3)
```

`adapter.py`, `sleep_edf.py`, and `run_baseline.py` at the root are small
compatibility entry points. Existing baseline imports and the old baseline
command still work. New code should use `sleepedf` and the stage packages.
The supplied reference preprocessing recipes remain available through
`sleepedf.reference.sleep_edf.SleepEDFTrack`; the project track deliberately
exposes P1–P5 as **unimplemented experiments**, not completed preprocessing.

## Run

All teammates use **Python 3.13.3** and the committed `uv.lock`. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) (version 0.9.26
or newer) once, then run this from the repository root after cloning or pulling:

```bash
uv sync --locked
```

This creates or synchronizes `.venv`, installs the project in editable mode,
and includes notebook kernel support. uv can download the pinned Python version
if it is missing. Use the same command on macOS, Linux, and Windows; activation
is unnecessary when running commands through `uv run`. Prebuilt scientific
packages target recent desktop systems (including macOS 12+ on Apple Silicon,
Linux with glibc 2.27+, and Windows x64). Other architectures may need build
tools when a pinned wheel is unavailable.

The NumPy, SciPy, scikit-learn, MNE, and Matplotlib versions match the existing
baseline environment. PyWavelets and imbalanced-learn are included for the
supplied wavelet and SMOTE/ADASYN options. The lock also fixes their transitive
dependencies; platform-specific supporting packages may differ by operating
system. Select this repository's `.venv` as your editor's Python interpreter and
notebook kernel.

Commit and share **`.python-version`, `pyproject.toml`, and `uv.lock` together**.
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

# Optional: download the original subjects 0, 1, 2 (first night)
uv run --locked python -B scripts/download.py

# Plot synthetic data, or a selected real epoch
uv run --locked python -B scripts/visualisation.py
uv run --locked python -B scripts/visualisation.py --cache-dir sleep_edf_data --record 0 --epoch 10
```

Baseline evaluation checks that each PSG has one matching hypnogram. It performs
no downloads, optional preprocessing, cropping, feature selection or tuning.
Each run creates `results/real_baseline_<UTC timestamp>/` containing:

- `report.json`: pooled metrics, confusion matrix, per-subject/fold metrics,
  spread, and aligned out-of-fold predictions and labels.
- `provenance.json`: EDF filenames/hashes, recordings, settings, feature names,
  package versions, source hashes (including the new modules), and Git state.

Use `--output-dir <new-directory>` to choose another location. Existing output
directories are never overwritten. These are evaluation reports, not a saved
final model or submission CSV. Existing result artifacts retain their original
provenance and the downloaded EDFs remain in their original cache.

## Development

Add a runtime dependency with `uv add <package>`, or an editor/notebook tool with
`uv add --dev <package>`. To change a pinned package deliberately, use, for
example, `uv add "mne==<new-version>"`. These commands update both the manifest
and lockfile. Run the smoke check and regression tests, review the version
changes, and commit both files together. Avoid individual `pip install` changes
in the shared project environment, since they bypass the lock.

The supplied reference notebook retains its original course bootstrap, which
can clone the upstream course repository. For team experiments, import
`SleepEDFTrack` from `sleepedf` using the project kernel so runs use this team's
code and locked dependencies.

See [the architecture and experiment interfaces](docs/ARCHITECTURE.md) for
contracts, feature order, and where to implement each stage. The original
[dataset card](docs/sleep_edf_card.md), [course instructions](docs/sleep_edf_instructions.md),
[background map](docs/BACKGROUND_MAP.md), and [results log](RESULTS.md) are retained.
