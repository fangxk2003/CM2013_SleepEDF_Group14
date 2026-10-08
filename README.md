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
pyproject.toml            # Package metadata and dependencies
```

`adapter.py`, `sleep_edf.py`, and `run_baseline.py` at the root are small
compatibility entry points. Existing baseline imports and the old baseline
command still work. New code should use `sleepedf` and the stage packages.
The supplied reference preprocessing recipes remain available through
`sleepedf.reference.sleep_edf.SleepEDFTrack`; the project track deliberately
exposes P1–P5 as **unimplemented experiments**, not completed preprocessing.

## Run

From this repository root, use the existing `.venv`, or create one and install:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

The scripts also work directly with the existing environment without installing
the local package. They resolve `src/` themselves.

```bash
# Offline synthetic check and regression tests
.venv/bin/python -B scripts/smoke_test.py
.venv/bin/python -B -m unittest discover -s tests -v

# Evaluate every PSG/hypnogram pair already in the local cache
.venv/bin/python -B scripts/run_baseline.py --cache-dir sleep_edf_data

# Optional: download the original subjects 0, 1, 2 (first night)
.venv/bin/python -B scripts/download.py

# Plot synthetic data, or a selected real epoch
.venv/bin/python -B scripts/visualisation.py
.venv/bin/python -B scripts/visualisation.py --cache-dir sleep_edf_data --record 0 --epoch 10
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

See [the architecture and experiment interfaces](docs/ARCHITECTURE.md) for
contracts, feature order, and where to implement each stage. The original
[dataset card](docs/sleep_edf_card.md), [course instructions](docs/sleep_edf_instructions.md),
[background map](docs/BACKGROUND_MAP.md), and [results log](RESULTS.md) are retained.
