# CM2013 Sleep-EDF — Group 14

We classify each 30-second segment of a sleep recording as **W, N1, N2, N3 or REM**, using EEG, EOG and EMG signals. The project has three stages: **preprocessing → feature extraction → machine learning**.

**Start with [Getting started](docs/GETTING_STARTED.md).** It takes you from an offline setup check to your first real-data baseline.

## First commands

Run these from this folder, `CM2013_SleepEDF_Group14/`, with `uv` available:

```bash
uv sync --locked
uv run --locked python scripts/smoke_test.py
```

The smoke test uses synthetic data and finishes with `PASS`. For the first real experiment, use exactly three subjects and both nights:

```bash
uv run --locked python scripts/run_experiment.py --subjects 0 1 2 --nights 1 2 --seed 0
```

If those recordings are missing, follow the download step in [Getting started](docs/GETTING_STARTED.md). Keep subject and night options explicit: omitting them evaluates **every cached recording**.

The defaults are no preprocessing (P0), 11 supplied features, and a scaled random forest. Real experiments use leave-one-subject-out validation: all nights from the test subject stay outside training.

## Where things belong

```text
scripts/                  Commands you run
  diagnostics/            Detailed P3/P5 investigations
src/sleepedf/             Pipeline you develop
  preprocessing/          Filters and artifact handling
  feature_extraction/     Numbers describing each epoch
  machine_learning/       Classifiers and evaluation
  track.py                Connects the three stages
  reference/              Supplied loading/evaluation framework
src/bsp/                  Supplied signal-processing helpers
results/                  Saved reports, provenance and plots
sleep_edf_data/            Downloaded recordings
tests/                   Offline regression checks
docs/                     Guides; original handouts in docs/course/
RESULTS.md                Experiment decisions and findings
```

`pyproject.toml`, `uv.lock` and `.python-version` define the shared Python 3.11 environment. Each teammate has their own `.venv/`.

| I want to… | Read |
| --- | --- |
| Run my first experiment | [Getting started](docs/GETTING_STARTED.md) |
| Know where to change the pipeline | [Architecture](docs/ARCHITECTURE.md) |
| Work with dependencies, notebooks or tests | [Development](docs/DEVELOPMENT.md) |
| Find and compare saved runs | [Results guide](results/README.md) |
| Read our experiment reasoning | [Results log](RESULTS.md) |
| Check the assignment and grading | [Course handouts](docs/course/README.md) |

The full course checkout and recovery backup sit beside this project under `../reference/` and `../archive/`. Everyday work stays here.
