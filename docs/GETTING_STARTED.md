# Getting started

The aim is to run one baseline, understand what it measures, and know where its files went. All commands below run from `CM2013_SleepEDF_Group14/` — the folder containing `pyproject.toml`.

## 1. Check your setup offline

Use the shared Python 3.11 environment:

```bash
uv sync --locked
uv run --locked python scripts/smoke_test.py
```

The smoke test generates synthetic recordings for five subjects, extracts 11 features per epoch, and checks repeatable predictions. A successful run ends with:

```text
PASS: shape, finite values, prediction count, five folds, confusion total, repeatability
```

After the environment is installed, this check needs no recordings or network connection. Its scores confirm that the pipeline runs; they are **not real-data experiment results**.

If `uv` is unavailable, install it before continuing. If `uv sync --locked` reports that the lockfile needs updating, use [Development](DEVELOPMENT.md) before changing shared dependency versions.

## 2. Get only the recordings for your first run

The project may already contain the recordings under `sleep_edf_data/`. If subjects 0, 1 and 2 with both nights are missing, download that selection:

```bash
uv run --locked python scripts/download.py --subjects 0 1 2 --nights 1 2
```

This command needs internet access. Each night requires a signal file (`PSG.edf`) and matching stage labels (`Hypnogram.edf`). Downloads stay in the local cache for later experiments.

## 3. Run the real-data baseline

```bash
uv run --locked python scripts/run_experiment.py --subjects 0 1 2 --nights 1 2 --seed 0
```

This command selects **three subjects, six recordings**, before loading data. It uses the default P0 preprocessing (leave signals unchanged), the 11 supplied features, and a 200-tree random forest with balanced class weights and scaling inside the training folds. It performs evaluation and writes reports; it does not download data or save a final submission model.

Keep the subject and night options in your command. **Omitting both evaluates all cached recordings**, which may include all 20 subjects in the team's cache. `--n-subjects 3` selects the three lowest cached subject IDs; `--subjects 0 1 2` states the IDs explicitly. A run needs at least two different subjects, and every explicitly requested subject/night pair must be cached.

The console reports the loaded recordings, feature extraction, LOSO folds, scores and output folder. LOSO means **leave one subject out**: train on the other subjects, test on that subject, then repeat. Both nights of a subject remain together. Scaling and any feature selection must be fitted using training subjects only.

## 4. Read the outputs

The run creates a new folder such as `results/real__P0_random_forest_<UTC timestamp>/`:

| File | What it tells you |
| --- | --- |
| `report.json` | Scores, confusion matrix, per-subject/fold results, spread and predictions made on held-out subjects |
| `provenance.json` | Subjects, nights, file hashes, pipeline settings, feature names, seed, package versions and source revision |

Existing output folders are never overwritten. See the [results guide](../results/README.md) to find the team's saved baseline and comparisons.

Read Cohen's κ, macro-F1 and the confusion matrix together. Rows of the confusion matrix are the true stage; columns are the predicted stage. Compare the per-subject scores as well as the pooled score: one easy subject can hide a hard one. Full recordings contain long Wake stretches, so accuracy alone can look reassuring while N1 is poorly recognized. These three-subject results are a small starting experiment, not evidence of performance on every sleeper.

For a dashboard without retraining, pass the new run folder explicitly:

```bash
uv run --locked python scripts/visualisation_results.py results/<your-run-folder>
```

Replace `<your-run-folder>` with the folder printed by the experiment. Open `results/visualisations/index.html` after plotting. Running the plotting script without a selection compares all saved runs recursively, including historical investigations.

## 5. Make one change and compare fairly

Keep the same subjects, nights and seed while changing one stage, for example:

```bash
uv run --locked python scripts/run_experiment.py --subjects 0 1 2 --nights 1 2 --seed 0 --preprocess bandpass
```

Use [Architecture](ARCHITECTURE.md) to find the preprocessing, features and model code. Record the change, reason, metrics with subject spread, output folder and source revision in [RESULTS.md](../RESULTS.md). Check provenance before comparing runs: different subjects, labels, code or environments can change the score.

For assessed work, also read the [Sleep-EDF instructions](course/sleep_edf_instructions.md), [team rubric](course/CAPSTONE_REPORT_RUBRIC.md), [individual assessment](course/INDIVIDUAL_ASSESSMENT.md) and [hold-out rules](course/HOLDOUT_EVALUATION.md). Follow the instructor's hold-out subject assignment before choosing subjects for training or evaluation.
