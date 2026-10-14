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

## 5. Use machine learning to predict sleep stages

You can compare models now, without writing new features. The pipeline already converts each epoch into **11 features**: EEG band powers and signal measurements, EOG movement, and EMG strength. A model learns how those feature values relate to the labelled stages, then predicts stages for subjects it did not train on.

Choose a model with `--model`:

| Choice | Plain-language idea | Where its factory lives |
| --- | --- | --- |
| `random_forest` | Many decision trees vote for a stage; this is the baseline | [baseline.py](../src/sleepedf/machine_learning/baseline.py) |
| `logistic_regression` | Uses a weighted combination of features to separate stages | [logistic_regression.py](../src/sleepedf/machine_learning/logistic_regression.py) |
| `rbf_svm` | Uses curved boundaries to separate stages in feature space | [svm.py](../src/sleepedf/machine_learning/svm.py) |
| `xgboost` | Builds trees in sequence, each improving on earlier errors; optional | [xgboost.py](../src/sleepedf/machine_learning/xgboost.py) |

These commands change **only the model**. All use subjects 0, 1 and 2, both nights, seed 0, the same 11 features, and P0 preprocessing (`--preprocess none`). Each writes to a separate new folder. Run each once; use a new output folder name when rerunning because existing folders are never overwritten.

```bash
# Random-forest baseline
uv run --locked python scripts/run_experiment.py --subjects 0 1 2 --nights 1 2 --seed 0 --preprocess none --model random_forest --output-dir results/first_ml_random_forest

# Change the model to logistic regression
uv run --locked python scripts/run_experiment.py --subjects 0 1 2 --nights 1 2 --seed 0 --preprocess none --model logistic_regression --output-dir results/first_ml_logistic_regression

# Change the model to an RBF support-vector machine
uv run --locked python scripts/run_experiment.py --subjects 0 1 2 --nights 1 2 --seed 0 --preprocess none --model rbf_svm --output-dir results/first_ml_rbf_svm
```

For optional XGBoost, install and run with the extra enabled:

```bash
uv sync --locked --extra xgboost
uv run --locked --extra xgboost python scripts/run_experiment.py --subjects 0 1 2 --nights 1 2 --seed 0 --preprocess none --model xgboost --output-dir results/first_ml_xgboost
```

Every model uses the same subject-wise LOSO evaluation described above. The random forest, logistic regression and SVM pipelines fit their scalers **inside each training fold**, then apply them to that fold's test subject. XGBoost uses tree features without a scaler. Do not scale all subjects together before evaluation, or train a model once on all subjects and score it on those same subjects.

Compare the exact two run folders you want, for example:

```bash
uv run --locked python scripts/visualisation_results.py results/first_ml_random_forest results/first_ml_logistic_regression --output-dir results/first_ml_comparison
```

Open `results/first_ml_comparison/index.html`. Inspect κ, macro-F1, stage confusion and variation between subjects; check each run's provenance to confirm the same inputs and preprocessing. Repeat with `results/first_ml_rbf_svm` or `results/first_ml_xgboost` if you ran them. A higher score on this small cohort is a reason to investigate, not proof that a model is best for all subjects.

Each run's `report.json` contains held-out predictions and evaluation metrics; `provenance.json` records the model settings and inputs. These commands train a separate model for each fold and save **evaluation outputs**, not one fitted final model for deployment or a hold-out submission CSV.

When you want to change model settings in Python, start at [machine_learning/models.py](../src/sleepedf/machine_learning/models.py), which selects the factories listed above. Edit a factory to change its defaults; the current CLI selects a model and seed, but does not expose every tuning parameter. Use [Architecture](ARCHITECTURE.md) for the optional `make_model()` / `evaluate_loso()` Python example and tuning guidance. Keep tuning within training subjects and record your choices in [RESULTS.md](../RESULTS.md).

## 6. Make one change and compare fairly

Keep the same subjects, nights and seed while changing one stage, for example:

```bash
uv run --locked python scripts/run_experiment.py --subjects 0 1 2 --nights 1 2 --seed 0 --preprocess bandpass
```

Use [Architecture](ARCHITECTURE.md) to find the preprocessing, features and model code. Record the change, reason, metrics with subject spread, output folder and source revision in [RESULTS.md](../RESULTS.md). Check provenance before comparing runs: different subjects, labels, code or environments can change the score.

For assessed work, also read the [Sleep-EDF instructions](course/sleep_edf_instructions.md), [team rubric](course/CAPSTONE_REPORT_RUBRIC.md), [individual assessment](course/INDIVIDUAL_ASSESSMENT.md) and [hold-out rules](course/HOLDOUT_EVALUATION.md). Follow the instructor's hold-out subject assignment before choosing subjects for training or evaluation.
