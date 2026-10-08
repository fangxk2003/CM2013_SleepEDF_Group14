# Pipeline architecture

`SleepEDFTrack` in `src/sleepedf/track.py` joins three independently editable stages. It inherits data loading, synthetic data generation and the evaluation harness from `reference/`. The reference code and `bsp/` retain the supplied algorithms; new project work belongs in the three stage packages.

## 1. Preprocessing

Input/output: a `Recording` with channel arrays shaped `(n_epochs, n_samples)`, per-epoch labels, a subject `group`, sampling rate `fs`, and metadata. The baseline uses 30-second epochs at 100 Hz. `Preprocessor.transform(recording)` is the recording-level interface.

`Recording` is provided in `sleepedf.reference.adapter`.

| Plan | Class | Status and contract |
| --- | --- | --- |
| P0 | `NoPreprocessing` | Implemented identity; the default |
| P1 | `EEGBandpass` | Implemented EEG 0.5–40 Hz filter, default order 4; EOG/EMG unchanged |
| P2 | `WaveletDenoising` | Implemented EEG db4 denoising, optional level and threshold mode |
| P3 | `BrokenSegmentHandling` | Implemented near-flat EEG/EOG detection; flag or exclude epochs, with alignment preserved |
| P4 | `SubjectRecordingNormalisation` | Recording-level `transform()` stub; normalise signals before feature extraction |
| P5 | `TargetedDenoising` | Implemented suspected EEG clipping flags; signals unchanged, other noise remedies remain planned |

`make_preprocessor(name="none")` selects `none`, `bandpass`, `wavelet`,
`broken_segments`, `normalisation`, or `denoise_clipping` (aliases `p0`–`p5`).
The factory selects `action="exclude"` for P3; construct `BrokenSegmentHandling`
directly to flag without excluding. P4 raises `NotImplementedError` when used.

Example

```python
from sleepedf import SleepEDFTrack
track = SleepEDFTrack()
recordings = track.smoke()
X, y, groups = track.build_dataset(recordings, cfg={"preprocess": "bandpass"})
```

## 2. Feature extraction

`BaselineFeatureExtractor.transform(recording, cfg=None)` returns `(X, y, group)`.

`X` has one row per epoch and these 11 columns in fixed order:

1. `eeg_delta` — band power, 0.5–4 Hz
2. `eeg_theta` — band power, 4–8 Hz
3. `eeg_alpha` — band power, 8–11 Hz
4. `eeg_sigma` — band power, 11–16 Hz
5. `eeg_beta` — band power, 16–30 Hz
6. `eeg_hjorth_activity`
7. `eeg_hjorth_mobility`
8. `eeg_hjorth_complexity`
9. `eeg_spec_entropy`
10. `eog_movement` — mean absolute first difference
11. `emg_rms`

The numerical functions still come from `bsp.sleep_pipeline.epoch_features`; Welch remains the default spectral estimator.

The supplied optional spectral estimators are preserved. `feature_names()` matches the column order.

Add alternative extractors in this package and connect them through the track's `extract_features()` and `feature_names()` methods. Preserve row/label alignment. `build_dataset()` concatenates recording features and expands subject IDs into one group ID per row.

## 3. Machine learning

`default_baseline()` returns a fresh supplied scikit-learn pipeline: `StandardScaler → RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0)`. It remains the track's default. `make_random_forest()` delegates to that implementation so the same defaults and optional `imbalance=` strategies remain available.

The public registry `make_model(name, **kwargs)` accepts these model names:

| Name | Factory | Default behavior |
| --- | --- | --- |
| `random_forest` | `make_random_forest` | Supplied scaler + 200-tree random forest, balanced class weights, seed 0 |
| `logistic_regression` | `make_logistic_regression` | Scaler + L2 logistic regression, balanced class weights; `C` controls regularization |
| `rbf_svm` | `make_rbf_svm` | Scaler + RBF SVM, balanced class weights; accepts `C` and `gamma`, probability estimation disabled |
| `xgboost` | `make_xgboost` | Optional boosted-tree classifier with training-fold class weighting and label encoding |

XGBoost requires the `xgboost` extra (`uv sync --locked --extra xgboost`). Its
`ClassWeightedXGBClassifier` wrapper is cloneable, uses `LabelEncoder` to convert
stage strings into contiguous integer labels at fit time, and converts predictions
back to the original labels. With `class_weight="balanced"`, it derives sample
weights from the labels in that fit call. The wrapped XGBoost estimator exposes
nested parameters for scikit-learn searches. Tree models require no additional
feature scaling.

`evaluate_loso(track, X, y, groups, clf=None, cfg=None)` calls the supplied
subject-separated evaluation harness and requests every subject fold. Existing
selection, scaling and estimator fitting happen within training folds. Reports
include pooled and per-subject metrics and out-of-fold predictions.

```python
from sleepedf import SleepEDFTrack
from sleepedf.machine_learning import evaluate_loso, make_model

track = SleepEDFTrack()
X, y, groups = track.build_dataset(track.smoke(n_subjects=3, n_epochs=40))
report = evaluate_loso(
    track, X, y, groups,
    clf=make_model("logistic_regression", C=1.0, class_weight="balanced"),
)
```

Pass a fresh estimator through `clf=` to compare models under the same split;
`cfg["imbalance"]` only configures the supplied default, not a custom classifier.
Use `imbalance=` for the random-forest factory and `class_weight=` for logistic
regression, SVM and XGBoost. Tune only on training subjects, using inner group-wise
validation when evaluating tuned models with outer LOSO. Preserve both nights of
each subject in the same group. The current factories consume the existing
per-epoch features; adding temporal context requires a separate dataset design
that preserves recording boundaries and epoch positions.

## Experiment entrypoint

`run_experiment.py` in the repository root delegates to
`scripts/run_experiment.py`. It selects a preprocessing transform and model,
then reuses real EDF loading, the existing 11 features, LOSO evaluation and
report/provenance saving:

```bash
.venv/bin/python run_experiment.py --preprocess bandpass --model rbf_svm --n-subjects 3
```

Preprocessing choices are `none`, `bandpass`, `wavelet`, `broken_segments`, and
`denoise_clipping`; model choices are the four registry names above. Defaults are
`none`, `random_forest`, and seed 0. The runner supports `--seed`,
`--subjects`/`--n-subjects`, `--nights`, `--cache-dir`, and `--output-dir`.
It uses cached real EDFs and performs no downloads or tuning. P5 is flag-only.

Outputs are `report.json` and `provenance.json` in a unique directory such as
`results/real__P1_rbf_svm_<UTC timestamp>/`, with the model included in the name.
Provenance records the selected model and factory settings alongside the seed,
preprocessing, EDF hashes, feature names, source hashes and environment versions.
`scripts/run_baseline.py` retains its existing behavior.

## Verification

`tests/test_pipeline.py` compares feature values, feature names and LOSO predictions with the supplied reference on a synthetic cohort.

It also checks P0 identity, placeholder failures, random-forest defaults and empty-recording feature width.
