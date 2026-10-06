# Pipeline architecture

`SleepEDFTrack` in `src/sleepedf/track.py` joins three independently editable stages. It inherits data loading, synthetic data generation and the evaluation harness from `reference/`. The reference code and `bsp/` retain the supplied algorithms; new project work belongs in the three stage packages.

## 1. Preprocessing

Input/output: a `Recording` with channel arrays shaped `(n_epochs, n_samples)`, per-epoch labels, a subject `group`, sampling rate `fs`, and metadata. The baseline uses 30-second epochs at 100 Hz. `Preprocessor.transform(recording)` is the recording-level interface.

`Recording` is provided in `adapter.py`

| Plan | Class | Status and contract |
| --- | --- | --- |
| P0 | `NoPreprocessing` | Implemented identity; the default |
| P1 | `EEGBandpass` | Stub; EEG 0.5–40 Hz, default filter order 4 |
| P2 | `WaveletDenoising` | Stub; db4, optional level and threshold mode |
| P3 | `BrokenSegmentHandling` | Stubs for `detect()` and `transform()`; flag/exclude broken epochs |
| P4 | `SubjectRecordingNormalisation` | Recording-level `transform()` stub; normalise signals before feature extraction |
| P5 | `TargetedDenoising` | Stub; identify impulse, baseline, mains or broadband corruption and record evidence |

`make_preprocessor.py`

```python
def make_preprocessor(name="none") -> Preprocessor:
    """Select a recording transform. Planned experiments fail when called."""
    choices = {
        "none": NoPreprocessing, "p0": NoPreprocessing,
        "bandpass": EEGBandpass, "p1": EEGBandpass,
        "wavelet": WaveletDenoising, "p2": WaveletDenoising,
        "broken_segments": BrokenSegmentHandling, "p3": BrokenSegmentHandling,
        "normalisation": SubjectRecordingNormalisation, "p4": SubjectRecordingNormalisation,
        "denoise": TargetedDenoising, "p5": TargetedDenoising,
    }
    key = str(name or "none").lower()
    if key not in choices:
        raise ValueError(f"Unknown preprocessing experiment: {name!r}")
    return choices[key]()
```

Example

```python
from sleepedf import SleepEDFTrack
from sleepedf.preprocessing import EEGBandpass

track = SleepEDFTrack()  # P0, runnable
recordings = track.smoke()
X, y, groups = track.build_dataset(recordings)

planned = EEGBandpass(low_hz=0.5, high_hz=40.0)
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

`default_baseline()` returns a fresh supplied scikit-learn pipeline: `StandardScaler → RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0)`. The factory delegates to the supplied implementation so its optional imbalance strategies remain available.

`evaluate_loso(track, X, y, groups, clf=None, cfg=None)` calls the suppliedsubject-separated evaluation harness and requests every subject fold. Existing selection, scaling and estimator fitting happen within training folds. Reports include pooled and per-subject metrics and out-of-fold predictions.

Add model factories in this package; pass a fresh estimator via `clf=` to compare models under the same split. Override `SleepEDFTrack.baseline()` if the project default later changes.

## Verification

`tests/test_pipeline.py` compares feature values, feature names and LOSO predictions with the supplied reference on a synthetic cohort.

It also checks P0 identity, placeholder failures, random-forest defaults and empty-recording feature width.
