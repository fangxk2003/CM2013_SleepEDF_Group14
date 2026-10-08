# Results log — `Sleep EDF`, `Group 14`

> **Copy this file into YOUR team's project repository** (not this scaffold repo) as
> `RESULTS.md`, and add one row per iteration as you go — not the night before the deadline.
> It exists because Chapter 16 §16.3 defines an iteration as *done* only when it (1) runs end
> to end to a result, (2) reports the primary metric **with its spread**, (3) is committed with
> a note saying what changed and why, and (4) is at least as good as the previous iteration —
> **or** says in writing why the change was kept anyway. This table is where (2), (3) and (4)
> land; `git log` is where (1) becomes checkable. It also makes writing the report an assembly
> job rather than an archaeology project.
>
> **This file is graded.** `CAPSTONE_REPORT_RUBRIC.md` **Criterion 9 — Iteration & revision
> history (3 pts)** takes `RESULTS.md` as its evidence, and it asks for one specific thing that
> a forward-only log cannot show: **at least one earlier decision you went back and changed
> because of a downstream result**, with the symptom that sent you back named. A row that
> *lowered* the headline metric and was kept for a stated reason is full-marks evidence, not a
> weakness. Rows here are also direct evidence for "Reproducibility & engineering" and "Report
> quality & defence".

**Track:** `sleep_edf` ·
**Split unit:** `subject` · **Primary metric:** Cohen's kappa ·
**Evaluation mode:** new-subject (LOSO; five synthetic subjects / three real subjects)

## Iteration log

Paste the metric straight from the harness — `rep["summary"]` prints the required shape, e.g.
`mean cohens_kappa 0.61 (sd 0.12, range 0.34-0.73 across 8 subjects)`. A pooled number with no
spread is half a result.

| # | Date | What changed & why (one line) | Primary metric **with spread** | Better than previous? | If not — why it was kept | Commit |
|---|---|---|---|---|---|---|
| 1 (smoke only) | 2026-09-30 | Supplied baseline on synthetic medium-difficulty data | mean kappa 0.682 (sd 0.318, range 0.155-0.933 across 5 subjects) | Baseline | Functional check only | `0a48f51` |
| 1a (real baseline) | 2026-10-01 | Same supplied model and features on SC4001, SC4011, SC4021; establish a real-data reference | mean kappa 0.618 (sd 0.202, range 0.386-0.760 across 3 subjects) | Baseline on real data | - | `dde6e44` |
| ~~1b (real baseline)~~ | ~~2026-10-02~~ | ~~Supplied baseline run end-to-end on real Sleep-EDF data to establish the reference performance before DSP/feature changes~~ | ~~mean cohens_kappa 0.720 (sd 0.119, range 0.589-0.822 across 3 subjects)~~ | ~~Baseline on real data~~ | - | ~~`69d392f`~~ |
| 2a | 2026-10-06 | P0: current baseline on 3 subjects × 2 nights, no preprocessing, to establish the development reference | mean cohens_kappa 0.704 (sd 0.087, range 0.627-0.799 across 3 subjects) | Baseline for Iteration 2 development experiments | - | `e6cada5` |
| 2b | 2026-10-06 | P1: EEG 0.5–40 Hz zero-phase Butterworth band-pass; tested as a controlled preprocessing candidate | mean cohens_kappa 0.737 (sd 0.074, range 0.666-0.813 across 3 subjects) | Yes — pooled κ 0.710→0.742; macro-F1 0.648→0.683 | - | `e6cada5` |
| 2c | 2026-10-06 | P2: EEG wavelet denoising using db4, automatic decomposition level (max 5), soft thresholding and automatic per-epoch VisuShrink threshold; EOG and EMG unchanged | mean Cohen's kappa 0.743 (sd 0.065, range 0.669–0.781 across 3 subjects) | Yes vs P0; only marginally higher kappa than P1, with lower macro-F1 and balanced accuracy | P1 retained as the current preprocessing candidate because it provides more balanced stage-wise performance |`97447e2` |
| 2d | 2026-10-06 | P3: objective broken-segment handling; epochs with simultaneous full-epoch near-flat EEG and EOG were excluded using thresholds fixed from QC before evaluation; 3/16,688 epochs were removed | mean Cohen's kappa 0.694 (sd 0.109, range 0.582–0.800 across 3 subjects) | No — mean kappa decreased from 0.704 to 0.694 vs P0; macro-F1 0.648→0.644 | Retained as a data-quality safeguard rather than a performance improvement: the rule removes objectively unreliable epochs and provides predefined handling if similar sustained flatlines occur in additional recordings |`7c34ebf` |

| 2e | 2026-10-08 | P5: EEG suspected-clipping detection using recording-specific extrema and plateaus of at least 3 consecutive samples (30 ms); 57/16,688 epochs flagged, with no signal modification or exclusion | mean cohens_kappa 0.704 (sd 0.087, range 0.627-0.799 across 3 subjects) | No improvement vs P0; all 16,688 predictions identical (0 differences) | Retained as a label-independent signal-quality safeguard: suspected clipping is documented without reconstructing potentially lost EEG information or altering classification | `7c838a1` |

| 3a (course environment, smoke) | 2026-10-08 | Align Python and dependencies with the course guide; repeat the P0 synthetic smoke check (5 subjects × 80 epochs, seed 0) | mean cohens_kappa 0.614 (sd 0.314, range 0.137-0.933 across 5 subjects) | New environment reference | Kept to match the course pins; scores from the earlier environment remain historical | Pending commit; source HEAD `e470502`, see environment record below |
| 3b (course environment, real baseline) | 2026-10-08 | Repeat P0 on subjects 0, 1, 2, both nights, under the course environment; unchanged features and classifier, seed 0 | mean cohens_kappa 0.723 (sd 0.115, range 0.596-0.822 across 3 subjects) | New environment reference | Kept to match the course pins; rerun preprocessing/model candidates under this environment before comparing them | Pending commit; source HEAD `e470502`, see saved provenance below |
| 4 |  |  |  | yes / no |  |  |

*"Better" means better under the same honest harness — same split unit, same evaluation mode,
same seed. A change that lowers the metric can still be the right call (simpler, faster, more
robust across subjects, removes a leak). Say so in the column instead of quietly reverting it:
"kept — κ fell 0.02 but the worst-subject κ rose from 0.18 to 0.31" is a stronger result than a
silent higher mean.*

## Course environment alignment — 2026-10-08

The project now selects Python **3.11**, matching the course CI; the installed
patch version used for these checks was **3.11.14**. The course guide specifies
no Python patch version. Both course requirements files were copied unchanged,
their pins mirrored in `pyproject.toml`, and `uv.lock` regenerated. Installed
versions were verified against all nine course pins: NumPy 2.2.6, SciPy 1.15.3,
scikit-learn 1.7.2, Matplotlib 3.10.9, PyWavelets 1.8.0, Pillow 12.2.0,
imbalanced-learn 0.14.2, MNE 1.10.1 and WFDB 4.3.0.

Validation passed: the synthetic smoke check including repeatability, all
**33 unit tests** (including optional XGBoost), and the six-recording real-data
P0 baseline. The synthetic run's pooled kappa was 0.616, macro-F1 0.686 and
accuracy 0.713. The real run covered 16,688 epochs and three subject-separated
LOSO folds; pooled kappa was 0.731, macro-F1 0.625, balanced accuracy 0.595 and
accuracy 0.869. These establish new environment references. Earlier experiment
scores retain their recorded environments and should be rerun under the course
pins before comparison.

Reproduce from the team repository root:

```bash
uv sync --locked
uv run --locked python -B scripts/smoke_test.py
uv run --locked python -B -m unittest discover -s tests -v
uv run --locked python -B scripts/run_baseline.py --subjects 0 1 2 --nights 1 2
```

To include optional XGBoost in verification, use `uv sync --locked --extra xgboost`
and `uv run --locked --extra xgboost python -B -m unittest discover -s tests -v`.

Saved artifacts: [report](results/real__P0_20261008T112303518579Z/report.json),
[provenance](results/real__P0_20261008T112303518579Z/provenance.json), and
[complete environment/check record](results/real__P0_20261008T112303518579Z/environment.json).
The provenance records source HEAD `e470502` together with the uncommitted
working-tree state and source/data SHA-256 hashes. The environment record adds
all installed package versions and SHA-256 hashes of the five environment files.

## Real EDF baseline - 2026-10-01

**Completed on real Sleep-EDF recordings, not synthetic data.** The supplied
baseline was unchanged: StandardScaler + RandomForestClassifier with 200 trees,
seed 0 and balanced class weights. Preprocessing was `none`, spectral estimation
was `welch`, all 11 supplied features were retained, and no wake cropping was
applied. Stages 3/4 were merged into N3 and Movement/Unknown dropped by the loader.
LOSO used three folds, each holding out all epochs of one subject. This small
subset is a baseline reference, not an untouched final test cohort.

Reproduce from the repository root:

```bash
.venv/bin/python -B run_baseline.py --cache-dir sleep_edf_data
```

The runner evaluates every PSG/hypnogram pair under the cache directory. This run
used exactly SC4001E0, SC4011E0 and SC4021E0. Adding files changes the cohort.
Results and provenance for this run are saved in
`results/real_baseline_20261001T193926289176Z/`:
- `report.json`: metrics, confusion matrix, per-subject/fold results and aligned
  out-of-fold labels, predictions, groups and fold IDs.
- `provenance.json`: input EDF SHA-256 hashes, configuration, package versions,
  source hashes and Git state (HEAD `dde6e44419ce6f8254717e1c7d3d027620a8cff3`).

| Metric | Pooled result |
|---|---|
| Accuracy | 0.756 (unrounded: 0.7564195736) |
| Cohen's kappa | 0.579 |
| Macro-F1 | 0.618 |
| Balanced accuracy | 0.699 |

Harness summary: `mean cohens_kappa 0.618 (sd 0.202, range 0.386-0.760 across 3 subjects)`.

| Held-out subject | Recording | Epochs | Accuracy | Kappa | Macro-F1 |
|---|---|---|---|---|---|
| SC400 | SC4001E0 | 2650 | 0.902264 | 0.760222 | 0.677821 |
| SC401 | SC4011E0 | 2802 | 0.530335 | 0.386262 | 0.577265 |
| SC402 | SC4021E0 | 2804 | 0.844508 | 0.707555 | 0.684391 |

Confusion matrix: rows=true, columns=predicted.

| True / predicted | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| W | 4312 | 255 | 111 | 1076 | 6 |
| N1 | 43 | 120 | 74 | 1 | 23 |
| N2 | 44 | 33 | 1193 | 28 | 59 |
| N3 | 61 | 0 | 74 | 284 | 1 |
| REM | 1 | 13 | 108 | 0 | 336 |

There were 8,256 epochs: W=5,760, N1=261, N2=1,357, N3=420, REM=458.
Wake accounts for about 69.8% of epochs, so accuracy alone is insufficient.
SC401 has the weakest kappa; the largest pooled confusion is Wake predicted as
N3 (1,076 epochs). No causal explanation or improvement is claimed yet.
These real-data metrics must not be interpreted as a gain/loss against the
synthetic smoke metrics, because the datasets differ.

Checks passed: all three EDF pairs loaded; 100 Hz / 3,000 samples per epoch;
finite feature matrix of shape (8256, 11); one prediction per epoch; three LOSO
folds; recomputed confusion matrix and kappa match the saved report; source hashes
match; JSON serialization, missing-input handling and overwrite protection tested.
Python package versions: NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, MNE 1.13.2.
The first execution reached the output step but could not save due to sandbox
permissions; the same configuration was rerun with write permission. No tuning
was performed. Runner and documentation prepared/executed with Codex assistance
at Xingkai's request. No commit or push was performed.

## Synthetic smoke-test evidence - 2026-09-30

PASS. These are synthetic-data results, not performance on real Sleep-EDF.

### Configuration

- Data: `track.smoke(n_subjects=5, n_epochs=80, seed=0, difficulty="medium")`.
- 400 labelled 30-second epochs, 100 Hz, 11 supplied features per epoch.
- Preprocessing: `none`; spectral estimator: `welch`; feature selection: `none`.
- Classifier: `adapter.default_baseline(seed=0, n_estimators=200, imbalance="balanced")`.
- Pipeline: 200-tree random forest (balanced class weights) -> StandardScaler
- Validation: five LOSO folds; 320 training and 80 test epochs per fold.
- Environment: Python 3.13.3; NumPy 2.5.3; SciPy 1.18.1; scikit-learn 1.9.1; Matplotlib 3.11.2.

### Results

| Pooled metric | Harness value |
|---|---|
| Accuracy | 0.753 |
| Cohen's kappa | 0.677 |
| Macro-F1 | 0.730 |
| Balanced accuracy | 0.747 |

Harness summary: `mean cohens_kappa 0.682 (sd 0.318, range 0.155-0.933 across 5 subjects)`.

| Held-out subject | Epochs | Accuracy | Kappa | Macro-F1 |
|---|---|---|---|---|
| S01 | 80 | 0.9125 | 0.867862 | 0.894755 |
| S02 | 80 | 0.3125 | 0.155470 | 0.354552 |
| S03 | 80 | 0.7125 | 0.615706 | 0.487004 |
| S04 | 80 | 0.9500 | 0.933040 | 0.951562 |
| S05 | 80 | 0.8750 | 0.837662 | 0.862940 |

Confusion matrix: rows are true labels, columns are predicted labels.

| True \ predicted | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| W | 28 | 1 | 0 | 0 | 13 |
| N1 | 7 | 25 | 7 | 0 | 2 |
| N2 | 0 | 38 | 103 | 12 | 2 |
| N3 | 0 | 1 | 12 | 71 | 0 |
| REM | 1 | 1 | 2 | 0 | 74 |

Observed labels: W=42, N1=41, N2=155, N3=84, REM=78. The largest confusion is N2 predicted as N1 (38 epochs). S02 is much weaker than the other subjects, the pooled score alone hides this.

### Code

```bash
import numpy as np
from adapter import default_baseline
from sleep_edf import SleepEDFTrack

track = SleepEDFTrack()
recs = track.smoke(n_subjects=5, n_epochs=80, seed=0, difficulty="medium")
X, y, groups = track.build_dataset(recs)
assert X.shape == (400, 11) and np.isfinite(X).all()
clf = default_baseline(seed=0, n_estimators=200, imbalance="balanced")
rep = track.evaluate(X, y, groups, clf=clf)
assert len(rep["y_pred"]) == len(y) == 400
assert rep["n_groups"] == len(rep["per_fold"]) == 5
assert np.asarray(rep["confusion"]).sum() == 400
assert all(np.isfinite(rep[k]) for k in (
    "accuracy", "cohens_kappa", "macro_f1", "balanced_accuracy"))
repeat = track.run_smoke()
assert np.array_equal(rep["y_true"], repeat["y_true"])
assert np.array_equal(rep["y_pred"], repeat["y_pred"])
print(rep["summary"])
for key in ("accuracy", "cohens_kappa", "macro_f1", "balanced_accuracy",
            "labels", "confusion", "per_group", "spread"):
    print(key, rep[key])
print("PASS: shape, finite values, prediction count, five folds, confusion total, repeatability")
```
### P0 vs P1 vs P2 preprocessing comparison — 2026-10-06

Development subset: 3 subjects (SC400, SC401, SC402), 2 nights per subject, 6 recordings and 16,688 epochs. The dataset, 11 supplied features, Random Forest classifier and subject-wise LOSO evaluation were kept unchanged across all three experiments. Only the preprocessing step was changed.

- P0: no preprocessing.
- P1: EEG 0.5–40 Hz zero-phase Butterworth band-pass; EOG and EMG unchanged.
- P2: EEG wavelet denoising using db4, automatic decomposition level capped at 5, soft thresholding and an automatically estimated per-epoch VisuShrink threshold; EOG and EMG unchanged.


| Metric | P0 | P1 | P2 |
|---|---:|---:|---:|
| Mean subject Cohen's kappa | 0.704 | 0.737 | **0.743** |
| SD subject Cohen's kappa | 0.087 | 0.074 | **0.065** |
| Subject kappa range | 0.627–0.799 | 0.666–0.813 | 0.669–0.781 |
| Accuracy | 0.849 | 0.868 | **0.872** |
| Pooled Cohen's kappa | 0.710 | 0.742 | **0.747** |
| Macro-F1 | 0.648 | **0.683** | 0.677 |
| Balanced accuracy | 0.650 | **0.680** | 0.666 |

Per-stage recall also increased for all five stages:

| Stage | P0 | P1 | P2 |
|---|---:|---:|---:|
| W | 0.918 | 0.933 | **0.944** |
| N1 | 0.408 | **0.416** | 0.315 |
| N2 | 0.900 | **0.905** | 0.897 |
| N3 | 0.460 | **0.533** | 0.513 |
| REM | 0.566 | 0.611 | **0.660** |

P0 VS P1
The largest recall improvements were observed for N3 and REM. N3→N2
confusions decreased from 320 to 273 epochs and REM→N2 confusions from
356 to 313 epochs.

P0 VS P2, P2 VS P1
P2 improved substantially over P0 in the primary metric (mean subject Cohen's kappa: 0.704 to 0.743) and also reduced between-subject variability (SD: 0.087 to 0.065). It also improved Wake, N3 and REM recall relative to P0. However, the comparison between P1 and P2 was less clear. P2 increased mean subject kappa by only 0.006 and pooled kappa by 0.005 relative to P1, while macro-F1 decreased from 0.683 to 0.677 and balanced accuracy decreased from 0.680 to 0.666.

The main stage-level trade-off was N1. Its recall decreased from 0.416 with P1 to 0.315 with P2. N2 recall also decreased slightly from 0.905 to 0.897, while REM recall improved from 0.611 to 0.660.

**Decision:** 
P1 : retain the 0.5–40 Hz EEG band-pass as the current preprocessing
candidate. It improved all primary and secondary metrics on the development
subset and reduced between-subject kappa variability. This decision remains
provisional until the selected final pipeline is evaluated on a larger
subject set.
P2 : Decision: P2 is considered beneficial relative to P0, but not clearly superior to P1. The small improvement in the primary metric is accompanied by less balanced stage-wise performance, particularly for N1. P1 is therefore retained as the current preprocessing candidate for subsequent development experiments. This decision remains provisional because the comparison is based on only three development subjects.

### P3 — Broken-segment handling

**Motivation.**  
The real-data QC identified a localized near-flat event in recording
`SC4012E0`. A continuous short-window analysis confirmed that EEG and EOG
were simultaneously near-flat for approximately 100 s, spanning the end of
epoch 2844 and epochs 2845–2847. Shorter low-variability periods were also
observed in individual channels in other recordings, so low variability in a
single channel was not considered sufficient evidence for exclusion.

**Decision.**  
P3 uses a conservative, label-independent detector based on the standard
deviation of the complete 30 s epoch. An epoch is classified as broken only
when both:

- EEG standard deviation < `4.820492569885e-06 V`
- EOG standard deviation < `7.200175602286e-06 V`

The thresholds were fixed from the QC analysis before evaluating P3 and are
not recomputed during LOSO evaluation. Epochs detected as broken are excluded,
with the same mask applied to EEG, EOG, EMG and labels. The detected indices
and QC decision are also stored in the recording metadata.

On the current six-recording development dataset, the detector excluded only
three epochs, all from `SC4012E0`: epochs 2845, 2846 and 2847. The dataset
therefore changed from 16,688 to 16,685 epochs (3/16,688 = 0.018%).

**Controlled comparison with P0.**

| Metric | P0 | P3 |
|---|---:|---:|
| Mean subject-wise Cohen's kappa | 0.704 | 0.694 |
| SD of subject-wise kappa | 0.087 | 0.109 |
| Subject-wise kappa range | 0.627–0.799 | 0.582–0.800 |
| Accuracy | 0.849 | 0.843 |
| Pooled Cohen's kappa | 0.710 | 0.700 |
| Macro-F1 | 0.648 | 0.644 |
| Balanced accuracy | 0.650 | 0.647 |

P3 did not improve classification performance on the current three-subject
development cohort. Mean subject-wise kappa decreased by 0.010 and the other
summary metrics also decreased slightly. However, only three epochs (0.018%
of the dataset) were removed, so this experiment provides little evidence
about the effect of broken-segment handling when recordings contain more
substantial dropout.

**Decision after evaluation.**  
Retain P3 as a data-quality safeguard rather than as a performance-enhancing
preprocessing step. The exclusion criterion was motivated by signal QC rather
than classifier performance and targets epochs considered unreliable for
physiological feature extraction. Keeping the detector in the pipeline also
provides a predefined handling rule if similarly sustained multichannel
near-flat segments occur in additional recordings. The small decrease in
classification metrics on the current development cohort is therefore not
used as a reason to retain objectively broken epochs.


### P5 — EEG clipping detection and quality flagging

**Motivation.**

Exploratory EEG quality-control analysis identified unusually repeated amplitude extrema in several Sleep-EDF recordings, particularly SC4012E0 and SC4022E0. Visual inspection showed flattened EEG peaks, suggesting possible amplitude saturation or clipping. However, without confirmed acquisition limits, these events are classified as suspected rather than confirmed clipping.

**Detection method.**

P5 uses a deterministic, label-independent detector applied separately to each complete recording:

1. Compute the minimum and maximum EEG amplitude within the recording.
2. Identify sequences of at least three consecutive samples exactly equal to either extreme.
3. Flag any 30-second epoch containing such a sequence.
4. Preserve all original EEG, EOG and EMG signals, labels and epochs.

At 100 Hz, three samples correspond to 30 ms. The upper and lower extrema are checked separately to avoid combining opposite-amplitude samples into a single plateau.

The detector stores the flagged epoch indices and detection parameters in the recording metadata. A separate `clipping_qc.json` file provides the recording-level quality-control report.

**Quality-control results.**

| Recording | Flagged epochs | Total epochs | Flagged (%) |
|---|---:|---:|---:|
| SC4001E0 | 2 | 2,650 | 0.075 |
| SC4002E0 | 0 | 2,829 | 0.000 |
| SC4011E0 | 3 | 2,802 | 0.107 |
| SC4012E0 | 29 | 2,848 | 1.018 |
| SC4021E0 | 0 | 2,804 | 0.000 |
| SC4022E0 | 23 | 2,755 | 0.835 |
| **Total** | **57** | **16,688** | **0.342** |

The longest observed extreme-value plateaus were 340 ms in SC4012E0 and 420 ms in SC4022E0. Most flagged epochs were concentrated in these two recordings.

**Controlled comparison with P0.**

Both experiments used the same six recordings, 11 supplied features, Random Forest classifier and three subject-wise LOSO folds.

| Metric | P0 | P5 |
|---|---:|---:|
| Mean subject-wise Cohen's kappa | 0.704 | 0.704 |
| SD of subject-wise kappa | 0.087 | 0.087 |
| Subject-wise kappa range | 0.627–0.799 | 0.627–0.799 |
| Accuracy | 0.849 | 0.849 |
| Pooled Cohen's kappa | 0.710 | 0.710 |
| Macro-F1 | 0.648 | 0.648 |
| Balanced accuracy | 0.650 | 0.650 |

A direct comparison of the P0 and P5 prediction arrays confirmed that all 16,688 predictions were identical, with zero differences.

**Validation and leakage considerations.**

The independent P5 validation script confirmed that EEG, EOG, EMG, labels and epoch alignment were preserved. The detector does not use sleep-stage labels, statistics pooled across subjects, or classifier predictions.

Recording extrema are calculated independently for each complete night. This is a recording-local, label-free operation suitable for the retrospective full-recording analysis considered here. However, it uses the complete test recording's unlabelled signal distribution and is therefore not directly applicable to real-time epoch-by-epoch inference.

The detector was designed after exploratory inspection of the current three-subject development cohort. Its parameters and performance should not be presented as independently validated on unseen subjects. The rule must be frozen before final evaluation, and any use of full-recording test statistics must remain consistent with the declared evaluation protocol.

**Decision after evaluation.**

Retain P5 as a signal-quality monitoring safeguard, not as a performance-enhancing denoising method. Flagging suspected clipping preserves the original physiological signals and provides traceable warnings without introducing potentially misleading interpolation or reconstruction.

P5 does not improve classification performance because its flags are stored as metadata and are not used as classifier features. All epochs remain in the evaluation.

The current detector identifies repeated extrema within individual epochs and does not yet merge plateaus crossing epoch boundaries. It also cannot independently confirm hardware saturation. These limitations should be considered before extending the method to additional recordings.

**Reproducibility.**

Command:
`python scripts/run_baseline.py --preprocess denoise_clipping`

Output directory:
`results/real__P5_20261008T100803267247Z/`

Artifacts:
- `report.json`
- `provenance.json`
- `clipping_qc.json`

Independent validation:
`python scripts/test_p5_flags.py`

P0/P5 prediction comparison:
16,688 identical predictions; 0 differences.



## ⚠️ Before you fill in many rows — the garden of forking paths

The harness guarantees that **no single run** leaks a subject. It cannot guarantee anything
about the *sequence* of runs. Every row in the table above is scored on the **same**
group-aware folds, and those are the same folds that produce the number you will report. Try
twelve configurations against one held-out set and report the best, and the winner was chosen
partly because it happens to suit *these particular subjects*. Its margin over the runner-up is
optimistically biased, and nothing in the report will look wrong. This is a **selection effect**,
not a leak — which is exactly why the leakage guard cannot catch it, and why it has to be
handled by how you *plan* the work rather than by the code.

It bites hardest here because the cohorts are small: a dozen comparisons on five or six subjects
is a great deal of selection pressure on very little data.

Two defensible ways to handle it. **Say in the report which one you used** — Criterion 7 grades
the quality of the comparison, and this is part of that quality:

| approach | what you do | what it costs |
|---|---|---|
| **an explicit development set** | hold out a few groups *up front* as a DEV set; compare every design option on DEV only; run the chosen pipeline **once** on the untouched evaluation folds and report that | fewer groups serving each purpose — painful on these cohorts — and a noisier DEV estimate |
| **a comparison budget** | decide in advance how many configurations you will compare (a handful, not a sweep), write them into this file **before** you run them, and report the count | you may miss a better option, and you have to resist re-opening the budget once you have seen the numbers — which is the whole discipline |

Either way, two habits that cost nothing:

- **Record the count.** "We compared 6 configurations" belongs in the report. A team that
  compares six and says so is doing better science than one that compares twenty and shows a
  longer table.
- **Read a small margin as no margin.** If the difference between two options is smaller than
  the spread across your subjects/folds, it is not a result. Say "no measurable difference; we
  kept the simpler one" — that sentence earns marks, and claiming the 0.01 does not.

**Related, and equally invisible: *where* your preprocessing is fitted.** The notebook builds
features for every recording **before** `evaluate()` cuts the folds. That is safe for everything
the scaffold ships, because all of it is **stateless** — a fixed filter, or a threshold estimated
from one epoch's own samples. It stops being safe the moment you add a stage that **learns**:
CSP, PCA/ICA, a cohort-wide z-score, a learned artifact template. Fit one of those over all the
recordings and every held-out subject has already shaped the transform that built the training
features — and `assert_no_subject_leak` will still pass, because it checks group ids and cannot
see what a filter was learned from. A learned stage must be fitted **inside** the fold: as a step
in the classifier pipeline (cloned and refit per fold) or via `select_features()`. If you added
one, record here which of the two you used.

## Decision log — the choices behind the numbers

Rows above say *what happened*; this says *what you chose and why*, which is what §16.4 asks you
to make traceable and what the report's defence is built from. There is no single correct
pipeline here — the scaffold deliberately ships options, not answers. One line per decision;
add rows as the pipeline grows, and note the alternative you rejected.

| Pipeline module | Option chosen | Alternative(s) considered | Why this one (one sentence) | Iteration | Revised later? |
|---|---|---|---|---|---|
| 1. Data loading | *e.g.* 8 subjects, both nights | more subjects, one night each | subject-level split needs both nights inside one group | 1 | — |
| 2. Preprocessing — P1 band-pass filtering| EEG 0.5–40 Hz zero-phase Butterworth band-pass | No preprocessing; 50 Hz notch; wavelet denoising | P1 improved mean subject kappa from 0.704 to 0.737 and macro-F1 from 0.648 to 0.683 without increasing between-subject variability; no 50 Hz notch was justified by QC | 2b | No — current choice, subject to later revision |
| 2. Preprocessing — P2 wavelet denoising | Not retained; P1 remains the current preprocessing choice | P2: EEG db4 wavelet denoising with automatic level (max 5), soft thresholding and automatic per-epoch threshold | P2 improved over P0 and achieved a slightly higher mean subject kappa than P1 (0.743 vs 0.737). However, the gain was small (+0.006), while macro-F1 decreased from 0.683 to 0.677, balanced accuracy from 0.680 to 0.666, and N1 recall from 0.416 to 0.315. P1 was therefore retained because its performance was more balanced across sleep stages. | 2c | No |
| 2. Preprocessing — P3 broken-segment handling | Retained as a data-quality safeguard: exclude complete epochs with simultaneous near-flat EEG and EOG | Flag detected epochs without exclusion; no broken-segment handling | QC identified a sustained simultaneous EEG/EOG near-flat event in SC4012E0; the fixed label-independent rule excluded only the three fully affected epochs (2845–2847). Although mean subject kappa decreased from 0.704 to 0.694 vs P0, the rule was retained because objectively broken epochs should not be treated as valid physiological signal and the same predefined rule can handle similar dropout in additional recordings. | 2d | No — retained for data quality, not classification improvement |
| 2. Preprocessing — P5 clipping QC | Retained as a quality-control safeguard: flag suspected EEG clipping without signal modification or epoch exclusion | Exclude flagged epochs; interpolate saturated samples; apply generic denoising; ignore suspected clipping | QC found 57/16,688 epochs with repeated recording-specific EEG extrema. P5 preserved all original data and produced exactly the same 16,688 predictions as P0. Flagging was preferred because the lost signal cannot be reliably reconstructed and exclusion of entire 30-second epochs was not justified by the observed short plateaus. | 2e | No — retained for signal-quality documentation, not performance improvement |

| 3. Feature extraction |  |  |  |  |  |
| 4. Feature selection | *e.g.* `select="none"` | ANOVA `SelectKBest`, tree importances | 14 features vs. ~1 800 epochs — pruning risked more than it saved | 1 | *e.g.* **yes, iter 4** — `select_k=20` was a no-op (harness said so); switched to `k=6` |
| 5. Classification, incl. `imbalance` | *e.g.* `imbalance="balanced"` | `"none"`, `"resample"`, `"threshold"` | *(if you kept the default, say you looked and why — a silent default earns nothing)* |  |  |
| 6. Inference |  |  |  |  |  |
| 7. Reporting |  |  |  |  |  |

## Revisions — the ones that went **backwards** (Criterion 9's actual evidence)

Adding iterations forward is a to-do list. What this section wants is the place a number
downstream sent you back **up** the pipeline. One row is enough; two is a good project.
The notebook's "Decision points on this track" section has a symptom → stage table to
diagnose from.

| # | The downstream result that triggered it | Which earlier decision it indicted | What you changed | What happened to the metric |
|---|---|---|---|---|
| 1 | *e.g.* worst-subject κ 0.14 vs. mean 0.61 | stage 2 — no per-recording normalisation | z-scored band powers within each recording | mean κ 0.61 → 0.59, **worst subject 0.14 → 0.38** — kept |
| 2 |  |  |  |  |

## Who did what

One line per person per iteration — the honest-disclosure requirement of §16.7, and the evidence
`INDIVIDUAL_ASSESSMENT.md` asks for. The seven modules do **not** have to map one-to-one onto
people: one person may own several modules, two people may share one, and ownership may rotate
between iterations. Record what actually happened.

| Iteration | Who | Modules / tasks owned | Reviewed by |
|---|---|---|---|
| 1 | Lili | Ran and verified the supplied baseline on real Sleep-EDF data; inspected LOSO metrics, subject-wise spread and confusion matrix |  |
| 2 | Lili | Signal-quality review; implemented and validated P1 EEG band-pass preprocessing, P2 EEG wavelet denoising and P3 broken-segment detection/exclusion; performed short-window and continuous-duration QC analysis of near-flat segments; ran controlled P0/P1, P0/P2 and P0/P3 LOSO comparisons; analysed subject-wise metrics, stage-wise performance and confusion matrices | |
| 2e | Lili | Designed and implemented P5 suspected EEG clipping detection and recording-level quality flagging; analysed amplitude extrema and plateau durations across six recordings; validated preservation of all signals and labels; exported clipping QC results; performed LOSO evaluation and verified that all 16,688 P5 predictions were identical to P0; documented leakage considerations and limitations | |


## Final numbers (fill in once, at the end)

| | Value | Under what split |
|---|---|---|
| Primary metric, development (mean + spread) |  |  |
| Secondary metrics (macro-F1 / balanced accuracy / per-class recall) |  |  |
| Held-out set — **scored once, never tuned against** |  |  |
| Supplied baseline, same harness |  |  |
| Yardstick from the dataset card (human ceiling / benchmark / chance) |  |  |
| **Configurations compared** before settling (a number) |  | on DEV folds / on the evaluation folds — say which |
| How option comparison was kept separate from final reporting |  | *e.g.* "separate DEV groups" / "budget of 6, fixed in advance" |

*Grading: `CAPSTONE_REPORT_RUBRIC.md` (team, 30 pts) and `INDIVIDUAL_ASSESSMENT.md`
(individual, 10 pts).*
