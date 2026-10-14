# Saved results: where to start

This folder holds the evidence from completed experiments. Use this index to choose runs, [RESULTS.md](../RESULTS.md) to read the team's decisions, and the [project README](../README.md) to run new experiments. Saved folders keep their original names so links in `RESULTS.md` still work.

Each saved run has two files:

- **`report.json` answers “how did it perform?”** It contains pooled metrics, the confusion matrix, subject/fold scores, spread, and aligned predictions and labels.
- **`provenance.json` answers “what produced that result?”** It records preprocessing, model settings, subjects/recordings, input hashes, source hashes, Python/package versions and Git state. Read it before comparing scores.

## Choose a useful starting set

**Recent baseline and P1 candidates:** the two 2026-10-12 runs below are the latest saved P0 (no preprocessing) and P1 (EEG bandpass) random-forest evaluations. Both use subjects 0–2, nights 1 and 2, and Windows/Python 3.11.16. They are a recent comparison pair; their date alone does not establish a final model.

**P0 / P1 / P2 on one recorded environment:** the Mac/Python 3.11.14 runs starting at 11:23:03 and 11:57:27 on 2026-10-08 provide a baseline, bandpass and wavelet set on the same six recordings. The Windows/Python 3.11.17 runs at 11:59:21, 12:08:46 and 12:16:36 provide another set. Compare within a set before comparing hosts.

**Model alternatives:** the six Mac/Python 3.11.14 runs at 19:27–19:36 on 2026-10-08 test logistic regression, random forest, RBF-SVM and XGBoost. The XGBoost runs cover P0, P1 and P2. These remain candidate experiments on only three subjects.

**History and quality checks:** keep the older baseline, environment reproductions and the 20-subject baseline as context. P3 changes the evaluated population by excluding three epochs. P5 adds suspected-clipping flags without changing signals; [p5_diagnostics/](p5_diagnostics/) contains the current signal-inspection figures. These records answer different questions and should not be ranked together as an overall leaderboard.

## All 22 saved real-data runs

The date links open `report.json`; **setup** opens its paired `provenance.json`. Dates are the saved start times in **UTC**. Subject ID 0 means `SC400`, ID 1 means `SC401`, and so on. Scope is **subject IDs / nights**.

The metric is **mean subject-wise Cohen's κ ± sample SD**, rounded to three decimals from `report.spread.cohens_kappa`. It is different from pooled κ, and the SD is not a confidence interval. Every row uses subject-wise leave-one-subject-out evaluation. Mac/Windows labels follow the saved cache paths; exact Python and package versions are in the setup file.

| Started UTC → report | Preprocessing | Model | Subjects / nights | Environment | Mean subject κ ± SD | Provenance |
|---|---|---|---|---|---:|---|
| [2026-10-01 19:39:26](real_baseline_20261001T193926289176Z/report.json) | P0: none | Random forest | 0–2 / 1 | Mac 3.13.3 | 0.618 ± 0.202 | [setup](real_baseline_20261001T193926289176Z/provenance.json) |
| [2026-10-08 08:27:35](real__P0_20261008T082735842383Z/report.json) | P0: none | Random forest | 0–19 / available¹ | Mac 3.13.3 | 0.796 ± 0.081 | [setup](real__P0_20261008T082735842383Z/provenance.json) |
| [2026-10-08 08:56:14](real__P0_20261008T085614831480Z/report.json) | P0: none | Random forest | 0–2 / 1+2 | Mac 3.13.3 | 0.704 ± 0.086 | [setup](real__P0_20261008T085614831480Z/provenance.json) |
| [2026-10-08 08:59:29](real__P0_20261008T085929101187Z/report.json) | P0: none | Random forest | 0–2 / 1 | Mac 3.13.3 | 0.618 ± 0.202 | [setup](real__P0_20261008T085929101187Z/provenance.json) |
| [2026-10-08 09:04:32](real__P1_20261008T090432159744Z/report.json) | P1: bandpass | Random forest | 0–2 / 1+2 | Mac 3.13.3 | 0.737 ± 0.074 | [setup](real__P1_20261008T090432159744Z/provenance.json) |
| [2026-10-08 10:08:03](real__P5_20261008T100803267247Z/report.json) | P5: clipping flags | Random forest | 0–2 / 1+2 | Windows 3.12.15 | 0.704 ± 0.087 | [setup](real__P5_20261008T100803267247Z/provenance.json) |
| [2026-10-08 11:23:03](real__P0_20261008T112303518579Z/report.json) | P0: none | Random forest | 0–2 / 1+2 | Mac 3.11.14 | 0.723 ± 0.115 | [setup](real__P0_20261008T112303518579Z/provenance.json) |
| [2026-10-08 11:57:27](real__P2_20261008T115727461006Z/report.json) | P2: wavelet | Random forest | 0–2 / 1+2 | Mac 3.11.14 | 0.735 ± 0.103 | [setup](real__P2_20261008T115727461006Z/provenance.json) |
| [2026-10-08 11:57:27](real__P1_20261008T115727461008Z/report.json) | P1: bandpass | Random forest | 0–2 / 1+2 | Mac 3.11.14 | 0.741 ± 0.116 | [setup](real__P1_20261008T115727461008Z/provenance.json) |
| [2026-10-08 11:59:21](real__P0_20261008T115921097052Z/report.json) | P0: none | Random forest | 0–2 / 1+2 | Windows 3.11.17 | 0.720 ± 0.119 | [setup](real__P0_20261008T115921097052Z/provenance.json) |
| [2026-10-08 12:08:46](real__P1_20261008T120846779566Z/report.json) | P1: bandpass | Random forest | 0–2 / 1+2 | Windows 3.11.17 | 0.743 ± 0.114 | [setup](real__P1_20261008T120846779566Z/provenance.json) |
| [2026-10-08 12:16:36](real__P2_20261008T121636395830Z/report.json) | P2: wavelet | Random forest | 0–2 / 1+2 | Windows 3.11.17 | 0.736 ± 0.102 | [setup](real__P2_20261008T121636395830Z/provenance.json) |
| [2026-10-08 12:20:06](real__P3_20261008T122006941322Z/report.json) | P3: flat-epoch exclusion | Random forest | 0–2 / 1+2² | Windows 3.11.17 | 0.721 ± 0.125 | [setup](real__P3_20261008T122006941322Z/provenance.json) |
| [2026-10-08 12:23:18](real__P5_20261008T122318721140Z/report.json) | P5: clipping flags | Random forest | 0–2 / 1+2 | Windows 3.11.17 | 0.720 ± 0.119 | [setup](real__P5_20261008T122318721140Z/provenance.json) |
| [2026-10-08 19:27:15](real__P0_logistic_regression_20261008T192715175113Z/report.json) | P0: none | Logistic regression | 0–2 / 1+2 | Mac 3.11.14 | 0.671 ± 0.191 | [setup](real__P0_logistic_regression_20261008T192715175113Z/provenance.json) |
| [2026-10-08 19:32:47](real__P0_random_forest_20261008T193247471859Z/report.json) | P0: none | Random forest | 0–2 / 1+2 | Mac 3.11.14 | 0.723 ± 0.115 | [setup](real__P0_random_forest_20261008T193247471859Z/provenance.json) |
| [2026-10-08 19:33:47](real__P0_rbf_svm_20261008T193347413006Z/report.json) | P0: none | RBF-SVM | 0–2 / 1+2 | Mac 3.11.14 | 0.658 ± 0.124 | [setup](real__P0_rbf_svm_20261008T193347413006Z/provenance.json) |
| [2026-10-08 19:34:33](real__P0_xgboost_20261008T193433876633Z/report.json) | P0: none | XGBoost | 0–2 / 1+2 | Mac 3.11.14 | 0.703 ± 0.097 | [setup](real__P0_xgboost_20261008T193433876633Z/provenance.json) |
| [2026-10-08 19:35:24](real__P1_xgboost_20261008T193524702647Z/report.json) | P1: bandpass | XGBoost | 0–2 / 1+2 | Mac 3.11.14 | 0.714 ± 0.083 | [setup](real__P1_xgboost_20261008T193524702647Z/provenance.json) |
| [2026-10-08 19:36:16](real__P2_xgboost_20261008T193616042677Z/report.json) | P2: wavelet | XGBoost | 0–2 / 1+2 | Mac 3.11.14 | 0.754 ± 0.011 | [setup](real__P2_xgboost_20261008T193616042677Z/provenance.json) |
| [2026-10-12 14:06:03](real__P0_20261012T140603413225Z/report.json) | P0: none | Random forest | 0–2 / 1+2 | Windows 3.11.16 | 0.720 ± 0.119 | [setup](real__P0_20261012T140603413225Z/provenance.json) |
| [2026-10-12 14:29:41](real__P1_20261012T142941417332Z/report.json) | P1: bandpass | Random forest | 0–2 / 1+2 | Windows 3.11.16 | 0.743 ± 0.114 | [setup](real__P1_20261012T142941417332Z/provenance.json) |

¹ Subjects 0–19: 39 recordings, 106,414 evaluated epochs. Night 2 for subject 13 (`SC413`) is absent; this is not a complete 40-recording set.

² P3 evaluates 16,685 epochs after excluding three near-flat epochs. The other subjects 0–2 / nights 1+2 runs evaluate 16,688 epochs. The night-1-only runs evaluate 8,256 epochs. These are distinct evaluation populations even where raw input recordings overlap.

**Environment history matters:** Mac/Python 3.13.3 records NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 and MNE 1.13.2. The early Windows/Python 3.12.15 P5 run records the same first three versions but MNE 1.10.1. The Python 3.11 runs record course pins: NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2 and MNE 1.10.1; the XGBoost runs also record XGBoost 3.2.0. Check source hashes and settings as well as these versions before attributing a difference to preprocessing or a model.

## Plot a comparison

Run commands from the project root. Pass the exact folders you want to compare; no EDF loading or training is needed. For example, compare the recent P0/P1 pair:

```bash
uv run --locked python -B scripts/visualisation_results.py \
  results/real__P0_20261012T140603413225Z \
  results/real__P1_20261012T142941417332Z
```

For the Mac P0/P1/P2 set:

```bash
uv run --locked python -B scripts/visualisation_results.py \
  results/real__P0_20261008T112303518579Z \
  results/real__P1_20261008T115727461008Z \
  results/real__P2_20261008T115727461006Z
```

The default output is [visualisations/index.html](visualisations/index.html), a regenerated dashboard with confusion matrices, stage scores, subject variation and a comparison summary. It displays cohort IDs derived from input hashes, recordings, labels, groups, folds and epoch counts. Matching cohort IDs establish the same evaluated population; they do not establish matching code, model settings or environment. Check the displayed configuration/source/package changes too.

Running without folder arguments scans **all result pairs recursively**, including historical runs and the nested Python comparison captures:

```bash
uv run --locked python -B scripts/visualisation_results.py
```

Use that view for browsing history. For a focused experiment comparison, pass compatible run folders explicitly. Each invocation regenerates its chosen output directory; add `--output-dir <another-directory>` if you want to retain separate views. Raw `report.json` and `provenance.json` files retain their original names and contents.

## Historical Python comparison

[python_patch_comparison_20261008/REPORT.md](python_patch_comparison_20261008/REPORT.md) documents a controlled P1 test of Python 3.11.14 versus 3.11.17 on the same Mac and installed package files. Its five captures reproduced the same report and predictions; the Python patch change alone did not reproduce the historical Mac/Windows difference. The folder remains historical evidence, including arrays, logs and build details. It does not establish equivalence across platforms.
