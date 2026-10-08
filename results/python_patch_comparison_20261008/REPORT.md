# P1 Python patch-version comparison — 2026-10-08

Changing Python 3.11.14 to 3.11.17 on this Mac produced **no differences** in the P1 experiment. Both versions exactly reproduced historical run 08. This test did not reproduce the 60 changed predictions observed between historical macOS run 08 and Windows run 10.

## Controlled experiment

Both Python versions were built from official CPython source on the same arm64 Mac, using Clang 21.0.0, the same configure options and effective optimization flags. The interpreters remained in a temporary directory; the project's existing environment was not upgraded or replaced.

Both interpreters loaded the exact same installed package files from the existing `.venv`. Versions were NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2 and MNE 1.10.1. Captures record all installed versions, imported module paths and hashes, numerical-library configuration, thread pools and build settings. Runs used isolated imports and one numerical-library thread.

P1 used the existing `scripts/run_baseline.py` without source changes: subjects SC400, SC401 and SC402, both nights, no cropping, 0.5–40 Hz EEG band-pass, Welch features, 11 features, three subject-wise LOSO folds, a 200-tree balanced Random Forest and seed 0. Input-file hashes, source hashes, labels, groups and fold assignments matched across runs.

One existing Homebrew Python 3.11.14 run served as an anchor. Each newly built Python version ran twice, in alternating order. The anchor matched the complete historical run 08 report exactly, confirming that capture instrumentation and thread settings preserved the original result.

## Results

| Check | Python 3.11.14 vs 3.11.17 | Repeats of each version |
|---|---|---|
| Feature matrix, 16,688 × 11 | Bitwise identical | Bitwise identical |
| Maximum absolute feature difference | 0.0 | 0.0 |
| Scalers and all 600 trained trees | Identical signatures | Identical signatures |
| Predictions changed | 0 / 16,688 | 0 / 16,688 |
| Complete evaluation report | Identical | Identical |

All five captures, including the Homebrew anchor, had the following metrics. Pooled values below use the original harness's three-decimal reporting precision.

| Metric | All captures |
|---|---:|
| Mean subject-wise Cohen's kappa | 0.741 |
| SD of subject-wise kappa | 0.116 |
| Pooled Cohen's kappa | 0.750 |
| Pooled macro F1 | 0.651 |
| Pooled balanced accuracy | 0.615 |
| Correct predictions | 14,670 / 16,688 |

## Interpretation and limits

The Python patch change alone did not alter features, trained trees or predictions in this controlled Mac experiment. The historical run 08 versus run 10 comparison also changed operating system, CPU/platform and compiled numerical builds. Those differences remain plausible explanations for the historical drift; this test does not identify which Windows computation differs or exclude a Windows-specific interaction with Python.

No Windows feature matrix or trained forest was available locally. Locating the original drift would require a capture on that machine and comparison of the exported features and fitted-tree signatures.

## Saved evidence and reproduction

- `comparison.json`: complete equality checks and historical comparisons.
- `runs/`: feature, label and group arrays, reports, provenance, environment details and fitted-tree signatures for all five captures.
- `logs/`: complete output from each P1 run.
- `builds/`: source hashes, interpreter build details and configure/build logs.
- `capture_run.py`: observational wrapper around the unchanged P1 harness.
- `run_comparison.py` and `compare_outputs.py`: runner and comparison scripts.

From the repository root, while the temporary interpreters remain available:

```bash
.venv/bin/python -B results/python_patch_comparison_20261008/run_comparison.py \
  --python-314 /private/tmp/sleepedf-python-patch-20261008/Python-3.11.14/python.exe \
  --python-317 /private/tmp/sleepedf-python-patch-20261008/Python-3.11.17/python.exe \
  --resume
.venv/bin/python -B results/python_patch_comparison_20261008/compare_outputs.py
```

`--resume` reuses completed captures. To perform fresh runs, copy the three scripts to a new sibling directory under `results/`, then run the copied runner without `--resume`. It refuses to overwrite existing run directories.

Official source archives were downloaded from [Python 3.11.14](https://www.python.org/ftp/python/3.11.14/Python-3.11.14.tar.xz) and [Python 3.11.17](https://www.python.org/ftp/python/3.11.17/Python-3.11.17.tar.xz). Both used `CC=/usr/bin/clang`, `CFLAGS=-O2`, `--without-ensurepip`, `--with-openssl=/opt/homebrew/opt/openssl@3` and a version-specific temporary prefix, followed by `make -j4`. Version 3.11.17 repeats a trailing `-O2` in its recorded flags; the effective settings are identical.

Source archive SHA-256:

```text
3.11.14  8d3ed8ec5c88c1c95f5e558612a725450d2452813ddad5e58fdb1a53b1209b78
3.11.17  bfb74ad39efae27cda510f134ab408e00f9992c56851cfc0b1cdb5646da11599
```
