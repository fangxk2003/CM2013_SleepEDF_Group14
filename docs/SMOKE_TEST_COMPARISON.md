# Smoke-test comparison — 6 October 2026

Compared the attached `it1_track_sleep_edf_smoke.ipynb` with
`scripts/smoke_test.py`, the modular track, and the original local course checkout.
The notebook was inspected as data; its installation/download cells were not run.

## Findings

The visible baseline workflow is equivalent: synthetic recordings from
`track.smoke()`, default feature extraction, and subject-wise evaluation.
The local defaults are five subjects, 80 epochs each, seed 0, medium difficulty,
11 features, no optional preprocessing/selection, and a scaled 200-tree random
forest with balanced class weights and random state 0. The notebook explicitly
reports `USE_REAL=False` and subjects S01–S05.

| Run | NumPy | SciPy | scikit-learn | Kappa | Macro-F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Current project, reproduced | 2.5.3 | 1.18.1 | 1.9.1 | 0.677 | 0.730 | 0.752 |
| Original course code, current environment | 2.5.3 | 1.18.1 | 1.9.1 | 0.677 | 0.730 | 0.752 |
| Change only scikit-learn | 2.5.3 | 1.18.1 | 1.7.2 | 0.616 | 0.686 | 0.713 |
| Course-pinned numerical libraries | 2.2.6 | 1.15.3 | 1.7.2 | 0.616 | 0.686 | 0.713 |
| Additional version check | 2.5.3 | 1.18.1 | 1.6.1 | 0.616 | 0.686 | 0.713 |
| Additional version check | 2.5.3 | 1.18.1 | 1.8.0 | 0.616 | 0.686 | 0.713 |
| Teammate's saved executed baseline cell | Not recorded | Not recorded | Not recorded | 0.605 | 0.680 | 0.705 |

Changing only scikit-learn from 1.9.1 to 1.7.2 changed 37 of 400 predictions.
The feature matrix, labels and subject IDs were bit-identical in this comparison.
The synthetic signal hash also matched across all tested environments. Thus the
0.677 versus 0.616 difference is reproducibly caused by the library version,
not the modular reorganisation or a different random seed.

The installed random-forest source differs in its weighting/bootstrap behaviour:
the older implementation draws bootstrap indices uniformly and applies weights
to the resulting tree samples; the newer implementation uses weights as sampling
probabilities. The baseline uses `class_weight="balanced"`, so this change is
relevant even though our code never explicitly supplies `sample_weight`.
See the upstream [random-forest weighting change, PR #31529](https://github.com/scikit-learn/scikit-learn/pull/31529).
A fixed seed does not make these different implementations equivalent.

## The notebook contains outputs from different runs

Cell indices here are zero-based positions in the notebook JSON:

- Cell 12, execution count 3: kappa 0.605, macro-F1 0.680, accuracy 0.705.
- Cell 19, no execution count: the comparison table's baseline is 0.616.
- Cell 23, no execution count: the honest-split example also reports 0.616.
- Cells 19, 23, 26 and 28 have source and output objects exactly identical to
  the corresponding cells in the original local course notebook.

The latter outputs are retained course outputs, not evidence that the teammate's
current run reproduced those values. Clear outputs and rerun from a fresh kernel
before comparing the baseline with the later experiments.

## What is still unknown

The exact 0.605 result was not reproduced in the tested environments. The
attached notebook does not record the imported source revision, package versions,
feature matrix or prediction arrays for that run. Its setup can clone an
unpinned GitHub default branch and installs packages without version pins. The
Python version in notebook metadata is not a dependable record of the actual
Colab runtime. Source changes, additional dependency/platform differences or
retained kernel state cannot be distinguished from this attachment alone.
Do not claim that a specific scikit-learn version produced the teammate's 0.605
without their runtime information.

## Reproducible comparison

Agree on one source revision and one pinned environment. To match the course
reference, its local `requirements-lock.txt` specifies NumPy 2.2.6, SciPy 1.15.3
and scikit-learn 1.7.2 (plus plotting/optional dependencies). To retain the current
project baseline instead, pin the current versions consistently for everyone.
Restart the kernel after installing packages, clear outputs and run from the top.
Use the same explicit smoke parameters and compare pooled kappa with pooled
kappa, rather than the mean of per-subject kappas in `summary`.

This cell can be run immediately after the teammate's baseline cell to capture
the missing provenance without changing their model:

```python
import hashlib
import inspect
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import sklearn
from bsp import sleep_pipeline, biosignals

modules = [inspect.getmodule(type(track)),
           inspect.getmodule(track.evaluate), sleep_pipeline, biosignals]
sources = {}
for module in modules:
    path = Path(module.__file__).resolve()
    sources[module.__name__] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()

print(json.dumps({
    "python": platform.python_version(),
    "platform": platform.platform(),
    "numpy": np.__version__, "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__,
    "sources": sources,
    "smoke_signature": str(inspect.signature(track.smoke)),
    "config_overrides": track.cfg,
    "model": repr(track.baseline()),
    "shape": list(X.shape), "X_sha256": array_hash(X),
    "labels": np.asarray(y).tolist(), "groups": np.asarray(groups).tolist(),
    "y_true": np.asarray(rep["y_true"]).tolist(),
    "y_pred": np.asarray(rep["y_pred"]).tolist(),
}, indent=2, default=str))
```

For the current project environment, `X_sha256` is
`3f0e74b4a4df8f24bc503717b0cccb6621c6461d109083f9c386dd3e9c2b34c5`.
Exact floating-point hashes can differ across numerical libraries/platforms;
if hashes differ, comparing actual arrays with numerical tolerances is the next
step rather than assuming the datasets are entirely different.

The alternative dependencies were installed only in temporary folders. The
project `.venv`, baseline implementation, existing results and teammate's
notebook were not modified during this investigation.
