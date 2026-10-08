"""Run the existing P1 harness, retaining intermediate arrays for comparison.

Use an isolated interpreter: python -I -S -B capture_run.py --help.
No dependencies are installed and the project's source is not modified.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
import time


def array_digest(array):
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes(order="C"))
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--site-packages", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-python", required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    site_packages = args.site_packages.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if platform.python_version() != args.expected_python:
        parser.error(f"Unexpected Python: {platform.python_version()}")
    # -I -S excludes user packages, environment paths and .pth side effects.
    sys.path[:0] = [str(root / "scripts"), str(root / "src"), str(site_packages)]

    import numpy as np
    import run_baseline
    from sklearn.pipeline import Pipeline
    from threadpoolctl import threadpool_info

    expected_packages = {
        "numpy": "2.2.6", "scipy": "1.15.3",
        "scikit-learn": "1.7.2", "mne": "1.10.1",
    }
    packages = {dist.metadata["Name"]: dist.version
                for dist in metadata.distributions(path=[str(site_packages)])}
    for name, expected in expected_packages.items():
        if metadata.version(name) != expected:
            raise RuntimeError(f"Unexpected {name} version: {metadata.version(name)}")

    module_names = [
        "numpy", "scipy", "sklearn", "mne", "sleepedf",
        "numpy._core._multiarray_umath", "scipy.signal._sigtools",
        "scipy.fft._pocketfft.pypocketfft", "scipy.linalg._fblas",
        "sklearn.tree._tree",
    ]
    modules = {}
    for name in module_names:
        path = Path(importlib.import_module(name).__file__).resolve()
        expected_root = root / "src" if name == "sleepedf" else site_packages
        if not path.is_relative_to(expected_root):
            raise RuntimeError(f"Unexpected module location: {name}: {path}")
        modules[name] = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    config_text = io.StringIO()
    with contextlib.redirect_stdout(config_text):
        np.show_config()
    captured = []
    forests = []
    original_build = run_baseline.SleepEDFTrack.build_dataset
    original_fit = Pipeline.fit

    def capture_dataset(self, *positional, **keywords):
        data = original_build(self, *positional, **keywords)
        captured.append(data)
        return data

    def capture_fit(self, *positional, **keywords):
        result = original_fit(self, *positional, **keywords)
        clf = self.named_steps.get("clf")
        if hasattr(clf, "estimators_"):
            scale = self.named_steps["scale"]
            trees = []
            for estimator in clf.estimators_:
                tree = estimator.tree_
                trees.append({name: array_digest(getattr(tree, name)) for name in (
                    "children_left", "children_right", "feature", "threshold",
                    "impurity", "n_node_samples", "weighted_n_node_samples", "value",
                )})
            forests.append({
                "scaler": {name: array_digest(getattr(scale, name))
                           for name in ("mean_", "var_", "scale_")},
                "trees": trees,
            })
        return result

    run_baseline.SleepEDFTrack.build_dataset = capture_dataset
    Pipeline.fit = capture_fit
    sys.argv = [
        str(root / "scripts" / "run_baseline.py"), "--subjects", "0", "1", "2",
        "--nights", "1", "2", "--preprocess", "bandpass",
        "--cache-dir", str(root / "sleep_edf_data"), "--output-dir", str(output),
    ]
    started = time.monotonic()
    try:
        run_baseline.main()
    finally:
        run_baseline.SleepEDFTrack.build_dataset = original_build
        Pipeline.fit = original_fit
    if len(captured) != 1 or len(forests) != 3:
        raise RuntimeError(f"Unexpected captures: datasets={len(captured)}, folds={len(forests)}")
    X, y, groups = captured[0]
    for name, values in (("features", X), ("labels", y), ("groups", groups)):
        np.save(output / f"{name}.npy", values, allow_pickle=False)
    (output / "forest_signatures.json").write_text(json.dumps(forests, indent=2) + "\n")
    environment = {
        "python": platform.python_version(), "sys_version": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "machine": platform.machine(), "compiler": platform.python_compiler(),
        "config_args": sysconfig.get_config_var("CONFIG_ARGS"),
        "cflags": sysconfig.get_config_var("CFLAGS"),
        "isolated": sys.flags.isolated, "no_site": sys.flags.no_site,
        "sys_path": sys.path, "site_packages": str(site_packages),
        "packages": packages, "modules": modules,
        "numpy_config": config_text.getvalue(), "threadpools": threadpool_info(),
        "thread_settings": {name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        )},
        "runtime_seconds": time.monotonic() - started,
        "array_sha256": {name: array_digest(values)
                         for name, values in (("features", X), ("labels", y), ("groups", groups))},
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print("Captured feature arrays and all three trained forest signatures.", flush=True)


if __name__ == "__main__":
    main()
