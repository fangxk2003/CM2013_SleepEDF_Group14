"""Evaluate the supplied baseline on local EDFs, without downloading or tuning.

Run: .venv/bin/python -B scripts/run_baseline.py --cache-dir sleep_edf_data
Outputs: a timestamped results directory with report.json and provenance.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import _bootstrap  # Make src/ importable when run directly.

import numpy as np

from sleepedf.machine_learning import default_baseline, evaluate_loso
from sleepedf import SleepEDFTrack


ROOT = Path(__file__).resolve().parents[1]


def json_ready(value):
    """Convert NumPy values and undefined metrics to portable JSON values."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "sleep_edf_data")
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; existing directories are never overwritten.")
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    psgs = sorted(cache.rglob("*-PSG.edf"))
    if not psgs:
        parser.error(f"No PSG EDF files found under {cache}")

    # Match subject AND night, allowing the scorer suffix to differ (E0 vs EC/EH).
    input_files = []
    for psg in psgs:
        matches = sorted(psg.parent.glob(f"{psg.name[:6]}*Hypnogram.edf"))
        if len(matches) != 1:
            parser.error(f"Expected exactly one hypnogram for {psg.name}; found {len(matches)}")
        input_files.extend([psg, matches[0]])
    manifest = [{"path": str(p.relative_to(cache)), "bytes": p.stat().st_size,
                 "sha256": sha256(p)} for p in input_files]
    started = datetime.now(timezone.utc)
    output = (args.output_dir or ROOT / "results" /
              f"real_baseline_{started.strftime('%Y%m%dT%H%M%S%fZ')}").resolve()
    if output.exists():
        parser.error(f"Output directory already exists: {output}")

    cfg = {"seed": 0, "preprocess": "none", "spectral_method": "welch",
           "select": "none"}
    track = SleepEDFTrack()
    print(f"Loading {len(psgs)} PSG/hypnogram pairs from {cache}", flush=True)
    recs = track.load(str(cache))
    loaded = {r.meta["record"] for r in recs}
    expected = {p.name.removesuffix("-PSG.edf") for p in psgs}
    if loaded != expected or len(recs) != len(psgs):
        raise ValueError(f"Loader did not return every recording: expected={expected}, loaded={loaded}")
    subjects = sorted({r.group for r in recs})
    if len(subjects) < 2:
        raise ValueError("LOSO requires at least two different subjects.")
    cfg["loso_max_groups"] = len(subjects)
    for rec in recs:
        if len(rec.labels) == 0:
            raise ValueError(f"No labelled epochs in {rec.meta['record']}")
        if rec.fs != 100 or any(v.shape != (len(rec.labels), 3000)
                                for v in rec.epochs.values()):
            raise ValueError(f"Unexpected sampling rate or epoch shape in {rec.meta['record']}")
        print(f"  {rec.meta['record']}: subject={rec.group}, epochs={len(rec.labels)}", flush=True)

    print("Extracting the 11 supplied features (no optional preprocessing or cropping)...", flush=True)
    X, y, groups = track.build_dataset(recs, cfg)
    if X.shape != (len(y), 11) or len(groups) != len(y) or not np.isfinite(X).all():
        raise ValueError("Invalid feature shape, label alignment, or non-finite features.")
    print(f"Evaluating {len(y)} epochs with {len(subjects)} LOSO folds...", flush=True)
    clf = default_baseline(seed=0, n_estimators=200, imbalance="balanced")
    rep = evaluate_loso(track, X, y, groups, clf=clf, cfg=cfg)
    if len(rep["y_pred"]) != len(y) or np.asarray(rep["confusion"]).sum() != len(y):
        raise ValueError("Evaluation did not account for every epoch.")
    if len(rep["per_fold"]) != len(subjects):
        raise ValueError("Expected one evaluation fold per subject.")

    provenance = {
        "data_kind": "real Sleep-EDF, not synthetic", "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache), "files": manifest, "config": cfg,
        "classifier": {"factory": "sleepedf.machine_learning.default_baseline", "seed": 0,
                       "n_estimators": 200, "imbalance": "balanced"},
        "cropping": "none", "epoch_seconds": 30,
        "fs": 100, "feature_shape": X.shape, "feature_names": track.feature_names(),
        "subjects": subjects,
        "recordings": [{"record": r.meta["record"], "subject": r.group,
                        "epochs": len(r.labels)} for r in recs],
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "mne")},
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(
            list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")))},
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("report.json", rep), ("provenance.json", provenance)):
        (output / name).write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n")
    print(rep["summary"])
    for key in ("accuracy", "cohens_kappa", "macro_f1", "balanced_accuracy"):
        print(f"{key}: {rep[key]}")
    print("Confusion matrix (rows=true, columns=predicted):", rep["labels"])
    print(np.asarray(rep["confusion"]))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
