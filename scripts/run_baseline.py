"""Evaluate the supplied baseline on local EDFs, without downloading or tuning.

Run: .venv/bin/python -B scripts/run_baseline.py --cache-dir sleep_edf_data
Subset: .venv/bin/python -B scripts/run_baseline.py --n-subjects 3 --nights 1 2
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
import re
import subprocess

import _bootstrap  # Make src/ importable when run directly.

import numpy as np

from sleepedf.machine_learning import default_baseline, evaluate_loso
from sleepedf import SleepEDFTrack
from sleepedf.preprocessing.experiments import TargetedDenoising


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


def subject_id(value):
    subject = int(value)
    if not 0 <= subject <= 82:
        raise argparse.ArgumentTypeError("subject IDs must be between 0 and 82")
    return subject


def subject_count(value):
    count = int(value)
    if count < 2:
        raise argparse.ArgumentTypeError("LOSO requires at least 2 subjects")
    return count


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "sleep_edf_data")
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; existing directories are never overwritten.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--n-subjects", type=subject_count, metavar="N",
        help="Use the first N cached Sleep-Cassette subjects in ascending ID order (N >= 2).",
    )
    selection.add_argument(
        "--subjects", type=subject_id, nargs="+", metavar="ID",
        help="Use these Sleep-Cassette subject IDs, e.g. 0 1 2 (default: all cached subjects).",
    )
    parser.add_argument(
        "--nights", type=int, nargs="+", choices=[1, 2],
        help="Use these night indices, e.g. 1 or 1 2 (default: all available nights).",
    )
    parser.add_argument(
        "--preprocess",
        choices=["none","bandpass", "wavelet","broken_segments","denoise_clipping"],
        default="none",
        help="Preprocessing experiment: none=P0, bandpass=P1 EEG 0.5-40Hz, wavelet=P2 EEG wavelet denoising, broken_segments=P3 exclude objectively identified near-flat epochs (EEG+EOG), denoise_clipping=P5 flag suspected EEG clipping.",
    )
    return parser


def select_psgs(psgs, *, subjects=None, n_subjects=None, nights=None):
    """Select Sleep-Cassette files before loading, requiring requested pairs."""
    psgs = sorted(psgs)
    if subjects is None and n_subjects is None and nights is None:
        return psgs

    indexed = []
    for psg in psgs:
        match = re.fullmatch(r"SC4(\d{2})([12]).*-PSG\.edf", psg.name)
        if match:
            indexed.append((psg, int(match[1]), int(match[2])))
    available = sorted({subject for _, subject, _ in indexed})
    if not available:
        raise ValueError("No Sleep-Cassette PSG files found for subject/night selection.")
    if n_subjects is not None:
        if n_subjects < 2:
            raise ValueError("LOSO requires at least two different subjects.")
        if n_subjects > len(available):
            raise ValueError(
                f"Requested {n_subjects} subjects, but only {len(available)} are cached."
            )
        requested = set(available[:n_subjects])
    elif subjects is not None:
        requested = set(subjects)
        missing = sorted(requested - set(available))
        if missing:
            raise ValueError(f"Requested subject IDs are not cached: {missing}")
    else:
        requested = set(available)

    if nights is not None:
        pairs = {(subject, night) for _, subject, night in indexed}
        missing = sorted((subject, night) for subject in requested for night in set(nights)
                         if (subject, night) not in pairs)
        if missing:
            details = ", ".join(f"subject {subject}, night {night}" for subject, night in missing)
            raise ValueError(f"Requested recordings are not cached: {details}")
    selected = [psg for psg, subject, night in indexed
                if subject in requested and (nights is None or night in nights)]
    if len({psg.name[:5] for psg in selected}) < 2:
        raise ValueError("LOSO requires at least two different subjects.")
    return selected


def main():
    parser = build_parser()
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    psgs = sorted(cache.rglob("*-PSG.edf"))
    if not psgs:
        parser.error(f"No PSG EDF files found under {cache}")
    try:
        psgs = select_psgs(psgs, subjects=args.subjects, n_subjects=args.n_subjects,
                          nights=args.nights)
    except ValueError as error:
        parser.error(str(error))

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
    experiment = {
        "none": "P0",
        "bandpass": "P1",
        "wavelet": "P2",
        "broken_segments": "P3",
        "denoise_clipping": "P5",
    }[args.preprocess]
    output = (args.output_dir or ROOT / "results" /
              f"real__{experiment}_{started.strftime('%Y%m%dT%H%M%S%fZ')}").resolve()
    if output.exists():
        parser.error(f"Output directory already exists: {output}")

    cfg = {"seed": 0, "preprocess": args.preprocess, "spectral_method": "welch",
           "select": "none"}
    track = SleepEDFTrack()
    print(f"Loading {len(psgs)} PSG/hypnogram pairs from {cache}", flush=True)
    recs = track.load(str(cache), psg_paths=psgs)
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

    clipping_qc = None

    if args.preprocess == "denoise_clipping":
        detector = TargetedDenoising(
            noise_type="clipping",
            min_plateau_samples=3,
        )

        clipping_qc = {
            "method": "recording_extrema_plateau",
            "channel": "eeg",
            "action": "flag_only",
            "confirmed_clipping": False,
            "min_plateau_samples": 3,
            "threshold_duration_ms": 30.0,
            "extrema_scope": "individual_full_recording",
            "recordings": [],
        }

        for rec in recs:
            processed = detector.transform(rec)
            info = processed.meta["suspected_clipping"]

            clipping_qc["recordings"].append({
                "record": rec.meta["record"],
                "subject": rec.group,
                "n_epochs": len(rec.labels),
                "n_flagged": info["n_flagged"],
                "flagged_epoch_indices": info["flagged_epoch_indices"],
                "lower_extreme_uv": (
                    info["lower_extreme"] * 1e6
                    if info["lower_extreme"] is not None else None
                ),
                "upper_extreme_uv": (
                    info["upper_extreme"] * 1e6
                    if info["upper_extreme"] is not None else None
                ),
            })

        clipping_qc["total_epochs"] = sum(
            item["n_epochs"] for item in clipping_qc["recordings"]
        )

        clipping_qc["total_flagged"] = sum(
            item["n_flagged"] for item in clipping_qc["recordings"]
        )

        print(
            f"P5 QC: {clipping_qc['total_flagged']}/"
            f"{clipping_qc['total_epochs']} epochs flagged.",
            flush=True,
        )


    print(
        f"Extracting the 11 supplied features "
        f"(preprocess={cfg['preprocess']}, no cropping)...",
        flush=True,
    )
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
        "selection": {"subjects": args.subjects, "n_subjects": args.n_subjects,
                      "nights": args.nights},
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

    if clipping_qc is not None:
        (output / "clipping_qc.json").write_text(
            json.dumps(
                json_ready(clipping_qc),
                indent=2,
                allow_nan=False,
            ) + "\n"
        )

    print(rep["summary"])
    for key in ("accuracy", "cohens_kappa", "macro_f1", "balanced_accuracy"):
        print(f"{key}: {rep[key]}")
    print("Confusion matrix (rows=true, columns=predicted):", rep["labels"])
    print(np.asarray(rep["confusion"]))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
