
"""Evaluate spindle candidate re-detection after preprocessing.

Raw candidates are the reference for temporal matching, not
expert-annotated ground truth.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

import detect_spindle_candidates as detector


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "spindle_redetection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_THRESHOLD = 0.30


def interval_iou(a, b):
    """Temporal intersection over union of two intervals."""
    intersection = max(
        0.0,
        min(a["end_s"], b["end_s"])
        - max(a["start_s"], b["start_s"])
    )

    union = (
        max(a["end_s"], b["end_s"])
        - min(a["start_s"], b["start_s"])
    )

    return intersection / union if union > 0 else 0.0


def match_candidates(reference, detected):
    """One-to-one matching, maximizing the number of valid matches.

    Among matchings with the same number of matches, prefer
    the highest total temporal IoU.
    """
    from scipy.optimize import linear_sum_assignment

    n_ref = len(reference)
    n_det = len(detected)

    if n_ref == 0 or n_det == 0:
        return [], list(range(n_ref)), list(range(n_det))

    ious = np.array([
        [interval_iou(r, d) for d in detected]
        for r in reference
    ])

    valid = ious >= IOU_THRESHOLD

    # Dummy assignments allow unmatched reference events.
    # A valid match always has greater benefit than any
    # possible improvement in summed IoU alone.
    reward = n_ref + 1.0
    scores = np.concatenate(
        [
            np.where(valid, reward + ious, -reward),
            np.zeros((n_ref, n_ref))
        ],
        axis=1
    )

    row_idx, col_idx = linear_sum_assignment(
        scores, maximize=True
    )

    matches = [
        (int(i), int(j), float(ious[i, j]))
        for i, j in zip(row_idx, col_idx)
        if j < n_det and valid[i, j]
    ]

    matched_ref = {i for i, _, _ in matches}
    matched_det = {j for _, j, _ in matches}

    missed = [
        i for i in range(n_ref)
        if i not in matched_ref
    ]

    new = [
        j for j in range(n_det)
        if j not in matched_det
    ]

    return matches, missed, new


def analyze_epoch(recording_raw, recording_p1,
                  recording_p2, epoch_idx):

    fs = float(recording_raw.fs)

    if recording_raw.labels[epoch_idx] != "N2":
        raise ValueError(f"Epoch {epoch_idx} is not N2")

    signals = {
        "Raw": np.asarray(
            recording_raw.epochs["eeg"][epoch_idx],
            dtype=float
        ),
        "P1": np.asarray(
            recording_p1.epochs["eeg"][epoch_idx],
            dtype=float
        ),
        "P2": np.asarray(
            recording_p2.epochs["eeg"][epoch_idx],
            dtype=float
        )
    }

    envelopes = {}
    candidates = {}
    thresholds = {}

    # Independent detection with adaptive thresholds
    for name, signal in signals.items():

        _, envelope = detector.spindle_band_analysis(
            signal, fs
        )

        detected, high_threshold = (
            detector.detect_candidates(envelope, fs)
        )

        envelopes[name] = envelope
        candidates[name] = detected
        thresholds[name] = high_threshold

    reference = candidates["Raw"]

    print(f"\nEPOCH {epoch_idx}")
    print("-" * 55)

    print(
        "Detected candidates: "
        + ", ".join(
            f"{name}={len(candidates[name])}"
            for name in ("Raw", "P1", "P2")
        )
    )

    rows = []
    summary = []

    for name in ("P1", "P2"):

        matches, missed, new = match_candidates(
            reference, candidates[name]
        )

        matched_by_ref = {
            i: (j, iou) for i, j, iou in matches
        }

        for i, raw_candidate in enumerate(reference):

            match = matched_by_ref.get(i)
            j, iou = match if match else (None, np.nan)

            processed_candidate = (
                candidates[name][j]
                if j is not None else None
            )

            rows.append({
                "epoch": epoch_idx,
                "pipeline": name,
                "raw_candidate_id": i + 1,
                "raw_start_s": raw_candidate["start_s"],
                "raw_end_s": raw_candidate["end_s"],
                "redetected": int(j is not None),
                "matched_candidate_id": (
                    j + 1 if j is not None else ""
                ),
                "matched_start_s": (
                    processed_candidate["start_s"]
                    if processed_candidate else ""
                ),
                "matched_end_s": (
                    processed_candidate["end_s"]
                    if processed_candidate else ""
                ),
                "temporal_iou": iou,
            })

        n_ref = len(reference)
        n_matched = len(matches)

        rate = (
            100 * n_matched / n_ref
            if n_ref else np.nan
        )

        summary.append({
            "epoch": epoch_idx,
            "pipeline": name,
            "n_raw": n_ref,
            "n_detected": len(candidates[name]),
            "n_matched": n_matched,
            "n_missed": len(missed),
            "n_unmatched_processed": len(new),
            "redetection_rate_pct": rate,
            "mean_matched_iou": (
                float(np.mean([m[2] for m in matches]))
                if matches else np.nan
            ),
            "high_threshold_uv": thresholds[name] * 1e6,
            "raw_high_threshold_uv": (
                thresholds["Raw"] * 1e6
            ),
        })

        print(
            f"{name}: matched={n_matched}/{n_ref}, "
            f"missed={len(missed)}, "
            f"unmatched processed={len(new)}, "
            f"redetection={rate:.1f}%"
        )

    # Visualize all detections over the same 30-s epoch
    fig, axes = plt.subplots(
        3, 1, figsize=(15, 8), sharex=True
    )

    time = np.arange(len(signals["Raw"])) / fs

    colors = {
        "Raw": "black",
        "P1": "tab:blue",
        "P2": "tab:green"
    }

    for ax, name in zip(axes, ("Raw", "P1", "P2")):

        ax.plot(
            time, envelopes[name] * 1e6,
            color=colors[name], linewidth=1
        )

        ax.axhline(
            thresholds[name] * 1e6,
            color="red", linestyle="--",
            label="Adaptive high threshold"
        )

        for candidate in candidates[name]:
            ax.axvspan(
                candidate["start_s"],
                candidate["end_s"],
                color="orange", alpha=0.25
            )

        ax.set_ylabel(f"{name}\nEnvelope (µV)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time within epoch (s)")

    fig.suptitle(
        f"Independent spindle detection | N2 | Epoch {epoch_idx}"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"epoch_{epoch_idx}_redetection.png",
        dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    return rows, summary


def save_csv(filename, rows):
    if not rows:
        return

    with open(
        OUTPUT_DIR / filename,
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    print("=" * 60)
    print("CM2013 - SPINDLE CANDIDATE RE-DETECTION")
    print("=" * 60)

    recording_raw = detector.load_recording()

    recording_p1, recording_p2 = (
        detector.apply_preprocessing(recording_raw)
    )

    all_rows = []
    all_summary = []

    for epoch_idx in detector.EPOCH_INDICES:

        rows, summary = analyze_epoch(
            recording_raw,
            recording_p1,
            recording_p2,
            epoch_idx
        )

        all_rows.extend(rows)
        all_summary.extend(summary)

    save_csv("redetection_events.csv", all_rows)
    save_csv("redetection_summary.csv", all_summary)

    print("\nOVERALL RESULTS")
    print("-" * 60)

    for name in ("P1", "P2"):

        selected = [
            r for r in all_summary
            if r["pipeline"] == name
        ]

        n_raw = sum(r["n_raw"] for r in selected)
        n_matched = sum(r["n_matched"] for r in selected)
        n_missed = sum(r["n_missed"] for r in selected)
        n_new = sum(
            r["n_unmatched_processed"]
            for r in selected
        )

        rate = (
            100 * n_matched / n_raw
            if n_raw else np.nan
        )

        print(
            f"{name}: {n_matched}/{n_raw} matched "
            f"({rate:.1f}%), "
            f"{n_missed} missed, "
            f"{n_new} unmatched processed"
        )

    print("\nResults saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
