
"""Sensitivity of spindle re-detection to temporal IoU threshold.

Raw detections are a reference, not expert-annotated ground truth.
Detections are recomputed independently for Raw, P1 and P2.
"""

from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import detect_spindle_candidates as detector
from evaluate_spindle_redetection import (
    interval_iou,
    match_candidates,
)
import evaluate_spindle_redetection as redetection


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT / "results" / "spindle_redetection" / "sensitivity"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_THRESHOLDS = np.round(np.arange(0.10, 0.71, 0.10), 2)
REFERENCE_THRESHOLD = 0.30


def detect_all(recording_raw, recording_p1, recording_p2, epoch):
    """Detect events independently using the existing detector."""

    fs = float(recording_raw.fs)

    if recording_raw.labels[epoch] != "N2":
        raise ValueError(f"Epoch {epoch} is not N2")

    recordings = {
        "Raw": recording_raw,
        "P1": recording_p1,
        "P2": recording_p2,
    }

    candidates = {}

    for name, recording in recordings.items():
        signal = np.asarray(
            recording.epochs["eeg"][epoch],
            dtype=float,
        )

        _, envelope = detector.spindle_band_analysis(
            signal, fs
        )

        events, _ = detector.detect_candidates(
            envelope, fs
        )

        candidates[name] = events

    return candidates


def analyze_threshold(epoch, candidates, threshold):
    """Match events using the existing one-to-one algorithm."""

    raw = candidates["Raw"]
    summary_rows = []
    matched_rows = []

    # The original matcher reads its threshold from this module.
    redetection.IOU_THRESHOLD = float(threshold)

    for pipeline in ("P1", "P2"):
        processed = candidates[pipeline]

        matches, missed, unmatched = match_candidates(
            raw, processed
        )

        n_raw = len(raw)
        n_matched = len(matches)

        rate = (
            100.0 * n_matched / n_raw
            if n_raw else np.nan
        )

        summary_rows.append({
            "epoch": epoch,
            "pipeline": pipeline,
            "iou_threshold": threshold,
            "n_raw": n_raw,
            "n_detected": len(processed),
            "n_matched": n_matched,
            "n_missed": len(missed),
            "n_unmatched_processed": len(unmatched),
            "redetection_rate_pct": rate,
            "mean_matched_iou": (
                np.mean([m[2] for m in matches])
                if matches else np.nan
            ),
        })

        for i, j, iou in matches:
            r = raw[i]
            p = processed[j]

            raw_duration = r["end_s"] - r["start_s"]
            processed_duration = p["end_s"] - p["start_s"]

            matched_rows.append({
                "epoch": epoch,
                "pipeline": pipeline,
                "iou_threshold": threshold,
                "raw_candidate_id": i + 1,
                "processed_candidate_id": j + 1,
                "temporal_iou": iou,
                "raw_start_s": r["start_s"],
                "raw_end_s": r["end_s"],
                "processed_start_s": p["start_s"],
                "processed_end_s": p["end_s"],
                "start_shift_s": (
                    p["start_s"] - r["start_s"]
                ),
                "end_shift_s": (
                    p["end_s"] - r["end_s"]
                ),
                "duration_change_s": (
                    processed_duration - raw_duration
                ),
            })

    return summary_rows, matched_rows


def save_csv(filename, rows):
    if rows:
        pd.DataFrame(rows).to_csv(
            OUTPUT_DIR / filename, index=False
        )


def plot_sensitivity(summary):
    fig, ax = plt.subplots(figsize=(9, 5))

    for pipeline in ("P1", "P2"):
        subset = summary[
            summary["pipeline"] == pipeline
        ]

        grouped = subset.groupby(
            "iou_threshold", as_index=False
        )[["n_raw", "n_matched"]].sum()

        grouped["rate"] = (
            100 * grouped["n_matched"] / grouped["n_raw"]
        )

        ax.plot(
            grouped["iou_threshold"],
            grouped["rate"],
            marker="o",
            linewidth=2,
            label=pipeline,
        )

    ax.axvline(
        REFERENCE_THRESHOLD,
        linestyle="--",
        color="black",
        label="Reference IoU = 0.30",
    )

    ax.set(
        xlabel="Minimum temporal IoU",
        ylabel="Redetection rate (%)",
        title="Sensitivity to temporal matching threshold",
        ylim=(-5, 105),
    )

    ax.set_xticks(IOU_THRESHOLDS)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "iou_sensitivity.png", dpi=200
    )
    plt.close(fig)


def plot_boundary_shifts(matches):
    selected = matches[
        np.isclose(
            matches["iou_threshold"],
            REFERENCE_THRESHOLD,
        )
    ]

    if selected.empty:
        return

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4)
    )

    metrics = [
        ("start_shift_s", "Start shift (s)"),
        ("end_shift_s", "End shift (s)"),
        ("duration_change_s", "Duration change (s)"),
    ]

    for ax, (column, label) in zip(axes, metrics):
        values = [
            selected.loc[
                selected["pipeline"] == pipeline,
                column,
            ].dropna().to_numpy()
            for pipeline in ("P1", "P2")
        ]

        ax.boxplot(
            values,
            tick_labels=["P1", "P2"],
        )
        ax.axhline(0, color="black", linestyle="--")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Temporal boundary changes at IoU = 0.30"
    )
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "boundary_shifts.png", dpi=200
    )
    plt.close(fig)


def main():
    print("=" * 65)
    print("CM2013 - SPINDLE MATCHING SENSITIVITY")
    print("=" * 65)

    recording_raw = detector.load_recording()
    recording_p1, recording_p2 = (
        detector.apply_preprocessing(recording_raw)
    )

    all_summary = []
    all_matches = []

    for epoch in detector.EPOCH_INDICES:
        candidates = detect_all(
            recording_raw,
            recording_p1,
            recording_p2,
            epoch,
        )

        print(
            f"Epoch {epoch}: "
            + ", ".join(
                f"{p}={len(candidates[p])}"
                for p in ("Raw", "P1", "P2")
            )
        )

        for threshold in IOU_THRESHOLDS:
            summary, matches = analyze_threshold(
                epoch, candidates, threshold
            )
            all_summary.extend(summary)
            all_matches.extend(matches)

    summary_df = pd.DataFrame(all_summary)
    matches_df = pd.DataFrame(all_matches)

    save_csv("iou_sensitivity.csv", all_summary)
    save_csv("matched_boundary_shifts.csv", all_matches)

    plot_sensitivity(summary_df)

    if not matches_df.empty:
        plot_boundary_shifts(matches_df)

    print("\nOVERALL SENSITIVITY")
    print("-" * 65)

    overall = summary_df.groupby(
        ["iou_threshold", "pipeline"], as_index=False
    )[["n_raw", "n_matched"]].sum()

    overall["redetection_rate_pct"] = (
        100 * overall["n_matched"] / overall["n_raw"]
    )

    print(
        overall.pivot(
            index="iou_threshold",
            columns="pipeline",
            values="redetection_rate_pct",
        ).round(1).to_string()
    )

    print("\nBOUNDARY SHIFTS AT IoU = 0.30")
    print("-" * 65)

    if not matches_df.empty:
        selected = matches_df[
            np.isclose(
                matches_df["iou_threshold"],
                REFERENCE_THRESHOLD,
            )
        ]

        if not selected.empty:
            print(
                selected.groupby("pipeline")[
                    [
                        "start_shift_s",
                        "end_shift_s",
                        "duration_change_s",
                    ]
                ].agg(["mean", "median", "std"]).round(3)
            )

    print("\nResults saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
