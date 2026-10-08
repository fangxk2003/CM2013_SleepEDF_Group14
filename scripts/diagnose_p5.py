
"""Inspect high-amplitude EEG transients before defining P5."""

import _bootstrap

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sleepedf import SleepEDFTrack


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "sleep_edf_data"
OUTPUT = ROOT / "results" / "p5_diagnostics"

OUTPUT.mkdir(parents=True, exist_ok=True)

track = SleepEDFTrack()
recs = track.load(str(CACHE))

N_TOP = 12

# ------------------------------------------------------------
# 1. Compute EEG peak-to-peak and SD for every epoch
# ------------------------------------------------------------

candidates = []

for rec in recs:
    record_id = rec.meta.get("record", rec.group)

    eeg = np.asarray(rec.epochs["eeg"])

    ptp = np.ptp(eeg, axis=1)
    std = np.std(eeg, axis=1)

    for epoch_idx in range(len(eeg)):
        candidates.append({
            "record": record_id,
            "recording": rec,
            "epoch_idx": epoch_idx,
            "ptp": float(ptp[epoch_idx]),
            "std": float(std[epoch_idx]),
        })


# ------------------------------------------------------------
# 2. Rank candidates by peak-to-peak amplitude
# ------------------------------------------------------------

candidates.sort(
    key=lambda item: item["ptp"],
    reverse=True,
)

print("\n=== TOP EEG PEAK-TO-PEAK EPOCHS ===")

for rank, item in enumerate(candidates[:N_TOP], start=1):
    print(
        f"{rank:2d}. "
        f"{item['record']} "
        f"epoch={item['epoch_idx']:4d} "
        f"PTP={item['ptp'] * 1e6:8.2f} µV "
        f"SD={item['std'] * 1e6:8.2f} µV"
    )


# ------------------------------------------------------------
# 3. Plot EEG, EOG and EMG for the selected epochs
# ------------------------------------------------------------

for rank, item in enumerate(candidates[:N_TOP], start=1):

    rec = item["recording"]
    idx = item["epoch_idx"]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(13, 8),
        sharex=True,
    )

    for ax, ch in zip(axes, ("eeg", "eog", "emg")):
        signal = np.asarray(rec.epochs[ch][idx])
        time = np.arange(len(signal)) / rec.fs

        ax.plot(time, signal * 1e6, linewidth=0.8)
        ax.set_ylabel(f"{ch.upper()} (µV)")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time within epoch (s)")

    fig.suptitle(
        f"{item['record']} | epoch {idx} | "
        f"EEG PTP = {item['ptp'] * 1e6:.1f} µV"
    )

    fig.tight_layout()

    filename = (
        f"{rank:02d}_{item['record']}_epoch_{idx}.png"
    )

    fig.savefig(
        OUTPUT / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


print(f"\nSaved {N_TOP} diagnostic figures to: {OUTPUT}")



# ------------------------------------------------------------
# 4. Investigate repeated EEG extrema and possible clipping
# ------------------------------------------------------------

print("\n=== EEG EXTREMA AND POSSIBLE CLIPPING ===")

for rec in recs:
    record_id = rec.meta.get("record", rec.group)
    eeg = np.asarray(rec.epochs["eeg"])

    epoch_min = np.min(eeg, axis=1)
    epoch_max = np.max(eeg, axis=1)
    epoch_ptp = epoch_max - epoch_min

    print(f"\n{record_id}")
    print(f"  Global minimum: {np.min(eeg) * 1e6:.6f} µV")
    print(f"  Global maximum: {np.max(eeg) * 1e6:.6f} µV")

    # Number of epochs with peak-to-peak approximately 383 µV
    near_383 = np.isclose(
        epoch_ptp * 1e6,
        383.0,
        atol=0.01,
        rtol=0,
    )

    print(f"  Epochs with PTP ≈ 383 µV: {np.sum(near_383)}")

    # Inspect repeated sample values near the extremes
    flat = eeg.ravel()
    values, counts = np.unique(flat, return_counts=True)

    extreme = np.abs(values * 1e6) >= 180

    extreme_values = values[extreme]
    extreme_counts = counts[extreme]

    order = np.argsort(extreme_counts)[::-1][:10]

    print("  Most frequent extreme EEG values:")

    for idx in order:
        print(
            f"    {extreme_values[idx] * 1e6:12.6f} µV"
            f" | count={extreme_counts[idx]}"
        )

    # Exact consecutive repeated samples
    equal_adjacent = eeg[:, 1:] == eeg[:, :-1]

    print(
        "  Fraction of adjacent identical samples:",
        f"{np.mean(equal_adjacent):.6f}",
    )

# ------------------------------------------------------------
# 5. Quantify repeated extreme values and plateau durations
# ------------------------------------------------------------

print("\n=== CLIPPING PLATEAU ANALYSIS ===")

for rec in recs:
    record_id = rec.meta.get("record", rec.group)
    eeg = np.asarray(rec.epochs["eeg"])

    # Recording-specific observed extrema
    lower = np.min(eeg)
    upper = np.max(eeg)

    # Samples exactly at either extreme
    at_limit = (eeg == lower) | (eeg == upper)

    # Epochs with at least one extreme sample
    affected_epochs = np.any(at_limit, axis=1)

    # Find consecutive runs of extreme samples
    plateau_lengths = []
    plateau_epochs = []

    for epoch_idx, mask in enumerate(at_limit):
        padded = np.concatenate(([False], mask, [False]))
        changes = np.diff(padded.astype(int))

        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        lengths = ends - starts

        # At least 3 consecutive samples at a limit
        qualifying = lengths[lengths >= 3]

        if len(qualifying) > 0:
            plateau_epochs.append(epoch_idx)
            plateau_lengths.extend(qualifying.tolist())

    n_epochs = len(eeg)
    n_affected = np.sum(affected_epochs)
    n_plateau = len(plateau_epochs)

    print(f"\n{record_id}")
    print(f"  Lower extreme: {lower * 1e6:.3f} µV")
    print(f"  Upper extreme: {upper * 1e6:.3f} µV")

    print(
        f"  Epochs touching an extreme: "
        f"{n_affected}/{n_epochs} "
        f"({100 * n_affected / n_epochs:.3f}%)"
    )

    print(
        f"  Epochs with >=3 consecutive extreme samples: "
        f"{n_plateau}/{n_epochs} "
        f"({100 * n_plateau / n_epochs:.3f}%)"
    )

    if plateau_lengths:
        print(
            "  Longest plateau:",
            f"{max(plateau_lengths)} samples "
            f"({1000 * max(plateau_lengths) / rec.fs:.1f} ms)"
        )
    else:
        print("  No qualifying plateau detected")
