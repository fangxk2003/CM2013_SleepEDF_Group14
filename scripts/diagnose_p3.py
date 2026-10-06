"""Diagnose near-flat / dropout epochs before defining P3 thresholds."""

import _bootstrap

from pathlib import Path
import numpy as np

from sleepedf import SleepEDFTrack


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "sleep_edf_data"


track = SleepEDFTrack()
recs = track.load(str(CACHE))

channels = ("eeg", "eog", "emg")


# ------------------------------------------------------------
# 1. Collect per-epoch standard deviation
# ------------------------------------------------------------

std_by_channel = {ch: [] for ch in channels}

for rec in recs:
    for ch in channels:
        epoch_std = np.std(rec.epochs[ch], axis=1)
        std_by_channel[ch].append(epoch_std)

for ch in channels:
    std_by_channel[ch] = np.concatenate(std_by_channel[ch])


# ------------------------------------------------------------
# 2. Inspect the lower tail of each distribution
# ------------------------------------------------------------

percentiles = [0, 0.1, 0.5, 1, 2, 5]

print("\n=== Per-channel epoch SD distributions ===")

for ch in channels:
    x = std_by_channel[ch]

    print(f"\n{ch.upper()}")
    for p in percentiles:
        print(f"  p{p:>4}: {np.percentile(x, p):.6f}")


# ------------------------------------------------------------
# 3. Candidate percentile-based definitions
#
# An epoch is considered suspicious only when ALL THREE
# channels fall below their channel-specific percentile.
# ------------------------------------------------------------

candidate_percentiles = [0.1, 0.5, 1, 2, 5]

print("\n=== Multi-channel candidate rules ===")

for p in candidate_percentiles:

    thresholds = {
        ch: np.percentile(std_by_channel[ch], p)
        for ch in channels
    }

    print(f"\n--- Below p{p} in ALL channels ---")
    print(
        "thresholds:",
        ", ".join(
            f"{ch}={thresholds[ch]:.6f}"
            for ch in channels
        ),
    )

    total = 0

    for rec in recs:

        masks = []

        for ch in channels:
            epoch_std = np.std(rec.epochs[ch], axis=1)
            masks.append(epoch_std < thresholds[ch])

        broken = np.logical_and.reduce(masks)
        idx = np.flatnonzero(broken)

        if len(idx):
            print(
                f"  {rec.meta['record']}: "
                f"{len(idx)} epochs -> {idx.tolist()}"
            )

        total += len(idx)

    print(f"TOTAL: {total} / {sum(len(r.labels) for r in recs)} epochs")


# ------------------------------------------------------------
# 4. Inspect the known suspicious SC4012E0 region
# ------------------------------------------------------------

print("\n=== SC4012E0 epochs 2840-2847 ===")

for rec in recs:

    if rec.meta["record"] != "SC4012E0":
        continue

    for i in range(2840, min(2848, len(rec.labels))):

        values = {
            ch: np.std(rec.epochs[ch][i])
            for ch in channels
        }

        print(
            f"epoch {i}: "
            f"label={rec.labels[i]}, "
            f"EEG={values['eeg']:.12e}, "
            f"EOG={values['eog']:.12e}, "
            f"EMG={values['emg']:.12e}"
        )

print("\n=== Exact p0.1 thresholds ===")

for ch in channels:
    threshold = np.percentile(std_by_channel[ch], 0.1)
    print(f"{ch}: {threshold:.12e}")

print("\n=== Pairwise p0.1 rules ===")

p = 0.1

thresholds = {
    ch: np.percentile(std_by_channel[ch], p)
    for ch in channels
}

pairs = [
    ("eeg", "eog"),
    ("eeg", "emg"),
    ("eog", "emg"),
]

for ch1, ch2 in pairs:

    print(f"\n--- {ch1.upper()} + {ch2.upper()} below p{p} ---")

    total = 0

    for rec in recs:

        std1 = np.std(rec.epochs[ch1], axis=1)
        std2 = np.std(rec.epochs[ch2], axis=1)

        broken = (
            (std1 < thresholds[ch1])
            & (std2 < thresholds[ch2])
        )

        idx = np.flatnonzero(broken)

        if len(idx):
            print(
                f"  {rec.meta['record']}: "
                f"{len(idx)} epochs -> {idx.tolist()}"
            )

        total += len(idx)

    print(
        f"TOTAL: {total} / "
        f"{sum(len(r.labels) for r in recs)} epochs"
    )


# ------------------------------------------------------------
# 5. Within-epoch near-flat diagnostic
#
# Search the whole dataset for short near-flat periods that
# could be hidden by a normal SD over the complete 30 s epoch.
#
# IMPORTANT:
# - diagnostic only: this does NOT define P3 yet
# - labels are not used for detection
# - every recording and every epoch is examined
# ------------------------------------------------------------

print("\n=== Within-epoch near-flat diagnostic ===")

window_seconds_list = [1, 2, 5]

# Reuse the p0.1 thresholds obtained from the full-epoch
# distributions. These are diagnostic reference thresholds,
# not yet final P3 thresholds.
thresholds = {
    ch: np.percentile(std_by_channel[ch], 0.1)
    for ch in channels
}

print("\nReference p0.1 thresholds:")
for ch in channels:
    print(f"  {ch}: {thresholds[ch]:.12e}")


for window_seconds in window_seconds_list:

    print(
        f"\n--- Window length = {window_seconds} s "
        f"(EEG + EOG below thresholds) ---"
    )

    total_windows = 0
    total_flagged_windows = 0
    affected_epochs = []

    for rec in recs:

        fs = int(rec.fs)
        window_samples = window_seconds * fs

        eeg = np.asarray(rec.epochs["eeg"])
        eog = np.asarray(rec.epochs["eog"])

        n_epochs, n_samples = eeg.shape

        if n_samples % window_samples != 0:
            raise ValueError(
                f"{rec.meta['record']}: epoch length {n_samples} "
                f"is not divisible by window length {window_samples}"
            )

        n_windows = n_samples // window_samples

        # Shape:
        # (n_epochs, n_windows, samples_per_window)
        eeg_windows = eeg.reshape(
            n_epochs, n_windows, window_samples
        )
        eog_windows = eog.reshape(
            n_epochs, n_windows, window_samples
        )

        # SD of every short window
        eeg_sd = np.std(eeg_windows, axis=2)
        eog_sd = np.std(eog_windows, axis=2)

        # A short window is suspicious when BOTH EEG and EOG
        # are exceptionally flat.
        flat_windows = (
            (eeg_sd < thresholds["eeg"])
            & (eog_sd < thresholds["eog"])
        )

        total_windows += flat_windows.size
        total_flagged_windows += int(flat_windows.sum())

        # Find every epoch containing at least one flat window
        epoch_mask = np.any(flat_windows, axis=1)
        epoch_indices = np.flatnonzero(epoch_mask)

        for epoch_idx in epoch_indices:

            window_indices = np.flatnonzero(
                flat_windows[epoch_idx]
            )

            # Convert local window indices to seconds
            intervals = [
                (
                    int(w * window_seconds),
                    int((w + 1) * window_seconds),
                )
                for w in window_indices
            ]

            affected_epochs.append(
                (
                    rec.meta["record"],
                    int(epoch_idx),
                    intervals,
                )
            )

    print(
        f"Flagged short windows: "
        f"{total_flagged_windows} / {total_windows}"
    )

    print(
        f"Affected 30-s epochs: "
        f"{len(affected_epochs)} / "
        f"{sum(len(r.labels) for r in recs)}"
    )

    if affected_epochs:
        print("Affected epochs and within-epoch intervals:")

        for record, epoch_idx, intervals in affected_epochs:
            print(
                f"  {record} epoch {epoch_idx}: "
                f"{intervals}"
            )
    else:
        print("No affected epochs.")

# ------------------------------------------------------------
# 6. Continuous near-flat runs across epoch boundaries
#
# Diagnostic only.
# Measure how long low-variability periods persist continuously
# through the recording, without resetting at 30-s boundaries.
# ------------------------------------------------------------

print("\n=== Continuous near-flat runs (5-s windows) ===")

window_seconds = 5


def find_runs(mask, window_seconds):
    """
    Return continuous True runs as:
    (start_window, end_window, duration_seconds)

    end_window is inclusive.
    """
    runs = []
    start = None

    for i, value in enumerate(mask):

        if value and start is None:
            start = i

        if start is not None and (
            (not value) or (i == len(mask) - 1)
        ):
            if value and i == len(mask) - 1:
                end = i
            else:
                end = i - 1

            duration = (end - start + 1) * window_seconds

            runs.append((start, end, duration))
            start = None

    return runs


for rec in recs:

    fs = int(rec.fs)
    window_samples = window_seconds * fs

    masks = {}

    # --------------------------------------------------------
    # Build one continuous sequence of 5-s windows per channel
    # --------------------------------------------------------
    for ch in channels:

        # Flatten epochs:
        # (n_epochs, 3000) -> one continuous recording
        signal = np.asarray(rec.epochs[ch]).reshape(-1)

        n_complete = len(signal) // window_samples

        signal = signal[:n_complete * window_samples]

        windows = signal.reshape(
            n_complete,
            window_samples
        )

        window_sd = np.std(windows, axis=1)

        masks[ch] = (
            window_sd < thresholds[ch]
        )

    # Also examine simultaneous EEG + EOG near-flat periods
    masks["eeg+eog"] = (
        masks["eeg"] & masks["eog"]
    )

    print(f"\n--- {rec.meta['record']} ---")

    for name, mask in masks.items():

        runs = find_runs(mask, window_seconds)

        if not runs:
            print(f"{name}: no near-flat runs")
            continue

        # Longest first
        runs = sorted(
            runs,
            key=lambda x: x[2],
            reverse=True
        )

        print(f"{name}: top continuous runs")

        # Only print the 10 longest
        for start_w, end_w, duration in runs[:10]:

            start_sec = start_w * window_seconds
            end_sec = (end_w + 1) * window_seconds

            start_epoch = start_sec // 30
            start_inside = start_sec % 30

            end_epoch = (end_sec - 1) // 30
            end_inside = end_sec - end_epoch * 30

            print(
                f"  {duration:4d} s | "
                f"epoch {start_epoch} + {start_inside:02d}s "
                f"-> epoch {end_epoch} + {end_inside:02d}s"
            )

print("\n=== P3 implementation validation ===")

from sleepedf.preprocessing import BrokenSegmentHandling

p3 = BrokenSegmentHandling(action="exclude")

total_before = 0
total_after = 0

for rec in recs:

    broken = p3.detect(rec)
    processed = p3.transform(rec)

    n_before = len(rec.labels)
    n_after = len(processed.labels)

    total_before += n_before
    total_after += n_after

    idx = np.flatnonzero(broken)

    print(
        f"{rec.meta['record']}: "
        f"before={n_before}, "
        f"broken={len(idx)}, "
        f"indices={idx.tolist()}, "
        f"after={n_after}"
    )

    # Alignment checks
    assert len(processed.epochs["eeg"]) == n_after
    assert len(processed.epochs["eog"]) == n_after
    assert len(processed.epochs["emg"]) == n_after

    # Input must not have been modified
    assert len(rec.labels) == n_before

print(
    f"\nTOTAL: before={total_before}, "
    f"after={total_after}, "
    f"removed={total_before - total_after}"
)