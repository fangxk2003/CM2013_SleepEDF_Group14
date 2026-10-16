"""Diagnose recording-level signal scale before defining P4 normalisation."""

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
# 1. Recording-level signal scale
# ------------------------------------------------------------

print("\n=== RECORDING-LEVEL SIGNAL SCALE ===")

for rec in recs:
    record_id = rec.meta.get("record", rec.group)

    print(f"\n{record_id} | subject={rec.group}")

    for ch in channels:
        x = np.asarray(rec.epochs[ch])

        # All samples from all epochs of this recording
        flat = x.reshape(-1)

        global_mean = np.mean(flat)
        global_std = np.std(flat)

        # Variability of each 30-s epoch
        epoch_std = np.std(x, axis=1)
        median_epoch_std = np.median(epoch_std)

        print(
            f"  {ch.upper():3s}: "
            f"mean={global_mean:.6e}, "
            f"global SD={global_std:.6e}, "
            f"median epoch SD={median_epoch_std:.6e}"
        )


# ------------------------------------------------------------
# 2. Stage-specific scale inside each recording
# ------------------------------------------------------------

print("\n\n=== MEDIAN EPOCH SD BY STAGE AND RECORDING ===")

stages = ("W", "N1", "N2", "N3", "REM")

for ch in channels:
    print(f"\n--- {ch.upper()} ---")

    header = f"{'record':12s}" + "".join(f"{stage:>14s}" for stage in stages)
    print(header)

    for rec in recs:
        record_id = rec.meta.get("record", rec.group)
        labels = np.asarray(rec.labels)
        x = np.asarray(rec.epochs[ch])

        epoch_std = np.std(x, axis=1)

        row = f"{record_id:12s}"

        for stage in stages:
            mask = labels == stage

            if np.any(mask):
                value = np.median(epoch_std[mask])
                row += f"{value:14.6e}"
            else:
                row += f"{'NA':>14s}"

        print(row)


# ------------------------------------------------------------
# 3. Relative stage scale within each recording
# ------------------------------------------------------------

print("\n\n=== STAGE SD RELATIVE TO RECORDING MEDIAN ===")

for ch in channels:
    print(f"\n--- {ch.upper()} ---")

    header = f"{'record':12s}" + "".join(f"{stage:>10s}" for stage in stages)
    print(header)

    for rec in recs:
        record_id = rec.meta.get("record", rec.group)
        labels = np.asarray(rec.labels)
        x = np.asarray(rec.epochs[ch])

        epoch_std = np.std(x, axis=1)
        recording_median = np.median(epoch_std)

        row = f"{record_id:12s}"

        for stage in stages:
            mask = labels == stage

            if np.any(mask):
                stage_median = np.median(epoch_std[mask])
                ratio = stage_median / recording_median
                row += f"{ratio:10.3f}"
            else:
                row += f"{'NA':>10s}"

        print(row)