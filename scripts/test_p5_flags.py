
"""Validate P5 suspected-clipping flags without model training."""

import _bootstrap

from pathlib import Path
import numpy as np

from sleepedf import SleepEDFTrack
from sleepedf.preprocessing.experiments import TargetedDenoising


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "sleep_edf_data"

track = SleepEDFTrack()
recs = track.load(str(CACHE))

p5 = TargetedDenoising(
    noise_type="clipping",
    min_plateau_samples=3,
)

print("\n=== P5 CLIPPING FLAG VALIDATION ===")

total_flagged = 0
total_epochs = 0

for rec in recs:
    result = p5.transform(rec)

    record_id = rec.meta.get("record", rec.group)
    flags = result.meta["suspected_clipping_mask"]

    n_flagged = int(np.sum(flags))
    n_epochs = len(rec.labels)

    total_flagged += n_flagged
    total_epochs += n_epochs

    # Alignment and data preservation
    assert len(flags) == n_epochs
    assert np.array_equal(result.labels, rec.labels)

    for ch in rec.epochs:
        assert np.array_equal(
            result.epochs[ch],
            rec.epochs[ch],
        )

    assert result.group == rec.group
    assert result.fs == rec.fs

    assert result is not rec
    assert result.meta is not rec.meta

    print(
        f"{record_id}: "
        f"{n_flagged}/{n_epochs} flagged "
        f"({100 * n_flagged / n_epochs:.3f}%)"
    )

print("\nTOTAL")
print(f"Flagged: {total_flagged}/{total_epochs}")
print(f"Percentage: {100 * total_flagged / total_epochs:.3f}%")
print("PASS: P5 preserves signals, labels and epoch alignment.")
