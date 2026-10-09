
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleepedf.track import SleepEDFTrack


def impulse_metrics(eeg):
    """
    Maximum absolute sample-to-sample difference
    for each EEG epoch.
    """
    differences = np.diff(eeg, axis=1)
    return np.max(np.abs(differences), axis=1)


def baseline_metrics(eeg, fs):
    """
    Range of the mean EEG amplitude across
    consecutive non-overlapping 5-second windows.
    """
    window_samples = int(5 * fs)

    n_epochs, n_samples = eeg.shape
    n_windows = n_samples // window_samples

    trimmed = eeg[:, :n_windows * window_samples]

    windows = trimmed.reshape(
        n_epochs, n_windows, window_samples
    )

    window_means = np.mean(windows, axis=2)

    return np.ptp(window_means, axis=1)


def main():
    track = SleepEDFTrack()
    recordings = track.load(str(ROOT / "sleep_edf_data"))

    print("\n=== P5 ADDITIONAL NOISE DIAGNOSTICS ===")

    for rec in recordings:
        eeg = np.asarray(rec.epochs["eeg"])
        fs = rec.fs

        impulse = impulse_metrics(eeg)
        baseline = baseline_metrics(eeg, fs)

        name = rec.meta.get("record", rec.group)

        print(f"\nRecording: {name}")
        print(f"Epochs: {len(eeg)}")

        print("\nImpulse metric (µV/sample):")
        print(f"  Median: {np.median(impulse)*1e6:.3f}")
        print(f"  95th percentile: {np.percentile(impulse,95)*1e6:.3f}")
        print(f"  Maximum: {np.max(impulse)*1e6:.3f}")

        print("\nBaseline variation (µV):")
        print(f"  Median: {np.median(baseline)*1e6:.3f}")
        print(f"  95th percentile: {np.percentile(baseline,95)*1e6:.3f}")
        print(f"  Maximum: {np.max(baseline)*1e6:.3f}")

        top_impulse = np.argsort(impulse)[-5:][::-1]
        top_baseline = np.argsort(baseline)[-5:][::-1]

        print("\nTop 5 impulse epochs:")
        for idx in top_impulse:
            print(
                f"  Epoch {idx}: "
                f"{impulse[idx]*1e6:.3f} µV/sample"
            )

        print("\nTop 5 baseline-drift candidates:")
        for idx in top_baseline:
            print(
                f"  Epoch {idx}: "
                f"{baseline[idx]*1e6:.3f} µV"
            )


if __name__ == "__main__":
    main()
