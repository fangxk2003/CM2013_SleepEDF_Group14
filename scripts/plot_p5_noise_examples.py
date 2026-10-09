from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleepedf.track import SleepEDFTrack


def plot_epoch(recording, epoch_idx, title):
    eeg = np.asarray(recording.epochs["eeg"][epoch_idx])
    eog = np.asarray(recording.epochs["eog"][epoch_idx])

    fs = recording.fs
    time = np.arange(len(eeg)) / fs

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 7),
        sharex=True
    )

    axes[0].plot(time, eeg * 1e6, linewidth=0.8)
    axes[0].set_ylabel("EEG (µV)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)

    axes[1].plot(time, eog * 1e6, linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("EOG (µV)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_dir = ROOT / "results" / "p5_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = title.split(" — ")[0] + "_" + str(epoch_idx) + ".png"

    plt.savefig(output_dir / filename, dpi=150)
    plt.close()

    print(f"Saved: {output_dir / filename}")


def main():
    track = SleepEDFTrack()
    recordings = track.load(str(ROOT / "sleep_edf_data"))

    for rec in recordings:
        name = rec.meta.get("record", rec.group)

        if name != "SC4012E0":
            continue

        for epoch_idx in [2750, 2751, 2752]:
            plot_epoch(
                rec,
                epoch_idx,
                f"{name} — Epoch {epoch_idx} — EEG/EOG inspection"
            )


if __name__ == "__main__":
    main()
