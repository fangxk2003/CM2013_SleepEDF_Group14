
from pathlib import Path
import sys

import numpy as np
import matplotlib

# Save figures without opening a GUI window on Windows
matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleepedf.track import SleepEDFTrack


OUTPUT_DIR = ROOT / "results" / "p5_diagnostics" / "morphology"

EVENTS = [
    ("SC4002E0", 2003, 1.780),
    ("SC4012E0", 2751, 19.970),
    ("SC4012E0", 2752, 4.110),
]

WINDOWS = {
    "wide": 0.5,
    "zoom": 0.1,
}


def plot_event(eeg, eog, fs, record, epoch_idx, event_time,
               half_window, window_name):

    n_samples = len(eeg)

    # Convert the event time to a sample position
    center = int(round(event_time * fs))

    # Select the samples surrounding the event
    half_samples = int(round(half_window * fs))

    start = max(0, center - half_samples)
    stop = min(n_samples, center + half_samples + 1)

    time = np.arange(start, stop) / fs

    eeg_segment = eeg[start:stop] * 1e6
    eog_segment = eog[start:stop] * 1e6

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 6),
        sharex=True,
        constrained_layout=True
    )

    axes[0].plot(time, eeg_segment, linewidth=1.2)
    axes[0].axvline(
        event_time,
        color="red",
        linestyle="--",
        label="Detected event"
    )
    axes[0].set_ylabel("EEG (µV)")
    axes[0].set_title("EEG")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, eog_segment, linewidth=1.2)
    axes[1].axvline(
        event_time,
        color="red",
        linestyle="--"
    )
    axes[1].set_ylabel("EOG (µV)")
    axes[1].set_xlabel("Time within epoch (s)")
    axes[1].set_title("EOG")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"{record} | Epoch {epoch_idx} | "
        f"Event at {event_time:.3f} s | {window_name}"
    )

    filename = (
        f"{record}_epoch{epoch_idx}_"
        f"t{event_time:.3f}_{window_name}.png"
    )

    path = OUTPUT_DIR / filename

    fig.savefig(path, dpi=180)
    plt.close(fig)

    print(f"Saved: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    track = SleepEDFTrack()
    recordings = track.load(str(ROOT / "sleep_edf_data"))

    recordings_by_name = {
        rec.meta.get("record", rec.group): rec
        for rec in recordings
    }

    for record, epoch_idx, event_time in EVENTS:
        rec = recordings_by_name[record]

        eeg = np.asarray(rec.epochs["eeg"])[epoch_idx]
        eog = np.asarray(rec.epochs["eog"])[epoch_idx]

        for window_name, half_window in WINDOWS.items():
            plot_event(
                eeg=eeg,
                eog=eog,
                fs=rec.fs,
                record=record,
                epoch_idx=epoch_idx,
                event_time=event_time,
                half_window=half_window,
                window_name=window_name
            )


if __name__ == "__main__":
    main()
