
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleepedf.track import SleepEDFTrack


# Exploratory parameters, not validated artifact thresholds
THRESHOLD_MAD = 8.0
EOG_WINDOW_SECONDS = 0.2
MAX_EVENTS_PER_EPOCH = 20


def robust_threshold(differences):
    """Compute a robust threshold for absolute sample differences."""
    median = np.median(differences)
    mad = np.median(np.abs(differences - median))

    return median + THRESHOLD_MAD * 1.4826 * mad


def detect_events(signal, threshold):
    """Find local maxima of absolute sample-to-sample differences."""
    differences = np.abs(np.diff(signal))

    if len(differences) < 3:
        return np.array([], dtype=int)

    candidates = np.where(
        (differences[1:-1] > differences[:-2]) &
        (differences[1:-1] >= differences[2:]) &
        (differences[1:-1] > threshold)
    )[0] + 1

    return candidates


def analyze_epoch(eeg, eog, fs):
    """Classify EEG transients by nearby EOG derivative activity."""
    eeg_diff = np.abs(np.diff(eeg))
    eog_diff = np.abs(np.diff(eog))

    eeg_threshold = robust_threshold(eeg_diff)
    eog_threshold = robust_threshold(eog_diff)

    eeg_events = detect_events(eeg, eeg_threshold)

    eeg_events = np.sort(eeg_events)

    window = max(1, round(EOG_WINDOW_SECONDS * fs))
    results = []

    for idx in eeg_events:
        start = max(0, idx - window)
        end = min(len(eog_diff), idx + window + 1)

        # Fast EOG activity
        local_eog_max = np.max(eog_diff[start:end])
        fast_eog_event = local_eog_max > eog_threshold

        # EOG peak-to-peak amplitude in the same time window
        eog_segment = eog[start:min(len(eog), end + 1)]
        eog_ptp = np.ptp(eog_segment)

        # Exploratory diagnostic only: do not use an
        # unvalidated amplitude threshold for correction.
        shared = fast_eog_event

        results.append({
            "time_s": (idx + 1) / fs,
            "eeg_step_uv": eeg_diff[idx] * 1e6,
            "eog_step_uv": local_eog_max * 1e6,
            "category": (
                "shared-event candidate"
                if shared
                else "EEG-only candidate"
            ),
            "eog_ptp_uv": eog_ptp * 1e6,
            "fast_eog_event": fast_eog_event,
        })

    return results

def group_events(events, max_gap_s=0.2):
    """
    Group nearby sample-level EEG detections
    into candidate episodes.
    """
    if not events:
        return []

    episodes = []
    current = [events[0]]

    for event in events[1:]:
        gap = event["time_s"] - current[-1]["time_s"]

        if gap <= max_gap_s:
            current.append(event)
        else:
            episodes.append(current)
            current = [event]

    episodes.append(current)

    summary = []

    for group in episodes:
        strongest = max(
            group,
            key=lambda event: event["eeg_step_uv"]
        )

        summary.append({
            "start_s": group[0]["time_s"],
            "end_s": group[-1]["time_s"],
            "n_detections": len(group),
            "max_eeg_step_uv": strongest["eeg_step_uv"],
            "categories": sorted(set(
                event["category"] for event in group
            )),

        })

    return summary

def main():
    track = SleepEDFTrack()
    recordings = track.load(str(ROOT / "sleep_edf_data"))

    examples = {
        "SC4002E0": [2003],
        "SC4012E0": [2750, 2751, 2752],
    }

    print("\n=== P5 EEG/EOG EVENT DIAGNOSTICS ===")

    for rec in recordings:
        name = rec.meta.get("record", rec.group)

        if name not in examples:
            continue

        eeg_epochs = np.asarray(rec.epochs["eeg"])
        eog_epochs = np.asarray(rec.epochs["eog"])
        fs = rec.fs

        for epoch_idx in examples[name]:
            eeg = eeg_epochs[epoch_idx]
            eog = eog_epochs[epoch_idx]

            events = analyze_epoch(eeg, eog, fs)

            print("\nTop 5 EEG events by step amplitude:")

            strongest_events = sorted(
                events,
                key=lambda event: event["eeg_step_uv"],
                reverse=True
            )[:5]

            for event in strongest_events:
                print(
                    f"  t={event['time_s']:.3f}s | "
                    f"EEG step={event['eeg_step_uv']:.1f} uV | "
                    f"EOG step={event['eog_step_uv']:.1f} uV | "
                    f"EOG peak-to-peak={event['eog_ptp_uv']:.1f} uV | "
                    f"{event['category']}"
                )

            episodes = group_events(events)

            print(f"\n{name} — Epoch {epoch_idx}")
            print(f"Detected EEG events: {len(events)}")
            print(f"Grouped candidate into episodes: {len(episodes)}")

            for episode in episodes:
                print(
                    f"  {episode['start_s']:.3f}–"
                    f"{episode['end_s']:.3f}s | "
                    f"{episode['n_detections']} detections | "
                    f"max EEG step="
                    f"{episode['max_eeg_step_uv']:.1f} uV | "
                    f"{', '.join(episode['categories'])}"
                )


if __name__ == "__main__":
    main()
