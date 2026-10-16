
"""Check spindle candidates near 30-second epoch boundaries.

Uses the existing detector and preprocessing pipelines.
Raw detections are references, not expert annotations.
"""

import numpy as np
import detect_spindle_candidates as detector

EPOCH_DURATION_S = 30.0
EDGE_MARGIN_S = 1.0
TOL = 1e-6


def check_candidates(name, signal, fs):
    _, envelope = detector.spindle_band_analysis(signal, fs)
    candidates, threshold = detector.detect_candidates(
        envelope, fs
    )

    actual_duration = len(signal) / fs

    print(f"\n{name}")
    print(f"  Signal duration: {actual_duration:.3f} s")
    print(f"  Detection threshold: {threshold * 1e6:.3f} uV")
    print(f"  Number of candidates: {len(candidates)}")

    for i, candidate in enumerate(candidates, start=1):
        start = float(candidate["start_s"])
        end = float(candidate["end_s"])

        issues = []

        if start < -TOL:
            issues.append("START OUTSIDE EPOCH")

        if end > actual_duration + TOL:
            issues.append("END OUTSIDE EPOCH")

        if end <= start:
            issues.append("INVALID DURATION")

        if start < EDGE_MARGIN_S:
            issues.append("NEAR START")

        if end > actual_duration - EDGE_MARGIN_S:
            issues.append("NEAR END")

        if abs(start) <= TOL:
            issues.append("TOUCHES START")

        if abs(end - actual_duration) <= 1 / fs + TOL:
            issues.append("TOUCHES END")

        status = ", ".join(issues) if issues else "OK"

        print(
            f"  C{i}: {start:.3f}-{end:.3f} s "
            f"({end - start:.3f} s) | {status}"
        )


def main():
    raw = detector.load_recording()
    p1, p2 = detector.apply_preprocessing(raw)

    recordings = {
        "Raw": raw,
        "P1": p1,
        "P2": p2,
    }

    for epoch_idx in detector.EPOCH_INDICES:
        print("\n" + "=" * 65)
        print(f"EPOCH {epoch_idx}")
        print("=" * 65)

        for name, recording in recordings.items():
            fs = float(recording.fs)
            signal = np.asarray(
                recording.epochs["eeg"][epoch_idx],
                dtype=float,
            )

            check_candidates(name, signal, fs)


if __name__ == "__main__":
    main()
