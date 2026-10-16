
"""Exploratory sleep-spindle candidate detection for CM2013 Sleep-EDF.

Detect candidates from raw EEG, then compare their preservation in
P1 Butterworth bandpass and P2 wavelet denoising.

Outputs CSV tables and figures. This is not a validated spindle detector.
"""

from pathlib import Path
import csv

import _bootstrap  # Make src/ importable

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    sosfiltfilt,
    hilbert,
    find_peaks,
    welch,
)

from sleepedf import SleepEDFTrack
from sleepedf.preprocessing.experiments import (
    EEGBandpass,
    WaveletDenoising,
)
from scipy.ndimage import binary_closing


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "sleep_edf_data" / "physionet-sleep-data"
OUTPUT_DIR = ROOT / "results" / "spindle_candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PSG_PATH = DATA_DIR / "SC4001E0-PSG.edf"

EPOCH_INDICES = [1025, 1637]

BAND_LOW = 11.0
BAND_HIGH = 16.0

MIN_DURATION = 0.5
MAX_DURATION = 2.0

THRESHOLD_FACTOR = 1.5
SMOOTHING_SECONDS = 0.10
MIN_PEAK_DISTANCE = 0.5

# Exploratory spectral specificity (not a validated spindle criterion).
SPECTRAL_WINDOW_SECONDS = 2.0
SPECTRAL_LOWER_BAND = (6.0, 10.0)
SPECTRAL_SIGMA_BAND = (11.0, 16.0)
SPECTRAL_UPPER_BAND = (17.0, 21.0)

COLORS = {
    "Raw": "black",
    "P1": "tab:blue",
    "P2": "tab:green",
}

def detect_spindle_intervals(
    envelope,
    fs,
    high_factor=1.5,
    low_factor=0.5,
    min_duration=0.5,
    max_duration=2.0,
    max_gap=0.10,
):
    """
    Detect spindle candidates using hysteresis thresholding.

    Returns:
        intervals: list of (start_time, end_time)
        high_threshold: peak detection threshold
        low_threshold: event boundary threshold
    """

    median = np.median(envelope)
    mad = np.median(np.abs(envelope - median))
    robust_sigma = 1.4826 * mad

    high_threshold = median + high_factor * robust_sigma
    low_threshold = median + low_factor * robust_sigma

    # Regions exceeding the lower threshold
    low_mask = envelope > low_threshold

    # Bridge short interruptions
    gap_samples = max(1, int(round(max_gap * fs)))
    closed_mask = binary_closing(
        low_mask,
        structure=np.ones(gap_samples + 1, dtype=bool)
    )

    # Extract contiguous regions
    padded = np.pad(closed_mask.astype(int), (1, 1))
    changes = np.diff(padded)

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    intervals = []

    for start, end in zip(starts, ends):
        duration = (end - start) / fs

        # Require at least one strong peak
        has_strong_peak = np.any(
            envelope[start:end] > high_threshold
        )

        if (
            has_strong_peak
            and min_duration <= duration <= max_duration
        ):
            intervals.append((start / fs, end / fs))

    return intervals, high_threshold, low_threshold


# ============================================================
# 2. LOAD RECORDING
# ============================================================

def load_recording():

    if not PSG_PATH.exists():
        raise FileNotFoundError(
            f"PSG file not found: {PSG_PATH}"
        )

    track = SleepEDFTrack()

    recordings = track.load(
        str(DATA_DIR),
        psg_paths=[PSG_PATH],
    )

    if not recordings:
        raise RuntimeError("No recordings loaded.")

    return recordings[0]


# ============================================================
# 3. APPLY EXISTING PREPROCESSING PIPELINES
# ============================================================

def apply_preprocessing(recording):

    p1 = EEGBandpass(
        low_hz=0.5,
        high_hz=40.0,
        order=4,
    )

    p2 = WaveletDenoising(
        wavelet="db4",
        level=None,
        threshold_mode="soft",
    )

    recording_p1 = p1.transform(recording)
    recording_p2 = p2.transform(recording)

    return recording_p1, recording_p2


# ============================================================
# 4. SPINDLE-BAND SIGNAL AND ENVELOPE
# ============================================================

def spindle_band_analysis(signal, fs):

    sos = butter(
        4,
        [BAND_LOW, BAND_HIGH],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, signal)

    envelope = np.abs(hilbert(filtered))

    # Smooth the envelope without modifying the EEG itself.
    window_samples = max(
        1, int(round(SMOOTHING_SECONDS * fs))
    )

    kernel = (
        np.ones(window_samples) / window_samples
    )

    smoothed_envelope = np.convolve(
        envelope,
        kernel,
        mode="same",
    )

    return filtered, smoothed_envelope


# ============================================================
# 5. DETECT CANDIDATES FROM RAW EEG
# ============================================================

def detect_candidates(envelope, fs):

    intervals, high_threshold, low_threshold = detect_spindle_intervals(
        envelope,
        fs,
        high_factor=1.5,
        low_factor=0.5,
        min_duration=MIN_DURATION,
        max_duration=MAX_DURATION,
        max_gap=0.10,
    )

    candidates = []

    for start_s, end_s in intervals:

        start_idx = int(round(start_s * fs))
        end_idx = int(round(end_s * fs))

        segment = envelope[start_idx:end_idx]

        if len(segment) == 0:
            continue

        peak_idx = start_idx + int(np.argmax(segment))

        candidates.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "peak_idx": peak_idx,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "peak_envelope_uv": envelope[peak_idx] * 1e6,
        })

    return candidates, high_threshold

# ============================================================
# 6. FREQUENCY ESTIMATION
# ============================================================

def dominant_frequency(filtered_segment, fs):

    if len(filtered_segment) < 2:
        return np.nan

    frequencies, psd = welch(
        filtered_segment,
        fs=fs,
        nperseg=len(filtered_segment),
    )

    mask = (
        (frequencies >= BAND_LOW)
        & (frequencies <= BAND_HIGH)
    )

    if not np.any(mask):
        return np.nan

    band_frequencies = frequencies[mask]
    band_psd = psd[mask]

    return float(
        band_frequencies[np.argmax(band_psd)]
    )


# ============================================================
# 6B. SPECTRAL SPECIFICITY ON THE ORIGINAL (NOT SIGMA-FILTERED) EEG
# ============================================================

def spectral_specificity(signal, fs, start_idx, end_idx):
    """Compute band powers and sigma specificity in a 2-s window.

    The window is centered on the candidate and kept inside the epoch.
    Welch PSD uses a Hann window and 2-s segments when available.
    Band powers are estimated by integrating the PSD over frequency.
    """
    n = len(signal)
    target = min(n, max(2, int(round(SPECTRAL_WINDOW_SECONDS * fs))))
    center = (start_idx + end_idx) // 2
    left = max(0, min(center - target // 2, n - target))
    right = left + target
    segment = np.asarray(signal[left:right], dtype=float)

    if len(segment) < 2 or not np.all(np.isfinite(segment)):
        return {"lower_power_uv2": np.nan, "sigma_power_uv2": np.nan,
                "upper_power_uv2": np.nan, "sigma_specificity_ratio": np.nan,
                "spectral_window_start_s": left / fs,
                "spectral_window_end_s": right / fs}

    freqs, psd = welch(segment, fs=fs, window="hann",
                       nperseg=len(segment), detrend="constant",
                       scaling="density")

    def power_in_band(band):
        # Interpolate the PSD at band boundaries to avoid losing edge bins.
        lo, hi = band
        if hi > freqs[-1]:
            return np.nan
        inside = (freqs > lo) & (freqs < hi)
        f = np.concatenate(([lo], freqs[inside], [hi]))
        p = np.interp(f, freqs, psd)
        return float(np.trapezoid(p, f) * 1e12)  # V² -> µV²

    lower = power_in_band(SPECTRAL_LOWER_BAND)
    sigma = power_in_band(SPECTRAL_SIGMA_BAND)
    upper = power_in_band(SPECTRAL_UPPER_BAND)
    denominator = lower + upper
    ratio = sigma / denominator if np.isfinite(denominator) and denominator > 0 else np.nan
    return {"lower_power_uv2": lower, "sigma_power_uv2": sigma,
            "upper_power_uv2": upper, "sigma_specificity_ratio": ratio,
            "spectral_window_start_s": left / fs,
            "spectral_window_end_s": right / fs}


# ============================================================
# 6C. SPECTRAL PRESERVATION (SAME 2-S WINDOW AS SSR)
# ============================================================

def spectral_preservation(signal, fs, start_idx, end_idx):
    """Welch PSD on a fixed, epoch-clipped 2-s window, in µV²/Hz."""
    n = len(signal)
    target = min(n, max(2, int(round(SPECTRAL_WINDOW_SECONDS * fs))))
    center = (start_idx + end_idx) // 2
    left = max(0, min(center - target // 2, n - target))
    segment = np.asarray(signal[left:left + target], dtype=float)
    freqs, psd = welch(segment, fs=fs, window="hann",
                       nperseg=len(segment), detrend="constant",
                       scaling="density")
    psd_uv2_hz = psd * 1e12
    band = (freqs >= BAND_LOW) & (freqs <= BAND_HIGH)
    peak_hz = float(freqs[band][np.argmax(psd_uv2_hz[band])]) if np.any(band) else np.nan
    return freqs, psd_uv2_hz, peak_hz


def plot_candidate_psd(epoch_idx, candidate_id, candidate, signals, fs):
    """Compare broadband PSDs before the dedicated spindle-band filter."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ("Raw", "P1", "P2"):
        freqs, psd, _ = spectral_preservation(
            signals[name], fs, candidate["start_idx"], candidate["end_idx"])
        mask = (freqs >= 0) & (freqs <= 25)
        ax.plot(freqs[mask], 10 * np.log10(np.maximum(psd[mask], 1e-15)),
                label=name, color=COLORS[name])
    ax.axvspan(BAND_LOW, BAND_HIGH, color="orange", alpha=0.15,
               label="Sigma (11–16 Hz)")
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (dB re 1 µV²/Hz)",
           title=f"Welch PSD | Epoch {epoch_idx} | Candidate {candidate_id}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = OUTPUT_DIR / f"epoch_{epoch_idx}_candidate_{candidate_id:02d}_psd.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  PSD figure: {output.name}")


# ============================================================
# 7. COMPARE RAW / P1 / P2
# ============================================================

def analyze_epoch(
    recording_raw,
    recording_p1,
    recording_p2,
    epoch_idx,
):

    fs = float(recording_raw.fs)

    if recording_raw.labels[epoch_idx] != "N2":
        raise ValueError(
            f"Epoch {epoch_idx} is not labelled N2."
        )

    signals = {
        "Raw": np.asarray(
            recording_raw.epochs["eeg"][epoch_idx],
            dtype=float,
        ),
        "P1": np.asarray(
            recording_p1.epochs["eeg"][epoch_idx],
            dtype=float,
        ),
        "P2": np.asarray(
            recording_p2.epochs["eeg"][epoch_idx],
            dtype=float,
        ),
    }

    filtered = {}
    envelopes = {}

    for name, signal in signals.items():

        filtered[name], envelopes[name] = (
            spindle_band_analysis(signal, fs)
        )

    # Detection is performed on raw EEG only.
    candidates, threshold = detect_candidates(
        envelopes["Raw"],
        fs,
    )

    print(
        f"\nEpoch {epoch_idx}: "
        f"{len(candidates)} candidates detected"
    )

    rows = []

    for candidate_id, candidate in enumerate(
        candidates, start=1
    ):

        start = candidate["start_idx"]
        end = candidate["end_idx"]

        raw_mean = np.mean(
            envelopes["Raw"][start:end]
        )

        row = {
            "subject": recording_raw.group,
            "epoch": epoch_idx,
            "candidate_id": candidate_id,
            "start_s": candidate["start_s"],
            "end_s": candidate["end_s"],
            "duration_s": candidate["duration_s"],
            "peak_envelope_raw_uv": (
                candidate["peak_envelope_uv"]
            ),
        }

        for name in ["Raw", "P1", "P2"]:

            segment = filtered[name][start:end]
            envelope_segment = envelopes[name][start:end]

            mean_envelope = float(
                np.mean(envelope_segment)
            )

            rms = float(
                np.sqrt(np.mean(segment ** 2))
            )

            frequency = dominant_frequency(
                segment, fs
            )

            retention = (
                100 * mean_envelope / raw_mean
                if raw_mean > 0
                else np.nan
            )

            row[f"{name}_mean_envelope_uv"] = (
                mean_envelope * 1e6
            )

            row[f"{name}_rms_uv"] = rms * 1e6

            row[f"{name}_dominant_frequency_hz"] = (
                frequency
            )

            row[f"{name}_envelope_retention_pct"] = (
                retention
            )

            # Use the broadband input to each pipeline, not its 11–16 Hz output.
            spectral = spectral_specificity(signals[name], fs, start, end)
            for metric, value in spectral.items():
                row[f"{name}_{metric}"] = value

        # Sigma power is already computed using the same 2-s Welch window
        # by spectral_specificity; compare absolute sigma power, not SSR.
        raw_sigma = row["Raw_sigma_power_uv2"]
        for name in ("P1", "P2"):
            sigma = row[f"{name}_sigma_power_uv2"]
            row[f"{name}_sigma_power_retention_pct"] = (
                100.0 * sigma / raw_sigma if raw_sigma > 0 else np.nan)

        for name in ("Raw", "P1", "P2"):
            _, _, peak_hz = spectral_preservation(
                signals[name], fs, start, end)
            row[f"{name}_sigma_psd_peak_hz"] = peak_hz
        for name in ("P1", "P2"):
            row[f"{name}_sigma_peak_shift_hz"] = (
                row[f"{name}_sigma_psd_peak_hz"] - row["Raw_sigma_psd_peak_hz"])

        rows.append(row)

        print(
            f"  Candidate {candidate_id}: "
            f"{candidate['start_s']:.2f}-"
            f"{candidate['end_s']:.2f} s | "
            f"duration={candidate['duration_s']:.2f} s | "
            f"frequency={row['Raw_dominant_frequency_hz']:.2f} Hz | "
            f"P1={row['P1_envelope_retention_pct']:.1f}% | "
            f"P2={row['P2_envelope_retention_pct']:.1f}% | "
            f"SSR Raw={row['Raw_sigma_specificity_ratio']:.2f}, "
            f"P1={row['P1_sigma_specificity_ratio']:.2f}, "
            f"P2={row['P2_sigma_specificity_ratio']:.2f} | "
            f"Sigma power P1={row['P1_sigma_power_retention_pct']:.1f}%, "
            f"P2={row['P2_sigma_power_retention_pct']:.1f}%"
        )

    # PSD comparisons for each raw-defined candidate.
    for candidate_id, candidate in enumerate(candidates, start=1):
        plot_candidate_psd(epoch_idx, candidate_id, candidate, signals, fs)

    # Save one zoomed inspection figure per detected candidate.
    for candidate_id, candidate in enumerate(candidates, start=1):
        plot_candidate_inspection(
            epoch_idx, candidate_id, candidate,
            signals, filtered, envelopes, fs,
        )

    # --------------------------------------------------------
    # Figure 1: detection over complete epoch
    # --------------------------------------------------------

    time = np.arange(
        len(signals["Raw"])
    ) / fs

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
    )

    axes[0].plot(
        time,
        signals["Raw"] * 1e6,
        color="black",
        linewidth=0.8,
    )

    axes[0].set_title("Raw EEG")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].grid(alpha=0.3)

    for name in ["Raw", "P1", "P2"]:

        axes[1].plot(
            time,
            envelopes[name] * 1e6,
            color=COLORS[name],
            label=name,
            linewidth=1,
        )

    axes[1].axhline(
        threshold * 1e6,
        color="red",
        linestyle="--",
        label="Detection threshold (Raw)",
    )

    for candidate in candidates:

        for ax in axes:
            ax.axvspan(
                candidate["start_s"],
                candidate["end_s"],
                color="orange",
                alpha=0.2,
            )

    axes[1].set_title(
        "Spindle-band envelope and detected candidates"
    )

    axes[1].set_ylabel("Envelope (µV)")
    axes[1].set_xlabel("Time within epoch (s)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"{recording_raw.group} | N2 | Epoch {epoch_idx}"
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / f"epoch_{epoch_idx}_detection.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    # --------------------------------------------------------
    # Figure 2: retention for each candidate
    # --------------------------------------------------------

    if rows:

        ids = [
            row["candidate_id"]
            for row in rows
        ]

        p1_retention = [
            row["P1_envelope_retention_pct"]
            for row in rows
        ]

        p2_retention = [
            row["P2_envelope_retention_pct"]
            for row in rows
        ]

        x = np.arange(len(ids))
        width = 0.35

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.bar(
            x - width / 2,
            p1_retention,
            width,
            label="P1",
            color="tab:blue",
        )

        ax.bar(
            x + width / 2,
            p2_retention,
            width,
            label="P2",
            color="tab:green",
        )

        ax.axhline(
            100,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Raw reference",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(ids)

        ax.set_xlabel("Candidate ID")
        ax.set_ylabel("Mean envelope retention (%)")

        ax.set_title(
            f"Spindle candidate preservation | Epoch {epoch_idx}"
        )

        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"epoch_{epoch_idx}_retention.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(fig)

        # Figure: absolute sigma-band power retention (not SSR).
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(ids))
        width = 0.35
        for offset, name in [(-width / 2, "P1"), (width / 2, "P2")]:
            values = [r[f"{name}_sigma_power_retention_pct"] for r in rows]
            ax.bar(x + offset, values, width, label=name, color=COLORS[name])
        ax.axhline(100, color="black", linestyle="--", label="Raw reference")
        ax.set_xticks(x, ids)
        ax.set(xlabel="Candidate ID", ylabel="Sigma power retention (%)",
               title=f"Sigma-band power preservation | Epoch {epoch_idx}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"epoch_{epoch_idx}_sigma_power_retention.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Figure 3: spectral specificity across candidates.
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(ids))
        width = 0.25
        for offset, name in [(-width, "Raw"), (0, "P1"), (width, "P2")]:
            values = [r[f"{name}_sigma_specificity_ratio"] for r in rows]
            ax.bar(x + offset, values, width, label=name, color=COLORS[name])
        ax.set_xticks(x)
        ax.set_xticklabels(ids)
        ax.set_xlabel("Candidate ID")
        ax.set_ylabel("Sigma specificity ratio (unitless)")
        ax.set_title(f"Spectral specificity | Epoch {epoch_idx}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"epoch_{epoch_idx}_spectral_specificity.png",
                    dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    return rows



# ============================================================
# MORPHOLOGICAL INSPECTION (visual review, not classification)
# ============================================================

def plot_candidate_inspection(epoch_idx, candidate_id, candidate,
                              signals, filtered, envelopes, fs,
                              margin_seconds=0.5):
    """Save a three-panel zoom around a raw-defined spindle candidate."""
    n = len(signals["Raw"])
    start = max(0, int(candidate["start_idx"] - margin_seconds * fs))
    stop = min(n, int(candidate["end_idx"] + margin_seconds * fs))
    t = np.arange(start, stop) / fs

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, source, ylabel, title in zip(
        axes,
        (signals, filtered, envelopes),
        ("Amplitude (µV)", "Amplitude (µV)", "Envelope (µV)"),
        ("Original EEG signals", "Spindle-band EEG (11–16 Hz)",
         "Smoothed Hilbert amplitude envelopes"),
    ):
        for name in ("Raw", "P1", "P2"):
            ax.plot(t, source[name][start:stop] * 1e6,
                    label=name, color=COLORS[name], linewidth=1.1)
        ax.axvspan(candidate["start_s"], candidate["end_s"],
                   color="orange", alpha=0.16, label="Candidate interval")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", ncol=4, fontsize=9)

    axes[-1].set_xlabel("Time within epoch (s)")
    fig.suptitle(
        f"SC400 | N2 | Epoch {epoch_idx} | Candidate {candidate_id} | "
        f"{candidate['start_s']:.2f}–{candidate['end_s']:.2f} s",
        fontsize=13,
    )
    fig.tight_layout()
    output = OUTPUT_DIR / f"epoch_{epoch_idx}_candidate_{candidate_id:02d}_inspection.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Inspection figure: {output.name}")

# ============================================================
# 8. SAVE CSV
# ============================================================

def save_csv(rows):

    output_path = (
        OUTPUT_DIR / "spindle_candidates.csv"
    )

    if not rows:
        print("No candidates found. CSV not created.")
        return

    fieldnames = list(rows[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {output_path}")


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CM2013 - SPINDLE CANDIDATE DETECTION")
    print("=" * 60)

    recording_raw = load_recording()

    recording_p1, recording_p2 = (
        apply_preprocessing(recording_raw)
    )

    all_rows = []

    for epoch_idx in EPOCH_INDICES:

        rows = analyze_epoch(
            recording_raw,
            recording_p1,
            recording_p2,
            epoch_idx,
        )

        all_rows.extend(rows)

    save_csv(all_rows)

    print("\nAnalysis completed.")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
