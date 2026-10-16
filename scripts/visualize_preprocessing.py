"""Visual comparison of raw EEG, P1 bandpass and P2 wavelet denoising.

CM2013 Sleep-EDF Capstone
Preprocessing analysis

Objectives:
- Load one real Sleep-EDF recording.
- Select representative N2 and N3 epochs.
- Compare raw EEG with P1 and P2.
- Assess preservation of physiological EEG morphology.
- Save figures for further analysis.

P1: Butterworth bandpass 0.5-40 Hz, order 4, zero-phase.
P2: Wavelet db4, level auto (max 5), soft thresholding.
"""

from pathlib import Path

import _bootstrap  # Make src/ importable

import numpy as np
import matplotlib.pyplot as plt

from sleepedf import SleepEDFTrack
from sleepedf.preprocessing.experiments import (
    EEGBandpass,
    WaveletDenoising,
)
from scipy.signal import welch, butter, sosfiltfilt, hilbert


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "sleep_edf_data" / "physionet-sleep-data"

OUTPUT_DIR = ROOT / "results" / "preprocessing_visualization"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Select one real Sleep-EDF recording
PSG_FILENAME = "SC4001E0-PSG.edf"

PSG_PATH = DATA_DIR / PSG_FILENAME

# Sleep stages to investigate
STAGES = ["N2", "N3"]

# Number of epochs to visualize per stage
N_EPOCHS_PER_STAGE = 2


# ============================================================
# 2. LOAD REAL EEG DATA
# ============================================================

def load_recording():

    print("\nLoading Sleep-EDF recording...")

    if not PSG_PATH.exists():
        raise FileNotFoundError(
            f"PSG file not found: {PSG_PATH}"
        )

    track = SleepEDFTrack()

    recordings = track.load(
        str(DATA_DIR),
        psg_paths=[PSG_PATH],
    )

    if len(recordings) == 0:
        raise RuntimeError("No recordings were loaded.")

    recording = recordings[0]

    print("\nRecording information:")
    print("Subject:", recording.group)
    print("Sampling frequency:", recording.fs)
    print("Available channels:", list(recording.epochs.keys()))
    print("EEG shape:", recording.epochs["eeg"].shape)

    labels, counts = np.unique(
        recording.labels,
        return_counts=True,
    )

    print("\nSleep stage distribution:")

    for label, count in zip(labels, counts):
        print(f"  {label}: {count} epochs")

    return recording


# ============================================================
# 3. APPLY P1 AND P2
# ============================================================

def apply_preprocessing(recording):

    print("\nApplying preprocessing...")

    # P1: Butterworth bandpass
    p1 = EEGBandpass(
        low_hz=0.5,
        high_hz=40.0,
        order=4,
    )

    # P2: Wavelet denoising
    p2 = WaveletDenoising(
        wavelet="db4",
        level=None,
        threshold_mode="soft",
    )

    # Both methods are applied independently to raw EEG
    recording_p1 = p1.transform(recording)
    recording_p2 = p2.transform(recording)

    print("P1 Butterworth: completed")
    print("P2 Wavelet: completed")

    return recording_p1, recording_p2


# ============================================================
# 4. SELECT EPOCHS
# ============================================================

def select_epochs(recording, stage, n_epochs=2):

    labels = np.asarray(recording.labels)

    indices = np.where(labels == stage)[0]

    if len(indices) == 0:
        print(f"No epochs found for stage {stage}")
        return []

    # Select evenly spaced epochs to inspect different
    # portions of the recording.
    n_select = min(n_epochs, len(indices))

    positions = np.linspace(
        0,
        len(indices) - 1,
        n_select,
        dtype=int,
    )

    selected = indices[positions]

    print(
        f"\nSelected {stage} epochs:",
        selected.tolist(),
    )

    return selected.tolist()


# ============================================================
# 5. VISUALIZE RAW / P1 / P2
# ============================================================

def plot_epoch_comparison(
    recording_raw,
    recording_p1,
    recording_p2,
    epoch_idx,
    stage,
):

    # Extract the same epoch from each recording
    raw = np.asarray(
        recording_raw.epochs["eeg"][epoch_idx]
    )

    p1 = np.asarray(
        recording_p1.epochs["eeg"][epoch_idx]
    )

    p2 = np.asarray(
        recording_p2.epochs["eeg"][epoch_idx]
    )

    # Time axis
    fs = recording_raw.fs

    time = np.arange(len(raw)) / fs

    # Create figure
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 9),
        sharex=True,
        sharey=True,
    )

    signals = [raw, p1, p2]

    titles = [
        "Raw EEG",
        "P1 - Butterworth bandpass (0.5-40 Hz)",
        "P2 - Wavelet denoising (db4, soft)",
    ]

    colors = [
        "black",
        "tab:blue",
        "tab:green",
    ]

    for ax, signal, title, color in zip(
        axes,
        signals,
        titles,
        colors,
    ):

        # Convert volts to microvolts for readability
        signal_uv = signal * 1e6

        ax.plot(
            time,
            signal_uv,
            color=color,
            linewidth=0.7,
        )

        ax.set_title(title)

        ax.set_ylabel("Amplitude (µV)")

        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(
        f"Subject {recording_raw.group} | "
        f"Stage {stage} | Epoch {epoch_idx}",
        fontsize=14,
    )

    plt.tight_layout()

    # Save figure
    filename = (
        f"{recording_raw.group}_"
        f"{stage}_epoch_{epoch_idx}_comparison.png"
    )

    output_path = OUTPUT_DIR / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved figure:", output_path)

    plt.show()

    plt.close(fig)


# ============================================================
# 6. QUANTITATIVE COMPARISON
# ============================================================

def compare_signal_statistics(
    recording_raw,
    recording_p1,
    recording_p2,
    epoch_idx,
    stage,
):

    raw = np.asarray(
        recording_raw.epochs["eeg"][epoch_idx]
    )

    p1 = np.asarray(
        recording_p1.epochs["eeg"][epoch_idx]
    )

    p2 = np.asarray(
        recording_p2.epochs["eeg"][epoch_idx]
    )

    signals = {
        "Raw": raw,
        "P1": p1,
        "P2": p2,
    }

    print(
        f"\nSignal statistics | "
        f"Stage {stage} | Epoch {epoch_idx}"
    )

    print(
        f"{'Signal':<10}"
        f"{'Std (µV)':>15}"
        f"{'RMS (µV)':>15}"
        f"{'Peak-to-peak (µV)':>22}"
    )

    print("-" * 62)

    for name, signal in signals.items():

        signal_uv = signal * 1e6

        std = np.std(signal_uv)

        rms = np.sqrt(
            np.mean(signal_uv ** 2)
        )

        ptp = np.ptp(signal_uv)

        print(
            f"{name:<10}"
            f"{std:>15.3f}"
            f"{rms:>15.3f}"
            f"{ptp:>22.3f}"
        )


# ============================================================
# 7. SPINDLE ANALYSIS
# ============================================================

def analyze_spindles(
    recording_raw,
    recording_p1,
    recording_p2,
    epoch_idx,
):

    fs = recording_raw.fs

    signals = {
        "Raw": np.asarray(
            recording_raw.epochs["eeg"][epoch_idx]
        ),
        "P1": np.asarray(
            recording_p1.epochs["eeg"][epoch_idx]
        ),
        "P2": np.asarray(
            recording_p2.epochs["eeg"][epoch_idx]
        ),
    }

    # --------------------------------------------------------
    # A. Welch PSD
    # --------------------------------------------------------

    band_low = 11.0
    band_high = 16.0

    band_powers = {}
    psd_results = {}

    for name, signal in signals.items():

        frequencies, psd = welch(
            signal,
            fs=fs,
            nperseg=int(4 * fs),
            noverlap=int(2 * fs),
        )

        mask = (
            (frequencies >= band_low)
            & (frequencies <= band_high)
        )

        band_power = np.trapezoid(
            psd[mask],
            frequencies[mask],
        )

        band_powers[name] = band_power
        psd_results[name] = (frequencies, psd)

    raw_power = band_powers["Raw"]

    print(
        f"\nSpindle-band analysis - Epoch {epoch_idx}"
    )

    print("Band: 11-16 Hz")

    for name, power in band_powers.items():

        if raw_power > 0:
            retention = 100 * power / raw_power
        else:
            retention = np.nan

        print(
            f"{name}: "
            f"Power = {power * 1e12:.4f} µV² | "
            f"Retention = {retention:.2f}%"
        )

    # --------------------------------------------------------
    # B. Spindle-band temporal envelope
    # --------------------------------------------------------

    sos = butter(
        4,
        [band_low, band_high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    envelopes = {}

    for name, signal in signals.items():

        filtered = sosfiltfilt(sos, signal)

        analytic = hilbert(filtered)

        envelope = np.abs(analytic)

        envelopes[name] = envelope

    time = np.arange(len(signals["Raw"])) / fs

    # --------------------------------------------------------
    # C. Plot PSD
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
    )

    colors = {
        "Raw": "black",
        "P1": "tab:blue",
        "P2": "tab:green",
    }

    for name, (frequencies, psd) in psd_results.items():

        axes[0].plot(
            frequencies,
            psd * 1e12,
            label=name,
            color=colors[name],
        )

    axes[0].axvspan(
        band_low,
        band_high,
        color="orange",
        alpha=0.15,
        label="Spindle band",
    )

    axes[0].set_xlim(0, 30)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (µV²/Hz)")
    axes[0].set_title("Welch Power Spectral Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --------------------------------------------------------
    # D. Plot temporal envelope
    # --------------------------------------------------------

    for name, envelope in envelopes.items():

        axes[1].plot(
            time,
            envelope * 1e6,
            label=name,
            color=colors[name],
            linewidth=1,
        )

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Envelope amplitude (µV)")
    axes[1].set_title(
        "Spindle-band amplitude envelope (11-16 Hz)"
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Spindle-band analysis | "
        f"Subject {recording_raw.group} | "
        f"Epoch {epoch_idx}",
        fontsize=14,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{recording_raw.group}_N2_epoch_{epoch_idx}_spindles.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

# ============================================================
# SPINDLE CANDIDATE INSPECTION
# ============================================================

def inspect_spindle_candidate(
    recording_raw,
    recording_p1,
    recording_p2,
    epoch_idx,
    start_s,
    end_s,
):
    """Compare raw/P1/P2 in a candidate spindle window.

    The 11-16 Hz filter is used for visualization only.
    It does not modify the preprocessing pipelines.
    """

    fs = float(recording_raw.fs)

    signals = {
        "Raw": np.asarray(
            recording_raw.epochs["eeg"][epoch_idx], dtype=float
        ),
        "P1": np.asarray(
            recording_p1.epochs["eeg"][epoch_idx], dtype=float
        ),
        "P2": np.asarray(
            recording_p2.epochs["eeg"][epoch_idx], dtype=float
        ),
    }

    colors = {
        "Raw": "black",
        "P1": "tab:blue",
        "P2": "tab:green",
    }

    n_samples = len(signals["Raw"])

    if any(len(x) != n_samples for x in signals.values()):
        raise ValueError("Signals must have identical lengths.")

    if not (0 <= start_s < end_s <= n_samples / fs):
        raise ValueError("Invalid inspection window.")

    # --------------------------------------------------------
    # A. Spindle-band filtering
    # --------------------------------------------------------

    sos = butter(
        4,
        [11.0, 16.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = {}
    envelopes = {}

    for name, signal in signals.items():

        # Filter the complete epoch before selecting the window.
        # This avoids artificial edge effects at the zoom boundaries.
        filtered[name] = sosfiltfilt(sos, signal)

        envelopes[name] = np.abs(
            hilbert(filtered[name])
        )

    # --------------------------------------------------------
    # B. Select the candidate window
    # --------------------------------------------------------

    start_idx = int(round(start_s * fs))
    end_idx = int(round(end_s * fs))

    indices = slice(start_idx, end_idx)

    time = np.arange(n_samples) / fs
    time_zoom = time[indices]

    # --------------------------------------------------------
    # C. Plot the three comparisons
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 11),
        sharex=True,
    )

    # 1. Original EEG signals
    for name, signal in signals.items():

        axes[0].plot(
            time_zoom,
            signal[indices] * 1e6,
            label=name,
            color=colors[name],
            linewidth=1,
            alpha=0.85,
        )

    axes[0].set_title("Original EEG signals")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Spindle-band signals
    for name, signal in filtered.items():

        axes[1].plot(
            time_zoom,
            signal[indices] * 1e6,
            label=name,
            color=colors[name],
            linewidth=1.2,
            alpha=0.85,
        )

    axes[1].set_title(
        "EEG filtered in spindle band (11-16 Hz)"
    )
    axes[1].set_ylabel("Amplitude (µV)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3. Hilbert envelopes
    for name, envelope in envelopes.items():

        axes[2].plot(
            time_zoom,
            envelope[indices] * 1e6,
            label=name,
            color=colors[name],
            linewidth=1.5,
            alpha=0.85,
        )

    axes[2].set_title(
        "Spindle-band amplitude envelope"
    )
    axes[2].set_ylabel("Envelope (µV)")
    axes[2].set_xlabel("Time within epoch (s)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"Spindle candidate inspection | "
        f"{recording_raw.group} | "
        f"N2 Epoch {epoch_idx} | "
        f"{start_s}-{end_s} s",
        fontsize=15,
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # D. Save figure
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / (
            f"{recording_raw.group}_N2_epoch_{epoch_idx}"
            f"_spindle_zoom_{start_s:g}_{end_s:g}s.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"\nSpindle candidate: epoch {epoch_idx}")
    print(f"Window: {start_s}-{end_s} s")
    print(f"Saved figure: {output_path}")

    # --------------------------------------------------------
    # E. Quantitative envelope comparison
    # --------------------------------------------------------

    raw_envelope = envelopes["Raw"][indices]

    raw_mean = np.mean(raw_envelope)

    print("\nMean spindle-band envelope:")

    for name, envelope in envelopes.items():

        mean_amplitude = np.mean(
            envelope[indices]
        )

        retention = (
            100 * mean_amplitude / raw_mean
            if raw_mean > 0
            else np.nan
        )

        print(
            f"{name}: "
            f"{mean_amplitude * 1e6:.3f} µV | "
            f"Retention = {retention:.2f}%"
        )

    plt.show()
    plt.close(fig)

def main():

    print("=" * 60)
    print("CM2013 - EEG PREPROCESSING VISUALIZATION")
    print("=" * 60)

    # Load recording
    recording_raw = load_recording()

    # Apply P1 and P2
    recording_p1, recording_p2 = apply_preprocessing(
        recording_raw
    )

    # Compare N2 and N3
    for stage in STAGES:

        epoch_indices = select_epochs(
            recording_raw,
            stage,
            n_epochs=N_EPOCHS_PER_STAGE,
        )

        for epoch_idx in epoch_indices:

            print(
                f"\nProcessing {stage}, epoch {epoch_idx}"
            )

            # Visual comparison
            plot_epoch_comparison(
                recording_raw,
                recording_p1,
                recording_p2,
                epoch_idx,
                stage,
            )

            # Numerical comparison
            compare_signal_statistics(
                recording_raw,
                recording_p1,
                recording_p2,
                epoch_idx,
                stage,
            )

            if stage == "N2":
                analyze_spindles(
                    recording_raw,
                    recording_p1,
                    recording_p2,
                    epoch_idx,
                )
                if epoch_idx == 1025:
                    inspect_spindle_candidate(
                        recording_raw,
                        recording_p1,
                        recording_p2,
                        epoch_idx=1025,
                        start_s=26.0,
                        end_s=29.0,
                    )
                elif epoch_idx == 1637:
                    inspect_spindle_candidate(
                        recording_raw,
                        recording_p1,
                        recording_p2,
                        epoch_idx=1637,
                        start_s=6.0,
                        end_s=9.0,
                    )

    print("\nAnalysis completed.")
    print("Figures saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()