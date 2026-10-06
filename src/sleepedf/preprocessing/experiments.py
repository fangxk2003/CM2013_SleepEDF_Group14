"""P1–P5 interfaces only: no new preprocessing is implemented."""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..reference.adapter import Recording, bandpass_notch, wavelet_denoise



@dataclass
class EEGBandpass:
    """P1: filter EEG at 0.5–40 Hz; retain EOG/EMG and epoch alignment."""

    low_hz: float = 0.5
    high_hz: float = 40.0
    order: int = 4

    def transform(self, recording: Recording) -> Recording:
        return bandpass_notch(
            recording,
            band=(self.low_hz, self.high_hz),
            notch=None,
            order=self.order,
            channels=("eeg",),
            causal=False,
        )


@dataclass
class WaveletDenoising:
    """P2: wavelet denoising; assess preservation of N2 spindles and N3."""

    wavelet: str = "db4"
    level: int | None = None
    threshold_mode: Literal["soft", "hard"] = "soft"

    def transform(self, recording: Recording) -> Recording:
        return wavelet_denoise(
            recording,
            wavelet=self.wavelet,
            level=self.level,
            threshold_mode=self.threshold_mode,
            threshold=None,
            channels=("eeg",),
        )


@dataclass
class BrokenSegmentHandling:
    """P3: detect and handle objectively identified near-flat epochs.

    An epoch is considered broken when both EEG and EOG show
    exceptionally low variability over the complete epoch.

    The thresholds were fixed from the Iteration 2 QC analysis
    before classifier evaluation. They are not estimated from
    labels or recomputed during LOSO evaluation.

    If action="flag", the recording is returned with the broken
    epoch mask stored in metadata.

    If action="exclude", the same mask is applied to every signal
    channel and to the labels so that alignment is preserved.
    """

    action: Literal["flag", "exclude"] = "flag"

    eeg_std_threshold: float = 4.820492569885e-06
    eog_std_threshold: float = 7.200175602286e-06

    def detect(self, recording: Recording) -> np.ndarray:
        """Return one boolean per epoch; True means broken."""

        eeg = np.asarray(recording.epochs["eeg"])
        eog = np.asarray(recording.epochs["eog"])

        eeg_std = np.std(eeg, axis=1)
        eog_std = np.std(eog, axis=1)

        broken = (
            (eeg_std < self.eeg_std_threshold)
            & (eog_std < self.eog_std_threshold)
        )

        return broken

    def transform(self, recording: Recording) -> Recording:
        """Flag or exclude epochs identified by detect()."""

        broken = self.detect(recording)

        # Copy metadata so the input Recording is not modified.
        meta = dict(recording.meta)

        meta["broken_segments"] = {
            "method": "near_flat_eeg_eog",
            "eeg_std_threshold": self.eeg_std_threshold,
            "eog_std_threshold": self.eog_std_threshold,
            "n_broken": int(np.sum(broken)),
            "broken_epoch_indices": np.flatnonzero(broken).tolist(),
            "action": self.action,
        }

        if self.action == "flag":

            meta["broken_epoch_mask"] = broken.copy()

            return Recording(
                group=recording.group,
                fs=recording.fs,
                epochs={
                    ch: np.asarray(values).copy()
                    for ch, values in recording.epochs.items()
                },
                labels=np.asarray(recording.labels).copy(),
                meta=meta,
            )

        if self.action == "exclude":

            keep = ~broken

            epochs = {
                ch: np.asarray(values)[keep].copy()
                for ch, values in recording.epochs.items()
            }

            labels = np.asarray(recording.labels)[keep].copy()

            return Recording(
                group=recording.group,
                fs=recording.fs,
                epochs=epochs,
                labels=labels,
                meta=meta,
            )

        raise ValueError(
            f"Unknown BrokenSegmentHandling action: {self.action!r}"
        )

@dataclass
class SubjectRecordingNormalisation:
    """P4: normalise recording signals through the shared preprocessing interface.

    Define the normalisation rule and subject/recording scope before use.
    Preserve epoch alignment, labels, subject identity and metadata without
    modifying the input. Stage labels must not determine normalisation.
    """

    scope: Literal["subject", "recording"] = "recording"

    def transform(self, recording: Recording) -> Recording:
        raise NotImplementedError("P4: subject/recording normalisation is a planned experiment.")


@dataclass
class TargetedDenoising:
    """P5: choose a remedy only after visual/spectral evidence of corruption."""

    noise_type: Literal["impulse", "baseline", "mains", "broadband"] | None = None
    evidence: str | None = None

    def transform(self, recording: Recording) -> Recording:
        raise NotImplementedError("P5: document confirmed corruption and implement its remedy.")
