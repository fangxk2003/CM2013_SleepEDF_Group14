"""P1–P5 interfaces only: no new preprocessing is implemented."""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..reference.adapter import Recording, bandpass_notch



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
        raise NotImplementedError("P2: wavelet denoising is a planned experiment.")


@dataclass
class BrokenSegmentHandling:
    """P3: flag/exclude objectively identified dropout or near-flat epochs.

    Fix QC thresholds before evaluation. Report affected counts, stages and
    subjects. Never repair a broken epoch merely by making it look filtered.
    """

    action: Literal["flag", "exclude"] = "flag"

    def detect(self, recording: Recording) -> np.ndarray:
        """Return one boolean per epoch; True identifies a broken epoch."""
        raise NotImplementedError("P3: define and implement objective QC rules.")

    def transform(self, recording: Recording) -> Recording:
        raise NotImplementedError("P3: aligned flagging/exclusion is not implemented.")


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
