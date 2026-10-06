"""Preprocessing experiments P0–P5; only P0 is implemented."""
from .base import NoPreprocessing, Preprocessor
from .experiments import (
    BrokenSegmentHandling, EEGBandpass, SubjectRecordingNormalisation,
    TargetedDenoising, WaveletDenoising,
)


def make_preprocessor(name="none") -> Preprocessor:
    """Select the preprocessing transform. Planned experiments fail when called."""

    key = str(name or "none").lower()

    if key in ("broken_segments", "p3"):
        return BrokenSegmentHandling(action="exclude")

    choices = {
        "none": NoPreprocessing,
        "p0": NoPreprocessing,
        "bandpass": EEGBandpass,
        "p1": EEGBandpass,
        "wavelet": WaveletDenoising,
        "p2": WaveletDenoising,
        "normalisation": SubjectRecordingNormalisation,
        "p4": SubjectRecordingNormalisation,
        "denoise": TargetedDenoising,
        "p5": TargetedDenoising,
    }

    if key not in choices:
        raise ValueError(
            f"Unknown preprocessing experiment: {name!r}"
        )

    return choices[key]()