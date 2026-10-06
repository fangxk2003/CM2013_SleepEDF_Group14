"""Preprocessing experiments P0–P5; only P0 is implemented."""
from .base import NoPreprocessing, Preprocessor
from .experiments import (
    BrokenSegmentHandling, EEGBandpass, SubjectRecordingNormalisation,
    TargetedDenoising, WaveletDenoising,
)


def make_preprocessor(name="none") -> Preprocessor:
    """Select a recording transform. Planned experiments fail when called."""
    choices = {
        "none": NoPreprocessing, "p0": NoPreprocessing,
        "bandpass": EEGBandpass, "p1": EEGBandpass,
        "wavelet": WaveletDenoising, "p2": WaveletDenoising,
        "broken_segments": BrokenSegmentHandling, "p3": BrokenSegmentHandling,
        "normalisation": SubjectRecordingNormalisation, "p4": SubjectRecordingNormalisation,
        "denoise": TargetedDenoising, "p5": TargetedDenoising,
    }
    key = str(name or "none").lower()
    if key not in choices:
        raise ValueError(f"Unknown preprocessing experiment: {name!r}")
    return choices[key]()
