"""Recording-level preprocessing contract."""
from typing import Protocol

from ..reference.adapter import Recording


class Preprocessor(Protocol):
    def transform(self, recording: Recording) -> Recording:
        """Return aligned signals and labels, preserving subject and metadata.

        Do not modify the input recording. If epochs are removed, apply the
        same mask to every channel and the labels and record the QC decision.
        """
        ...


class NoPreprocessing:
    """P0: supplied baseline, with no signal modification."""

    def transform(self, recording: Recording) -> Recording:
        return recording
