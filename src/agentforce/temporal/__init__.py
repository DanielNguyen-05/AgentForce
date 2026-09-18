"""Temporal alignment and decoder-neutral dense frame refinement."""

from .alignment import (
    AlignmentResult,
    ViterbiPath,
    align_ordered_candidates,
    ordered_viterbi,
)
from .refinement import (
    DecodedFrame,
    DenseFrameRefiner,
    DenseFrameSource,
    DenseRefiner,
    FrameCandidate,
    FrameScorer,
    NoOpDenseRefiner,
)
from .opencv_source import OpenCVFrameSource
from .clip_scorer import OpenCLIPFrameScorer

__all__ = [
    "AlignmentResult",
    "DecodedFrame",
    "DenseFrameRefiner",
    "DenseFrameSource",
    "DenseRefiner",
    "FrameCandidate",
    "FrameScorer",
    "NoOpDenseRefiner",
    "OpenCLIPFrameScorer",
    "OpenCVFrameSource",
    "ViterbiPath",
    "align_ordered_candidates",
    "ordered_viterbi",
]
