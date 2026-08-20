"""Alignment engine (Milestone 3, offline nearest-neighbor + ZoH).

The data types (:class:`AlignedFrame`, :class:`AlignedRun`,
:class:`AlignedSampleMetadata`, :class:`AlignmentReport`) live in
:mod:`embodied_sync.core.episode`; they are re-exported here for
backward compatibility with callers that imported them from ``align``.
:data:`Episode` is a type alias of :class:`AlignedRun`.
"""

from embodied_sync.align.engine import (
    LINEAR_INTERPOLATION,
    NEAREST_NEIGHBOR,
    ZERO_ORDER_HOLD,
    AlignedFrame,
    AlignedRun,
    AlignedSampleMetadata,
    AlignmentReport,
    Method,
    MethodArg,
    aggregate_window,
    align_run,
)
from embodied_sync.align.online import MultiStreamAligner
from embodied_sync.align.ring_buffer import WINDOW, StreamRingBuffer
from embodied_sync.core.episode import Episode

__all__ = [
    "LINEAR_INTERPOLATION",
    "NEAREST_NEIGHBOR",
    "WINDOW",
    "ZERO_ORDER_HOLD",
    "AlignedFrame",
    "AlignedRun",
    "AlignedSampleMetadata",
    "AlignmentReport",
    "Episode",
    "Method",
    "MethodArg",
    "MultiStreamAligner",
    "StreamRingBuffer",
    "aggregate_window",
    "align_run",
]
