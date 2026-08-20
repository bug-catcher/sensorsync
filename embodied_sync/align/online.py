"""Multi-stream online alignment composite (D-0027).

The single-stream :class:`~embodied_sync.align.ring_buffer.StreamRingBuffer`
(D-0026) answers "what's the best sample from *this* stream at target
time T?" — but a policy tick usually wants an :class:`AlignedFrame`
covering *every* stream. :class:`MultiStreamAligner` is the small
composite that holds one :class:`StreamRingBuffer` per stream, dispatches
:meth:`push` by ``sample.stream_name``, and builds a frame by asking each
buffer for its ZoH pick at the same ``(target_ns, deadline_ns)`` pair.

Policy tick shape
-----------------

The natural online control loop is::

    aligner = MultiStreamAligner({...})
    while running:
        for sample in incoming_samples:  # producer thread(s), no batching
            aligner.push(sample)
        frame = aligner.get_latest_policy_frame(now_ns=clock())
        act(frame)

At ``deadline_ns == 0`` every per-stream pick satisfies the D-0026
causality invariant, so the frame as a whole is causal.
:meth:`get_aligned_frame` accepts an explicit ``deadline_ns`` for callers
that want to give slower streams a bounded amount of slack; it is passed
through to each buffer unchanged. Nothing here reads the wall clock —
the caller injects ``target_ns`` (and, for :meth:`get_latest_policy_frame`,
``now_ns``) so tests stay deterministic.

This is intentionally not a rewrite of :func:`~embodied_sync.align.align_run`
for the online case: it is the smallest possible composition that turns
the per-stream primitive into an :class:`AlignedFrame`-producing surface.
Multi-stream policy selection, window aggregation, and deadline-aware
nearest-neighbor stay deferred (D-0026).
"""

from __future__ import annotations

from typing import Mapping

from embodied_sync.align.ring_buffer import StreamRingBuffer
from embodied_sync.core.episode import AlignedFrame, AlignedSampleMetadata
from embodied_sync.core.sample import Sample

__all__ = ["MultiStreamAligner"]


class MultiStreamAligner:
    """Composes one :class:`StreamRingBuffer` per stream into a frame builder.

    Buffers are owned by the caller: pass a mapping of ``stream_name`` to
    ``StreamRingBuffer`` at construction. The mapping's iteration order is
    the frame's stream order — matching how :class:`AlignedFrame` preserves
    ``dict.keys()`` insertion order for the offline engine.
    """

    __slots__ = ("_buffers",)

    def __init__(self, buffers: Mapping[str, StreamRingBuffer]) -> None:
        if not buffers:
            raise ValueError("buffers must contain at least one stream")
        # Snapshot into our own dict so later external mutations of the
        # caller's mapping cannot re-order or drop streams mid-flight.
        self._buffers: dict[str, StreamRingBuffer] = dict(buffers)

    @property
    def stream_names(self) -> tuple[str, ...]:
        """Registered stream names in frame order."""
        return tuple(self._buffers.keys())

    def buffer(self, stream_name: str) -> StreamRingBuffer:
        """Return the underlying ring buffer for ``stream_name``."""
        return self._buffers[stream_name]

    def push(self, sample: Sample) -> None:
        """Route ``sample`` to the buffer named by ``sample.stream_name``.

        Raises :class:`KeyError` if the stream is not registered — a
        typo in a producer must not silently drop data.
        """
        try:
            buf = self._buffers[sample.stream_name]
        except KeyError:
            raise KeyError(
                f"sample stream {sample.stream_name!r} not registered; "
                f"known streams: {list(self._buffers)}"
            ) from None
        buf.push(sample)

    def get_aligned_frame(
        self,
        target_ns: int,
        *,
        deadline_ns: int = 0,
    ) -> AlignedFrame:
        """Assemble an :class:`AlignedFrame` at ``target_ns``.

        Each stream's pick comes from its ring buffer's
        :meth:`~StreamRingBuffer.get_aligned_observation` at the same
        ``(target_ns, deadline_ns)``. At ``deadline_ns == 0`` every pick
        respects the D-0026 causality invariant, so the composite is
        causal by construction. Missing picks land in ``samples`` as
        ``None`` with ``metadata[name].missing = True``, mirroring the
        offline engine's shape.
        """
        samples: dict[str, Sample | None] = {}
        metadata: dict[str, AlignedSampleMetadata] = {}
        for name, buf in self._buffers.items():
            sample, meta = buf.get_aligned_observation(
                target_ns, deadline_ns=deadline_ns
            )
            samples[name] = sample
            metadata[name] = meta
        return AlignedFrame(
            target_time_ns=target_ns,
            samples=samples,
            metadata=metadata,
        )

    def get_latest_policy_frame(self, now_ns: int) -> AlignedFrame:
        """Return the deadline-0 aligned frame for target=``now_ns``.

        Thin wrapper around
        ``get_aligned_frame(now_ns, deadline_ns=0)``. ``now_ns`` is an
        explicit argument — the library does not read the wall clock, so
        tests can pin ``now_ns`` and get bit-identical output every run.
        This is the multi-stream analogue of
        :meth:`StreamRingBuffer.get_latest_policy_observation`.
        """
        return self.get_aligned_frame(now_ns, deadline_ns=0)
