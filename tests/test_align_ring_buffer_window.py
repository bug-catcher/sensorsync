"""``StreamRingBuffer.get_window`` and the push-order tie-break (D-0037).

The existing ring-buffer tests pin the single-pick behaviour and must
keep passing unchanged; this file covers the two additions.
"""

from __future__ import annotations

import pytest

from embodied_sync.align import WINDOW, StreamRingBuffer
from embodied_sync.core.sample import Modality, Sample


def _sample(
    *,
    sequence_id: int,
    acquisition_time_ns: int,
    receive_time_ns: int | None = None,
    payload: object = None,
) -> Sample:
    return Sample(
        stream_name="s",
        modality=Modality.OTHER,
        sequence_id=sequence_id,
        acquisition_time_ns=acquisition_time_ns,
        receive_time_ns=(
            acquisition_time_ns if receive_time_ns is None else receive_time_ns
        ),
        source_clock_domain="host_mono",
        payload=payload,
    )


class TestGetWindow:
    def test_returns_every_sample_inside_the_window(self) -> None:
        buf = StreamRingBuffer(capacity=16, tolerance_ns=1_000)
        for i in range(10):
            buf.push(_sample(sequence_id=i, acquisition_time_ns=i * 100))
        # Window +-250 ns around 500 -> acquisition in [250, 750]. The
        # future half needs deadline slack (see get_window's contract).
        window, metadata = buf.get_window(500, window_ns=500, deadline_ns=250)
        assert [s.acquisition_time_ns for s in window] == [300, 400, 500, 600, 700]
        assert metadata.method == WINDOW
        assert metadata.missing is False
        assert metadata.source_time_ns == 500
        assert metadata.skew_ns == 0
        assert metadata.confidence == 1.0

    def test_window_is_inclusive_at_both_edges(self) -> None:
        buf = StreamRingBuffer(capacity=16, tolerance_ns=1_000)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=250))
        buf.push(_sample(sequence_id=1, acquisition_time_ns=750))
        window, _ = buf.get_window(500, window_ns=500, deadline_ns=250)
        assert len(window) == 2

    def test_samples_come_back_in_push_order(self) -> None:
        buf = StreamRingBuffer(capacity=16, tolerance_ns=1_000)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=520))
        buf.push(_sample(sequence_id=1, acquisition_time_ns=480))
        buf.push(_sample(sequence_id=2, acquisition_time_ns=500))
        window, _ = buf.get_window(500, window_ns=200, deadline_ns=100)
        assert [s.sequence_id for s in window] == [0, 1, 2]

    def test_empty_window_is_missing(self) -> None:
        buf = StreamRingBuffer(capacity=16, tolerance_ns=1_000)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=0))
        window, metadata = buf.get_window(10_000, window_ns=100)
        assert window == []
        assert metadata.missing is True
        assert metadata.source_time_ns is None
        assert metadata.skew_ns is None
        assert metadata.confidence == 0.0

    def test_deadline_zero_excludes_samples_not_yet_received(self) -> None:
        buf = StreamRingBuffer(capacity=16, tolerance_ns=1_000)
        # Acquired inside the window but not received until well after it.
        buf.push(
            _sample(
                sequence_id=0, acquisition_time_ns=520, receive_time_ns=5_000
            )
        )
        buf.push(_sample(sequence_id=1, acquisition_time_ns=480))
        window, _ = buf.get_window(500, window_ns=200, deadline_ns=0)
        assert [s.sequence_id for s in window] == [1]
        # With enough deadline slack the late arrival becomes eligible.
        window, _ = buf.get_window(500, window_ns=200, deadline_ns=5_000)
        assert [s.sequence_id for s in window] == [0, 1]

    def test_representative_is_the_nearest_sample(self) -> None:
        buf = StreamRingBuffer(capacity=16, tolerance_ns=1_000)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=400))
        buf.push(_sample(sequence_id=1, acquisition_time_ns=560))
        window, metadata = buf.get_window(500, window_ns=400, deadline_ns=200)
        assert len(window) == 2
        assert metadata.source_time_ns == 560
        assert metadata.skew_ns == 60

    @pytest.mark.parametrize(("window_ns", "deadline_ns"), [(0, 0), (-1, 0), (100, -1)])
    def test_invalid_arguments_raise(self, window_ns: int, deadline_ns: int) -> None:
        buf = StreamRingBuffer(capacity=4, tolerance_ns=10)
        with pytest.raises(ValueError):
            buf.get_window(0, window_ns=window_ns, deadline_ns=deadline_ns)


class TestPushOrderTieBreak:
    """Equal acquisition times resolve to the last push (mcap's position rule)."""

    def test_zoh_prefers_the_last_push_on_an_exact_tie(self) -> None:
        buf = StreamRingBuffer(capacity=8, tolerance_ns=1_000)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=100, payload="first"))
        buf.push(_sample(sequence_id=1, acquisition_time_ns=100, payload="second"))
        buf.push(_sample(sequence_id=2, acquisition_time_ns=100, payload="third"))
        pick, metadata = buf.get_aligned_observation(150)
        assert pick is not None
        assert pick.payload == "third"
        assert metadata.source_time_ns == 100

    def test_nearest_neighbor_prefers_the_last_push_on_an_exact_tie(self) -> None:
        buf = StreamRingBuffer(capacity=8, tolerance_ns=1_000)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=100, payload="first"))
        buf.push(_sample(sequence_id=1, acquisition_time_ns=100, payload="second"))
        pick, _ = buf.get_nearest_neighbor(150, deadline_ns=1_000)
        assert pick is not None
        assert pick.payload == "second"

    def test_nearest_neighbor_keeps_the_causal_side_on_a_straddling_tie(self) -> None:
        """Equal |skew| at *different* times is a before/after choice, not a tie.

        The earlier (already-observed) sample is the safer pick and the
        existing behaviour; the push-order rule must not disturb it.
        """
        buf = StreamRingBuffer(capacity=8, tolerance_ns=1_000)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=100, payload="before"))
        buf.push(_sample(sequence_id=1, acquisition_time_ns=200, payload="after"))
        pick, _ = buf.get_nearest_neighbor(150, deadline_ns=1_000)
        assert pick is not None
        assert pick.payload == "before"

    def test_tie_break_is_stable_across_repeated_queries(self) -> None:
        buf = StreamRingBuffer(capacity=8, tolerance_ns=1_000)
        for i in range(5):
            buf.push(_sample(sequence_id=i, acquisition_time_ns=100, payload=i))
        picks = {buf.get_aligned_observation(100)[0] for _ in range(10)}
        assert len(picks) == 1
