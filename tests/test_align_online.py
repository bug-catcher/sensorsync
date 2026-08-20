"""Multi-stream online alignment composite (D-0027).

Covers dispatch by ``sample.stream_name``, stream ordering preserved in
frame output, deadline plumbing to each buffer, missing-when-empty
behavior, and a full-run replay whose per-target picks match the
per-stream :meth:`StreamRingBuffer.get_aligned_observation`.
"""

from __future__ import annotations

import pytest

from embodied_sync.align import (
    MultiStreamAligner,
    StreamRingBuffer,
    ZERO_ORDER_HOLD,
)
from embodied_sync.core.episode import AlignedFrame, AlignedSampleMetadata
from embodied_sync.core.sample import Modality, Sample


def _sample(
    *,
    stream_name: str = "robot_state",
    modality: Modality = Modality.ROBOT_STATE,
    sequence_id: int,
    acquisition_time_ns: int,
    receive_time_ns: int | None = None,
    payload: object = None,
) -> Sample:
    return Sample(
        stream_name=stream_name,
        modality=modality,
        sequence_id=sequence_id,
        acquisition_time_ns=acquisition_time_ns,
        receive_time_ns=(
            receive_time_ns if receive_time_ns is not None else acquisition_time_ns
        ),
        source_clock_domain="host_mono",
        payload=payload if payload is not None else [float(sequence_id)],
    )


def _make_aligner(streams: dict[str, tuple[int, int]]) -> MultiStreamAligner:
    return MultiStreamAligner(
        {
            name: StreamRingBuffer(capacity=cap, tolerance_ns=tol)
            for name, (cap, tol) in streams.items()
        }
    )


class TestConstruction:
    def test_rejects_empty_buffer_mapping(self) -> None:
        with pytest.raises(ValueError, match="at least one stream"):
            MultiStreamAligner({})

    def test_stream_names_preserve_insertion_order(self) -> None:
        aligner = _make_aligner(
            {"robot_state": (10, 1_000), "cam_front": (5, 33_000_000), "tactile": (8, 8_000_000)}
        )
        assert aligner.stream_names == ("robot_state", "cam_front", "tactile")

    def test_buffer_lookup_returns_underlying_ring(self) -> None:
        buffers = {
            "robot_state": StreamRingBuffer(capacity=4, tolerance_ns=1_000),
            "cam_front": StreamRingBuffer(capacity=2, tolerance_ns=33_000_000),
        }
        aligner = MultiStreamAligner(buffers)
        assert aligner.buffer("robot_state") is buffers["robot_state"]
        assert aligner.buffer("cam_front") is buffers["cam_front"]

    def test_constructor_snapshots_input_mapping(self) -> None:
        buffers = {"robot_state": StreamRingBuffer(capacity=4, tolerance_ns=1_000)}
        aligner = MultiStreamAligner(buffers)
        # Mutating the original after construction must not affect us.
        buffers["cam_front"] = StreamRingBuffer(capacity=2, tolerance_ns=33_000_000)
        assert aligner.stream_names == ("robot_state",)
        with pytest.raises(KeyError):
            aligner.buffer("cam_front")


class TestPushDispatch:
    def test_push_routes_by_stream_name(self) -> None:
        aligner = _make_aligner({"robot_state": (10, 1_000), "cam_front": (5, 33_000_000)})
        aligner.push(_sample(sequence_id=0, acquisition_time_ns=100))
        aligner.push(_sample(sequence_id=1, acquisition_time_ns=200))
        aligner.push(
            _sample(
                stream_name="cam_front",
                modality=Modality.CAMERA,
                sequence_id=0,
                acquisition_time_ns=50,
                payload={"frame_index": 0},
            )
        )
        assert len(aligner.buffer("robot_state")) == 2
        assert len(aligner.buffer("cam_front")) == 1

    def test_push_unknown_stream_raises_keyerror(self) -> None:
        aligner = _make_aligner({"robot_state": (10, 1_000)})
        stray = _sample(
            stream_name="unknown_stream", sequence_id=0, acquisition_time_ns=100
        )
        with pytest.raises(KeyError, match="unknown_stream"):
            aligner.push(stray)


class TestGetAlignedFrame:
    def test_frame_shape_matches_registered_streams(self) -> None:
        aligner = _make_aligner(
            {"robot_state": (10, 1_000), "cam_front": (5, 33_000_000)}
        )
        aligner.push(_sample(sequence_id=0, acquisition_time_ns=100))
        aligner.push(
            _sample(
                stream_name="cam_front",
                modality=Modality.CAMERA,
                sequence_id=0,
                acquisition_time_ns=50,
                payload={"frame_index": 0},
            )
        )
        frame = aligner.get_aligned_frame(target_ns=200)
        assert isinstance(frame, AlignedFrame)
        assert frame.target_time_ns == 200
        # Keys in registration order.
        assert list(frame.samples.keys()) == ["robot_state", "cam_front"]
        assert list(frame.metadata.keys()) == ["robot_state", "cam_front"]

    def test_missing_when_buffer_empty(self) -> None:
        aligner = _make_aligner({"robot_state": (10, 1_000)})
        frame = aligner.get_aligned_frame(target_ns=200)
        assert frame.samples["robot_state"] is None
        md = frame.metadata["robot_state"]
        assert isinstance(md, AlignedSampleMetadata)
        assert md.missing is True
        assert md.method == ZERO_ORDER_HOLD

    def test_deadline_zero_default_is_causal(self) -> None:
        """A sample with receive_time_ns > target_ns must not be picked at deadline 0."""
        aligner = _make_aligner({"robot_state": (10, 1_000_000)})
        aligner.push(
            _sample(sequence_id=0, acquisition_time_ns=100, receive_time_ns=500)
        )
        frame = aligner.get_aligned_frame(target_ns=200)
        assert frame.samples["robot_state"] is None
        assert frame.metadata["robot_state"].missing is True

    def test_positive_deadline_extends_receive_bound(self) -> None:
        aligner = _make_aligner({"robot_state": (10, 1_000_000)})
        aligner.push(
            _sample(sequence_id=0, acquisition_time_ns=100, receive_time_ns=250)
        )
        # Without slack (deadline 0), receive 250 > target 200 → missing.
        frame_no = aligner.get_aligned_frame(target_ns=200, deadline_ns=0)
        assert frame_no.samples["robot_state"] is None
        # With 100 ns slack, receive 250 <= 200 + 100 = 300 → present.
        frame_yes = aligner.get_aligned_frame(target_ns=200, deadline_ns=100)
        picked = frame_yes.samples["robot_state"]
        assert picked is not None
        assert picked.sequence_id == 0
        assert frame_yes.metadata["robot_state"].missing is False

    def test_frame_composes_per_stream_picks(self) -> None:
        """Composite pick per stream must equal calling the ring buffer directly."""
        aligner = _make_aligner(
            {"robot_state": (10, 5_000_000), "cam_front": (5, 40_000_000)}
        )
        for i in range(5):
            aligner.push(_sample(sequence_id=i, acquisition_time_ns=i * 4_000_000))
        for i in range(3):
            aligner.push(
                _sample(
                    stream_name="cam_front",
                    modality=Modality.CAMERA,
                    sequence_id=i,
                    acquisition_time_ns=i * 33_000_000,
                    payload={"frame_index": i},
                )
            )
        target = 20_000_000
        frame = aligner.get_aligned_frame(target_ns=target)
        for name, buf in (
            ("robot_state", aligner.buffer("robot_state")),
            ("cam_front", aligner.buffer("cam_front")),
        ):
            expected_sample, expected_md = buf.get_aligned_observation(target)
            assert frame.samples[name] is expected_sample
            assert frame.metadata[name] == expected_md


class TestGetLatestPolicyFrame:
    def test_is_deadline_zero_wrapper(self) -> None:
        aligner = _make_aligner({"robot_state": (10, 1_000_000)})
        aligner.push(
            _sample(sequence_id=0, acquisition_time_ns=100, receive_time_ns=150)
        )
        latest = aligner.get_latest_policy_frame(now_ns=200)
        via_frame = aligner.get_aligned_frame(target_ns=200, deadline_ns=0)
        assert latest == via_frame

    def test_now_ns_is_explicit_no_wall_clock(self) -> None:
        """Same now_ns must produce a bit-identical frame across calls."""
        aligner = _make_aligner({"robot_state": (10, 1_000_000)})
        for i in range(3):
            aligner.push(_sample(sequence_id=i, acquisition_time_ns=i * 4_000_000))
        first = aligner.get_latest_policy_frame(now_ns=10_000_000)
        second = aligner.get_latest_policy_frame(now_ns=10_000_000)
        assert first == second


class TestOnlineReplaySynthetic:
    """Push a real synth run into the composite; frames match per-buffer picks."""

    def test_frames_match_per_buffer_picks_for_causal_targets(self) -> None:
        from embodied_sync.streams.synthetic import generate_synthetic_run

        run = generate_synthetic_run(duration_s=1.0, seed=0)
        # Use the three regular sensor streams for a manageable size.
        streams = ("robot_state", "tactile", "cam_front")
        buffers = {}
        for name in streams:
            samples = run[name]
            interval = (
                samples[1].acquisition_time_ns - samples[0].acquisition_time_ns
            )
            buffers[name] = StreamRingBuffer(
                capacity=len(samples), tolerance_ns=interval
            )
        aligner = MultiStreamAligner(buffers)
        for name in streams:
            for sample in run[name]:
                aligner.push(sample)

        # Grid at 10 Hz over the full run. Each per-stream pick must match
        # what the buffer returns on its own.
        for i in range(10):
            target = i * 100_000_000
            frame = aligner.get_aligned_frame(target_ns=target)
            assert frame.target_time_ns == target
            for name in streams:
                expected_sample, expected_md = aligner.buffer(
                    name
                ).get_aligned_observation(target)
                assert frame.samples[name] is expected_sample
                assert frame.metadata[name] == expected_md

    def test_deadline_zero_frame_is_end_to_end_causal(self) -> None:
        """Every present pick in a deadline-0 frame satisfies receive <= target."""
        from embodied_sync.streams.synthetic import generate_synthetic_run

        run = generate_synthetic_run(duration_s=1.0, seed=0)
        streams = ("robot_state", "tactile")
        buffers = {}
        for name in streams:
            samples = run[name]
            interval = (
                samples[1].acquisition_time_ns - samples[0].acquisition_time_ns
            )
            buffers[name] = StreamRingBuffer(
                capacity=len(samples), tolerance_ns=interval
            )
        aligner = MultiStreamAligner(buffers)
        for name in streams:
            for sample in run[name]:
                aligner.push(sample)
        for i in range(1, 10):
            target = i * 100_000_000
            frame = aligner.get_latest_policy_frame(now_ns=target)
            for name in streams:
                picked = frame.samples[name]
                if picked is None:
                    continue
                assert picked.acquisition_time_ns <= target
                assert picked.receive_time_ns <= target
