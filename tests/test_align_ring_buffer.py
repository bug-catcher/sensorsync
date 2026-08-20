"""Online alignment ring buffer (D-0026).

Covers push/eviction semantics, ZoH picking, the causality invariant at
``deadline_ns == 0`` (source in ``receive_time_ns <= target_ns``), the
deadline-shift semantic at ``deadline_ns > 0``, staleness gating via
``tolerance_ns``, missing-metadata shape, and construction guards.
"""

from __future__ import annotations

import pytest

from embodied_sync.align import StreamRingBuffer, ZERO_ORDER_HOLD
from embodied_sync.core.episode import AlignedSampleMetadata
from embodied_sync.core.sample import Modality, Sample


def _sample(
    *,
    sequence_id: int,
    acquisition_time_ns: int,
    receive_time_ns: int | None = None,
    payload: object = None,
) -> Sample:
    return Sample(
        stream_name="robot_state",
        modality=Modality.ROBOT_STATE,
        sequence_id=sequence_id,
        acquisition_time_ns=acquisition_time_ns,
        receive_time_ns=receive_time_ns if receive_time_ns is not None else acquisition_time_ns,
        source_clock_domain="host_mono",
        payload=payload if payload is not None else [float(sequence_id)],
    )


class TestConstruction:
    def test_rejects_non_positive_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            StreamRingBuffer(capacity=0, tolerance_ns=1_000_000)
        with pytest.raises(ValueError, match="capacity"):
            StreamRingBuffer(capacity=-1, tolerance_ns=1_000_000)

    def test_rejects_negative_tolerance(self) -> None:
        with pytest.raises(ValueError, match="tolerance_ns"):
            StreamRingBuffer(capacity=10, tolerance_ns=-1)

    def test_zero_tolerance_is_allowed(self) -> None:
        # Some callers may want strict "exactly on target only" semantics.
        buf = StreamRingBuffer(capacity=10, tolerance_ns=0)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=100))
        sample, md = buf.get_aligned_observation(target_ns=100)
        assert sample is not None
        assert md.confidence == 1.0
        # A stale-by-one-ns pick is missing.
        sample_stale, md_stale = buf.get_aligned_observation(target_ns=101)
        assert sample_stale is None
        assert md_stale.missing is True


class TestPushAndCapacity:
    def test_push_appends_in_order(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000)
        for i in range(5):
            buf.push(_sample(sequence_id=i, acquisition_time_ns=100 * i))
        assert len(buf) == 5
        seq_ids = [s.sequence_id for s in buf]
        assert seq_ids == [0, 1, 2, 3, 4]

    def test_capacity_evicts_oldest(self) -> None:
        buf = StreamRingBuffer(capacity=3, tolerance_ns=1_000)
        for i in range(5):
            buf.push(_sample(sequence_id=i, acquisition_time_ns=100 * i))
        assert len(buf) == 3
        seq_ids = [s.sequence_id for s in buf]
        assert seq_ids == [2, 3, 4]


class TestZoHPick:
    def test_picks_latest_non_future_acquisition(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000)
        for i in range(5):
            buf.push(_sample(sequence_id=i, acquisition_time_ns=100 * i))
        # target=250: eligible acquisitions are {0,100,200}; pick 200.
        sample, md = buf.get_aligned_observation(target_ns=250)
        assert sample is not None
        assert sample.sequence_id == 2
        assert md.method == ZERO_ORDER_HOLD
        assert md.missing is False
        assert md.source_time_ns == 200
        assert md.skew_ns == -50  # source - target = 200 - 250

    def test_missing_when_target_precedes_every_sample(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=500))
        sample, md = buf.get_aligned_observation(target_ns=100)
        assert sample is None
        assert md.missing is True
        assert md.method == ZERO_ORDER_HOLD
        assert md.source_time_ns is None
        assert md.skew_ns is None
        assert md.confidence == 0.0

    def test_empty_buffer_returns_missing(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000)
        sample, md = buf.get_aligned_observation(target_ns=0)
        assert sample is None
        assert md.missing is True
        assert isinstance(md, AlignedSampleMetadata)

    def test_stale_beyond_tolerance_is_missing(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=500)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=1000))
        # target 1400 → staleness 400 <= 500 tolerance → present.
        sample_ok, md_ok = buf.get_aligned_observation(target_ns=1400)
        assert sample_ok is not None
        assert md_ok.missing is False
        # target 1600 → staleness 600 > 500 tolerance → missing but source known.
        sample_stale, md_stale = buf.get_aligned_observation(target_ns=1600)
        assert sample_stale is None
        assert md_stale.missing is True
        assert md_stale.source_time_ns == 1000  # candidate is reported
        assert md_stale.skew_ns == -600
        assert md_stale.confidence == 0.0

    def test_skew_is_always_non_positive_for_zoh(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=10_000)
        for i in range(5):
            buf.push(_sample(sequence_id=i, acquisition_time_ns=100 * i))
        for target in (0, 100, 250, 400, 401):
            _, md = buf.get_aligned_observation(target_ns=target)
            if md.skew_ns is not None:
                assert md.skew_ns <= 0

    def test_confidence_peaks_on_anchor_and_decays_to_zero_at_tolerance(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=0))
        _, md_on = buf.get_aligned_observation(target_ns=0)
        _, md_mid = buf.get_aligned_observation(target_ns=500)
        _, md_edge = buf.get_aligned_observation(target_ns=1000)
        _, md_over = buf.get_aligned_observation(target_ns=1001)
        assert md_on.confidence == 1.0
        assert md_mid.confidence == pytest.approx(0.5)
        assert md_edge.confidence == pytest.approx(0.0, abs=1e-12)
        assert md_over.missing is True


class TestCausalityInvariant:
    """At deadline 0 the pick must satisfy receive_time_ns <= target_ns."""

    def test_deadline_zero_excludes_sample_not_yet_received(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000_000)
        # A sample with acquisition_time_ns=100 but receive_time_ns=500:
        # already in the buffer (pushed after receive), but at target=200
        # (asked at wall-clock 200) it should NOT be picked, because
        # receive_time_ns=500 > target_ns=200 — the "would-have-been"
        # arrival is in the future.
        buf.push(
            _sample(sequence_id=0, acquisition_time_ns=100, receive_time_ns=500)
        )
        # Nothing older is available; must return missing.
        sample, md = buf.get_aligned_observation(target_ns=200, deadline_ns=0)
        assert sample is None
        assert md.missing is True

    def test_deadline_zero_admits_sample_received_by_target(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000_000)
        buf.push(
            _sample(sequence_id=0, acquisition_time_ns=100, receive_time_ns=150)
        )
        sample, md = buf.get_aligned_observation(target_ns=200, deadline_ns=0)
        assert sample is not None
        assert md.missing is False

    def test_positive_deadline_admits_sample_arriving_within_slack(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000_000)
        buf.push(
            _sample(sequence_id=0, acquisition_time_ns=100, receive_time_ns=250)
        )
        # deadline 0 excludes (receive 250 > target 200).
        s_no, _ = buf.get_aligned_observation(target_ns=200, deadline_ns=0)
        assert s_no is None
        # deadline 100 excludes (receive 250 > 200+50=250? actually receive 250 <= 200+100=300, admits).
        s_yes, md_yes = buf.get_aligned_observation(target_ns=200, deadline_ns=100)
        assert s_yes is not None
        assert md_yes.missing is False

    def test_deadline_still_requires_acquisition_le_target(self) -> None:
        """Deadline extends the receive-time window, not the acquisition window."""
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000_000)
        buf.push(
            _sample(sequence_id=0, acquisition_time_ns=300, receive_time_ns=300)
        )
        # Even with a very long deadline, acquisition=300 > target=200 → missing.
        sample, md = buf.get_aligned_observation(target_ns=200, deadline_ns=10_000)
        assert sample is None
        assert md.missing is True

    def test_negative_deadline_rejected(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000)
        with pytest.raises(ValueError, match="deadline_ns"):
            buf.get_aligned_observation(target_ns=100, deadline_ns=-1)


class TestGetLatestPolicyObservation:
    """Thin wrapper around get_aligned_observation(now, deadline_ns=0)."""

    def test_matches_get_aligned_observation_at_deadline_zero(self) -> None:
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000)
        for i in range(3):
            buf.push(_sample(sequence_id=i, acquisition_time_ns=100 * i))
        now = 250
        via_wrapper = buf.get_latest_policy_observation(now_ns=now)
        via_full = buf.get_aligned_observation(target_ns=now, deadline_ns=0)
        assert via_wrapper == via_full

    def test_causality_holds_at_wrapper(self) -> None:
        """Wrapper must inherit deadline-0 causality: no future receive picks."""
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000_000)
        buf.push(
            _sample(sequence_id=0, acquisition_time_ns=100, receive_time_ns=500)
        )
        sample, md = buf.get_latest_policy_observation(now_ns=200)
        assert sample is None
        assert md.missing is True

    def test_wrapper_uses_explicit_now_no_wall_clock(self) -> None:
        """Same now_ns → bit-identical result across calls (deterministic)."""
        buf = StreamRingBuffer(capacity=10, tolerance_ns=1_000)
        buf.push(_sample(sequence_id=0, acquisition_time_ns=100))
        first = buf.get_latest_policy_observation(now_ns=150)
        second = buf.get_latest_policy_observation(now_ns=150)
        assert first == second


class TestOnlineReplaySynthetic:
    """Replay a real synth stream sample-by-sample and compare to offline ZoH."""

    def test_online_zoh_matches_offline_zoh_for_causal_targets(self) -> None:
        from embodied_sync.align import align_run
        from embodied_sync.streams.synthetic import generate_synthetic_run

        run = generate_synthetic_run(duration_s=1.0, seed=0)
        # 250 Hz robot_state is the natural fast stream for online replay.
        samples = run["robot_state"]
        interval = (
            samples[1].acquisition_time_ns - samples[0].acquisition_time_ns
        )
        tolerance = interval // 2
        buf = StreamRingBuffer(capacity=len(samples), tolerance_ns=tolerance)
        for sample in samples:
            buf.push(sample)
        # Ask for target = each sample's receive_time_ns: at deadline 0
        # the causality bound bites at receive_time (not acquisition), so
        # this is the earliest causal target for each sample. Because
        # clean synth runs use a fixed transport latency (D-0006),
        # skew = acquisition - target < 0 by exactly that latency.
        latency = samples[0].receive_time_ns - samples[0].acquisition_time_ns
        for sample in samples:
            picked, md = buf.get_aligned_observation(
                target_ns=sample.receive_time_ns, deadline_ns=0
            )
            assert picked is sample
            assert md.skew_ns == -latency
            assert md.missing is False

        # Compare to offline ZoH on the same grid at 10 Hz targets. The
        # offline picker uses the full stream; online (with the whole
        # stream pushed in) should pick the same sample *when the offline
        # pick is causal by receive time* — i.e. its receive_time_ns is
        # already <= target. For online we intentionally exclude picks
        # whose receive time is past the target (causality invariant).
        offline = align_run(run, target_rate_hz=10.0, method="zoh")
        for frame in offline.frames:
            target = frame.target_time_ns
            offline_sample = frame.samples["robot_state"]
            online_sample, online_md = buf.get_aligned_observation(
                target_ns=target, deadline_ns=0
            )
            if offline_sample is None:
                assert online_sample is None
                continue
            if offline_sample.receive_time_ns > target:
                # Offline (which ignores receive_time) picked a sample
                # that hadn't arrived yet — online must NOT pick it.
                if online_sample is not None:
                    assert (
                        online_sample.acquisition_time_ns
                        <= offline_sample.acquisition_time_ns
                    )
                    assert online_sample.receive_time_ns <= target
            else:
                assert online_sample is not None
                assert (
                    online_sample.acquisition_time_ns
                    == offline_sample.acquisition_time_ns
                )
                assert online_md.missing is False
