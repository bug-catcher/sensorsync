"""Clock epochs: a reconnect invalidates fits instead of poisoning them (A3).

The failure this machinery prevents is specific and nasty. A device
reconnects mid-session and its hardware counter restarts. Timestamps
either side of that discontinuity are in different timelines, but they
are still monotone-ish numbers, so every downstream fit happily
straddles the break: small residuals, high inlier fraction, and a drift
figure that is entirely an artefact of the jump. Nothing looks wrong.

So the tests here check two properties: the epoch counter itself
behaves (monotone, per-domain, never rewindable), and the session
*acts* on it — dropping mappings, clearing buffers, refusing stale
registrations.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from embodied_sync.session import CLOCK_EPOCH_ADVANCED, StreamConfig, SyncSession
from embodied_sync.session.violations import SyncViolation
from embodied_sync.time.clock_domain import (
    INITIAL_EPOCH,
    ClockDomain,
    ClockEpochError,
    ClockEpochRegistry,
    ClockKind,
    LatencyEstimate,
    require_same_epoch,
    translate_ns,
)

MS = 1_000_000
S = 1_000_000_000


class FakeClock:
    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self._now = start_ns
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._now

    def set(self, ns: int) -> None:
        with self._lock:
            self._now = ns


def _foreign() -> ClockDomain:
    return ClockDomain("cam_hw", ClockKind.HARDWARE)


def _host() -> ClockDomain:
    return ClockDomain("host_mono", ClockKind.MONOTONIC)


def _mapping(offset_ns: int = 1000, epoch: int = 0) -> LatencyEstimate:
    return LatencyEstimate(
        source=_foreign(), target=_host(), offset_ns=offset_ns, epoch=epoch
    )


class TestRegistry:
    def test_domains_start_in_the_initial_epoch(self) -> None:
        registry = ClockEpochRegistry()
        assert registry.current("anything") == INITIAL_EPOCH

    def test_advance_increments_and_records(self) -> None:
        registry = ClockEpochRegistry()
        record = registry.advance("cam_hw", reason="reconnect", at_ns=5 * S)
        assert record.epoch == 1
        assert record.reason == "reconnect"
        assert record.started_at_ns == 5 * S
        assert registry.current("cam_hw") == 1

    def test_epochs_are_per_domain(self) -> None:
        registry = ClockEpochRegistry()
        registry.advance("cam_hw")
        assert registry.current("cam_hw") == 1
        assert registry.current("imu_hw") == INITIAL_EPOCH

    def test_history_is_bounded_and_ordered(self) -> None:
        registry = ClockEpochRegistry(history_limit=3)
        for i in range(10):
            registry.advance("cam_hw", reason=f"r{i}")
        history = registry.history("cam_hw")
        assert len(history) == 3
        assert [record.epoch for record in history] == [8, 9, 10]

    def test_was_reset_compares_against_a_caller_held_generation(self) -> None:
        registry = ClockEpochRegistry()
        assert not registry.was_reset("cam_hw", INITIAL_EPOCH)
        registry.advance("cam_hw")
        assert registry.was_reset("cam_hw", INITIAL_EPOCH)
        assert not registry.was_reset("cam_hw", 1)

    def test_is_current_checks_a_mapping(self) -> None:
        registry = ClockEpochRegistry()
        mapping = _mapping()
        assert registry.is_current(mapping)
        registry.advance("cam_hw")
        assert not registry.is_current(mapping)

    def test_snapshot_and_domains(self) -> None:
        registry = ClockEpochRegistry()
        registry.advance("cam_hw")
        registry.advance("imu_hw")
        registry.advance("cam_hw")
        assert registry.domains() == ("cam_hw", "imu_hw")
        assert registry.snapshot() == {"cam_hw": 2, "imu_hw": 1}

    def test_history_limit_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="history_limit"):
            ClockEpochRegistry(history_limit=0)


class TestLatencyEstimateEpoch:
    def test_defaults_to_the_initial_epoch(self) -> None:
        assert _mapping().epoch == INITIAL_EPOCH

    def test_with_epoch_restamps_without_changing_the_fit(self) -> None:
        mapping = LatencyEstimate(
            source=_foreign(),
            target=_host(),
            offset_ns=1234,
            drift_ppb=99,
            anchor_time_ns=7,
            variance_ns=5,
        )
        restamped = mapping.with_epoch(4)
        assert restamped.epoch == 4
        assert translate_ns(1000, restamped) == translate_ns(1000, mapping)
        assert mapping.epoch == INITIAL_EPOCH, "with_epoch must not mutate"

    def test_negative_epochs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="epoch must be"):
            LatencyEstimate(source=_foreign(), target=_host(), offset_ns=0, epoch=-1)

    def test_non_integer_epochs_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="epoch must be int"):
            LatencyEstimate(
                source=_foreign(), target=_host(), offset_ns=0, epoch=1.5  # type: ignore[arg-type]
            )


class TestTranslateGuard:
    def test_matching_epoch_translates(self) -> None:
        assert translate_ns(100, _mapping(offset_ns=5), epoch=0) == 105

    def test_mismatched_epoch_raises_rather_than_returning_a_number(self) -> None:
        with pytest.raises(ClockEpochError) as excinfo:
            translate_ns(100, _mapping(offset_ns=5), epoch=1)
        assert excinfo.value.epochs == (0, 1)

    def test_omitting_the_epoch_keeps_the_old_behaviour(self) -> None:
        assert translate_ns(100, _mapping(offset_ns=5, epoch=3)) == 105


class TestRequireSameEpoch:
    def test_single_epoch_passes_through(self) -> None:
        assert require_same_epoch([2, 2, 2]) == 2

    def test_empty_is_the_initial_epoch(self) -> None:
        assert require_same_epoch([]) == INITIAL_EPOCH

    def test_mixed_epochs_raise_with_the_offenders_attached(self) -> None:
        with pytest.raises(ClockEpochError) as excinfo:
            require_same_epoch([0, 0, 1], context="calibration pairs")
        assert excinfo.value.epochs == (0, 1)
        assert "calibration pairs" in str(excinfo.value)


class TestSessionEpochs:
    def _session(self, clock: FakeClock, **kwargs: object) -> SyncSession:
        return SyncSession(
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=20.0, clock_domain="cam_hw"
                ),
                "depth": StreamConfig(
                    rate_hz=10, tolerance_ms=20.0, clock_domain="cam_hw"
                ),
                "robot": StreamConfig(rate_hz=10, tolerance_ms=20.0),
            },
            clock=clock,
            on_violation="ignore",
            **kwargs,  # type: ignore[arg-type]
        )

    def test_epoch_starts_at_zero_and_advances(self) -> None:
        session = self._session(FakeClock())
        assert session.clock_epoch("camera") == 0
        assert session.mark_clock_reset("camera") == 1
        assert session.clock_epoch("camera") == 1
        session.close()

    def test_streams_sharing_a_domain_share_its_epoch(self) -> None:
        session = self._session(FakeClock())
        session.mark_clock_reset("camera")
        assert session.clock_epoch("depth") == 1
        assert session.clock_epoch("robot") == 0
        session.close()

    def test_reset_drops_the_registered_mapping(self) -> None:
        clock = FakeClock()
        session = self._session(clock)
        session.register_clock_mapping(_mapping(offset_ns=999))
        clock.set(2 * S)
        session.push("camera", {}, t_ns=42)
        assert "clock_mapped" in session.push("camera", {}, t_ns=43).quality_flags
        session.mark_clock_reset("camera")
        # No mapping any more: the sample is flagged unmapped, not translated
        # with a mapping fitted against a dead timeline.
        after = session.push("camera", {}, t_ns=44)
        assert "unmapped_clock_domain" in after.quality_flags
        session.close()

    def test_reset_clears_the_buffer_so_old_times_cannot_be_matched(self) -> None:
        clock = FakeClock()
        session = self._session(clock)
        clock.set(2 * S)
        session.push("robot", {}, t_ns=2 * S)
        assert len(session.buffer("robot")) == 1
        session.mark_clock_reset("robot", reason="power cycle")
        assert len(session.buffer("robot")) == 0
        session.close()

    def test_reset_clears_the_monotonicity_tracker(self) -> None:
        """A restarting counter legitimately goes backwards; that is not a fault."""
        clock = FakeClock()
        seen: list[SyncViolation] = []
        session = SyncSession(
            streams={"robot": StreamConfig(rate_hz=10, tolerance_ms=20.0)},
            clock=clock,
            on_violation=seen.append,
        )
        clock.set(5 * S)
        session.push("robot", {}, t_ns=5 * S)
        session.mark_clock_reset("robot")
        seen.clear()
        session.push("robot", {}, t_ns=0)  # counter restarted at zero
        assert [v.reason for v in seen] == []
        session.close()

    def test_reset_emits_a_violation_per_affected_stream(self) -> None:
        clock = FakeClock()
        seen: list[SyncViolation] = []
        session = SyncSession(
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=20.0, clock_domain="cam_hw"
                ),
                "depth": StreamConfig(
                    rate_hz=10, tolerance_ms=20.0, clock_domain="cam_hw"
                ),
            },
            clock=clock,
            on_violation=seen.append,
        )
        session.mark_clock_reset("camera", reason="usb re-enumeration")
        assert {v.stream for v in seen} == {"camera", "depth"}
        assert {v.reason for v in seen} == {CLOCK_EPOCH_ADVANCED}
        assert "usb re-enumeration" in seen[0].message
        session.close()

    def test_a_stale_mapping_is_refused(self) -> None:
        session = self._session(FakeClock())
        session.mark_clock_reset("camera")
        with pytest.raises(ValueError, match="fitted in epoch 0 but the domain"):
            session.register_clock_mapping(_mapping(epoch=0))
        session.close()

    def test_a_current_mapping_is_accepted_after_a_reset(self) -> None:
        session = self._session(FakeClock())
        epoch = session.mark_clock_reset("camera")
        session.register_clock_mapping(_mapping(offset_ns=7).with_epoch(epoch))
        assert session.time_correction("camera").epoch == epoch
        session.close()

    def test_was_clock_reset_consumes_the_notification(self) -> None:
        session = self._session(FakeClock())
        assert session.was_clock_reset("camera") is False  # first read: baseline
        assert session.was_clock_reset("camera") is False
        session.mark_clock_reset("camera")
        assert session.was_clock_reset("camera") is True
        assert session.was_clock_reset("camera") is False
        session.close()

    def test_clock_epochs_lists_every_declared_domain(self) -> None:
        session = self._session(FakeClock())
        session.mark_clock_reset("camera")
        assert session.clock_epochs() == {"cam_hw": 1, "host_mono": 0}
        session.close()

    def test_manifest_records_epochs_and_their_history(self, tmp_path: Path) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        session = self._session(clock, run_dir=run_dir)
        clock.set(3 * S)
        session.mark_clock_reset("camera", reason="reconnect")
        session.close()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["session"]["clock_epochs"]["cam_hw"] == 1
        history = manifest["session"]["clock_epoch_history"]["cam_hw"]
        assert [entry["epoch"] for entry in history] == [0, 1]
        assert history[1]["reason"] == "reconnect"
        assert history[1]["started_at_ns"] == 3 * S

    def test_manifest_mapping_carries_its_epoch(self, tmp_path: Path) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        session = self._session(clock, run_dir=run_dir)
        epoch = session.mark_clock_reset("camera")
        session.register_clock_mapping(_mapping(offset_ns=55).with_epoch(epoch))
        session.close()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        recorded = manifest["session"]["clock_mappings"]["cam_hw"]
        assert recorded["epoch"] == 1
        assert recorded["offset_ns"] == 55
