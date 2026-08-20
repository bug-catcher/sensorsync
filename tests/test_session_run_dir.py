"""The live-to-offline contract: a recorded session is a valid run (D-0037).

The point of these tests is one claim: what ``SyncSession`` writes,
``load_run`` reads and ``align_run`` aligns — no conversion step, no
second format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import embodied_sync as embsync
from embodied_sync.align import align_run
from embodied_sync.core.sample import Modality
from embodied_sync.datasets.io import FORMAT_VERSION, load_run
from embodied_sync.session import SESSION_QUALITY_NAME, StreamConfig, SyncSession

from test_session_end_to_end import FakeClock

MS = 1_000_000


def _record(session: SyncSession, clock: FakeClock, *, frames: int = 20) -> None:
    base = 1_000_000_000
    for i in range(frames * 10):
        t = base + i * 10 * MS
        clock.set(t)
        session.push("robot", {"q": [float(i)]}, t_ns=t)
        if i % 10 == 0:
            session.push("camera", {"frame": i // 10}, t_ns=t)


class TestRunDirRoundTrip:
    def test_recorded_session_loads_with_load_run(
        self, tmp_path: Path
    ) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        with embsync.init(
            run_dir=run_dir,
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=60.0, modality="camera"
                ),
                "robot": StreamConfig(
                    rate_hz=100, tolerance_ms=6.0, modality="robot_state"
                ),
            },
            primary="camera",
            clock=clock,
        ) as session:
            _record(session, clock)

        run = load_run(run_dir)
        assert set(run) == {"camera", "robot"}
        assert len(run["camera"]) == 20
        assert len(run["robot"]) == 200
        assert run["camera"][0].modality is Modality.CAMERA
        assert run["robot"][0].modality is Modality.ROBOT_STATE
        # Timestamps survive as exact integers.
        assert run["robot"][0].acquisition_time_ns == 1_000_000_000
        assert run["robot"][-1].acquisition_time_ns == 1_000_000_000 + 1990 * MS
        assert all(isinstance(s.acquisition_time_ns, int) for s in run["robot"])
        # Sequence ids are contiguous from zero.
        assert [s.sequence_id for s in run["camera"]] == list(range(20))

    def test_recorded_run_aligns_with_align_run(self, tmp_path: Path) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        with embsync.init(
            run_dir=run_dir,
            streams={
                "camera": StreamConfig(rate_hz=10, tolerance_ms=60.0),
                "robot": StreamConfig(rate_hz=100, tolerance_ms=6.0),
            },
            primary="camera",
            clock=clock,
        ) as session:
            _record(session, clock)

        aligned = align_run(load_run(run_dir), target_rate_hz=10.0)
        assert aligned.frames
        assert aligned.report.missing_count == {"camera": 0, "robot": 0}

    def test_manifest_is_run_format_v0_plus_a_session_block(
        self, tmp_path: Path
    ) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        with embsync.init(
            run_dir=run_dir,
            streams={"robot": StreamConfig(rate_hz=100, tolerance_ms=6.0)},
            primary="robot",
            clock=clock,
        ) as session:
            session.push("robot", {"q": 1}, t_ns=1_000_000_000)

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["format_version"] == FORMAT_VERSION
        assert manifest["streams"]["robot"]["sample_count"] == 1
        assert manifest["streams"]["robot"]["clock_domains"] == ["host_mono"]
        assert manifest["streams"]["robot"]["policy"] == "latest_before"
        assert manifest["session"]["primary"] == "robot"
        assert manifest["session"]["clock_domain"] == "host_mono"

    def test_manifest_exists_before_any_sample_arrives(self, tmp_path: Path) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        session = embsync.init(
            run_dir=run_dir,
            streams={"robot": StreamConfig(rate_hz=100, tolerance_ms=6.0)},
            clock=clock,
        )
        assert (run_dir / "manifest.json").is_file()
        assert load_run(run_dir) == {"robot": []}
        session.close()

    def test_flush_makes_partial_results_readable_mid_session(
        self, tmp_path: Path
    ) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        session = embsync.init(
            run_dir=run_dir,
            streams={"robot": StreamConfig(rate_hz=100, tolerance_ms=6.0)},
            clock=clock,
        )
        for i in range(5):
            session.push("robot", {"i": i}, t_ns=1_000_000_000 + i * 10 * MS)
        session.flush()
        assert len(load_run(run_dir)["robot"]) == 5
        session.close()

    def test_run_dir_is_optional(self, tmp_path: Path) -> None:
        clock = FakeClock()
        session = embsync.init(
            streams={"robot": StreamConfig(rate_hz=100, tolerance_ms=6.0)},
            clock=clock,
        )
        assert session.run_dir is None
        session.push("robot", {"i": 0}, t_ns=1_000_000_000)
        session.close()
        assert list(tmp_path.iterdir()) == []


class TestPersistModes:
    def test_metadata_mode_nulls_the_payload(self, tmp_path: Path) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        with embsync.init(
            run_dir=run_dir,
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=60.0)},
            clock=clock,
        ) as session:
            session.push("camera", {"huge": [0] * 1000}, t_ns=1_000_000_000)
        sample = load_run(run_dir)["camera"][0]
        assert sample.payload is None
        assert sample.acquisition_time_ns == 1_000_000_000

    def test_full_mode_writes_the_payload(self, tmp_path: Path) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        with embsync.init(
            run_dir=run_dir,
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=60.0, persist="full"
                )
            },
            clock=clock,
        ) as session:
            session.push("camera", {"frame": 3}, t_ns=1_000_000_000)
        assert load_run(run_dir)["camera"][0].payload == {"frame": 3}

    def test_off_mode_omits_the_stream_entirely(self, tmp_path: Path) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        with embsync.init(
            run_dir=run_dir,
            streams={
                "camera": StreamConfig(rate_hz=10, tolerance_ms=60.0),
                "video": StreamConfig(
                    rate_hz=10, tolerance_ms=60.0, persist="off"
                ),
            },
            clock=clock,
        ) as session:
            session.push("camera", {"frame": 0}, t_ns=1_000_000_000)
            session.push("video", object(), t_ns=1_000_000_000)

        run = load_run(run_dir)
        assert set(run) == {"camera"}
        assert not (run_dir / "streams" / "video.jsonl").exists()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["session"]["omitted_streams"] == ["video"]

    def test_unserializable_payload_points_at_the_serialize_hook(
        self, tmp_path: Path
    ) -> None:
        clock = FakeClock()
        session = embsync.init(
            run_dir=tmp_path / "run",
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=60.0, persist="full"
                )
            },
            clock=clock,
        )
        with pytest.raises(TypeError, match="serialize="):
            session.push("camera", object(), t_ns=1_000_000_000)
        session.close()

    def test_serialize_hook_rescues_an_opaque_payload(self, tmp_path: Path) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"

        class Frame:
            def __init__(self, index: int) -> None:
                self.index = index

        def serialize(stream: str, payload: Any) -> Any:
            return {"stream": stream, "index": payload.index}

        with embsync.init(
            run_dir=run_dir,
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=60.0, persist="full"
                )
            },
            clock=clock,
            serialize=serialize,
        ) as session:
            session.push("camera", Frame(7), t_ns=1_000_000_000)

        assert load_run(run_dir)["camera"][0].payload == {
            "stream": "camera",
            "index": 7,
        }

    def test_refuses_to_overwrite_an_existing_recording(self, tmp_path: Path) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        with embsync.init(
            run_dir=run_dir,
            streams={"robot": StreamConfig(rate_hz=100, tolerance_ms=6.0)},
            clock=clock,
        ) as session:
            session.push("robot", {"i": 0}, t_ns=1_000_000_000)
        with pytest.raises(FileExistsError, match="existing recorded stream"):
            embsync.init(
                run_dir=run_dir,
                streams={"robot": StreamConfig(rate_hz=100, tolerance_ms=6.0)},
                clock=clock,
            )


class TestSessionQualitySidecar:
    def test_close_writes_the_quality_snapshot(self, tmp_path: Path) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        with embsync.init(
            run_dir=run_dir,
            streams={
                "camera": StreamConfig(rate_hz=10, tolerance_ms=60.0),
                "robot": StreamConfig(rate_hz=100, tolerance_ms=6.0),
            },
            primary="camera",
            clock=clock,
        ) as session:
            _record(session, clock)
            session.get()

        payload = json.loads((run_dir / SESSION_QUALITY_NAME).read_text())
        assert payload["type"] == "session_quality"
        assert set(payload["streams"]) == {"camera", "robot"}
        robot = payload["streams"]["robot"]
        assert robot["expected_rate_hz"] == 100
        assert robot["observed_rate_hz"] == pytest.approx(100.0)
        assert robot["match_count"] == 1
        assert robot["problems"] == []

    def test_sidecar_does_not_disturb_load_run(self, tmp_path: Path) -> None:
        clock = FakeClock()
        run_dir = tmp_path / "run"
        with embsync.init(
            run_dir=run_dir,
            streams={"robot": StreamConfig(rate_hz=100, tolerance_ms=6.0)},
            clock=clock,
        ) as session:
            session.push("robot", {"i": 0}, t_ns=1_000_000_000)
        assert (run_dir / SESSION_QUALITY_NAME).is_file()
        assert len(load_run(run_dir)["robot"]) == 1
