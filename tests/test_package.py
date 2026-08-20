"""Baseline green tests: package import, CLI placeholder, Sample contracts.

These must stay green so the suite distinguishes "designed but unimplemented"
(the red Milestone 1 tests) from "broken".
"""

from __future__ import annotations

import pytest

import embodied_sync
from embodied_sync import Modality, Sample
from embodied_sync.cli.main import build_parser, main


def test_version_string() -> None:
    assert isinstance(embodied_sync.__version__, str)
    assert embodied_sync.__version__


def test_import_pulls_no_optional_dependencies() -> None:
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        import embodied_sync  # noqa: F401

        for forbidden in ("mcap", "lerobot", "pylsl", "pyxdf", "zarr", "numcodecs", "rerun"):
            assert forbidden not in sys.modules, (
                f"importing embodied_sync must not import optional dependency "
                f"{forbidden!r}"
            )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


class TestSampleTimestampPreservation:
    """A Sample must hold exactly the integers it was given (D-0002/D-0003)."""

    def test_fields_preserved_exactly(self) -> None:
        s = Sample(
            stream_name="cam_front",
            modality=Modality.CAMERA,
            sequence_id=41,
            acquisition_time_ns=1_699_999_999_123_456_789,
            receive_time_ns=1_699_999_999_135_456_789,
            source_clock_domain="cam_front_hw",
            payload={"frame_index": 41},
            quality_flags=frozenset({"synthetic"}),
        )
        assert s.acquisition_time_ns == 1_699_999_999_123_456_789
        assert s.receive_time_ns == 1_699_999_999_135_456_789
        assert s.transport_latency_ns == 12_000_000
        assert s.source_clock_domain == "cam_front_hw"
        assert s.sequence_id == 41
        assert s.quality_flags == frozenset({"synthetic"})

    @pytest.mark.parametrize("bad", [1.5, 1e18, True, "0", None])
    def test_non_int_timestamps_rejected(self, bad: object) -> None:
        with pytest.raises(TypeError):
            Sample(
                stream_name="s",
                modality=Modality.OTHER,
                sequence_id=0,
                acquisition_time_ns=bad,  # type: ignore[arg-type]
                receive_time_ns=0,
                source_clock_domain="host_mono",
            )

    def test_samples_are_immutable(self) -> None:
        s = Sample(
            stream_name="s",
            modality=Modality.OTHER,
            sequence_id=0,
            acquisition_time_ns=0,
            receive_time_ns=0,
            source_clock_domain="host_mono",
        )
        with pytest.raises(AttributeError):
            s.acquisition_time_ns = 1  # type: ignore[misc]


class TestCliPlaceholder:
    def test_help_lists_planned_subcommands(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == 0
        out = capsys.readouterr().out
        for cmd in ("synth", "corrupt", "align", "report"):
            assert cmd in out

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--version"])
        assert excinfo.value.code == 0
        assert embodied_sync.__version__ in capsys.readouterr().out
