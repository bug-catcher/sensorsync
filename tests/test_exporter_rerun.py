"""Milestone 8: Rerun exporter with HTML fallback.

The base install rarely has ``rerun`` installed, so
:func:`save_rerun_episode` must be usable regardless — either by
detecting the absence and emitting HTML, or by ``force_html=True``.
This test module covers the fallback path (CI-friendly, no optional
dependency needed) and the module-import discipline.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from embodied_sync.align import align_run
from embodied_sync.exporters.rerun import (
    is_rerun_available,
    save_html_fallback,
    save_rerun_episode,
)
from embodied_sync.streams.synthetic import generate_synthetic_run


def _episode():
    run = generate_synthetic_run(duration_s=1.0, seed=0)
    return align_run(run, target_rate_hz=10.0)


def test_force_html_writes_a_self_contained_page(tmp_path: Path) -> None:
    output = tmp_path / "episode.rrd"
    outcome = save_rerun_episode(_episode(), output, force_html=True)
    assert outcome == "html"
    html_path = output.with_suffix(".html")
    assert html_path.is_file()
    text = html_path.read_text(encoding="utf-8")
    assert "<html" in text.lower()
    # No external references — the sync-quality report is
    # self-contained by construction.
    assert "http://" not in text
    assert "cdn." not in text


def test_save_html_fallback_directly(tmp_path: Path) -> None:
    output = tmp_path / "fallback.html"
    save_html_fallback(_episode(), output)
    assert output.is_file()
    assert output.read_text(encoding="utf-8").lower().startswith("<!doctype html>")


def test_rerun_availability_matches_importlib() -> None:
    assert is_rerun_available() == (importlib.util.find_spec("rerun") is not None)


def test_default_call_uses_fallback_when_rerun_absent(tmp_path: Path) -> None:
    if is_rerun_available():
        pytest.skip("rerun is installed; force_html covers the fallback")
    output = tmp_path / "episode.rrd"
    outcome = save_rerun_episode(_episode(), output)
    assert outcome == "html"
    assert output.with_suffix(".html").is_file()
    assert not output.exists()  # .rrd was NOT written


def _forget_rerun_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "rerun" or module_name.startswith("rerun."):
            del sys.modules[module_name]


@pytest.mark.optional_dep
def test_save_rerun_episode_round_trips_stream_entity_paths(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    _forget_rerun_modules()
    output = tmp_path / "episode.rrd"

    script = textwrap.dedent(
        """
        import subprocess
        import sys
        from pathlib import Path

        import pytest

        pytest.importorskip("rerun")

        from embodied_sync.align import align_run
        from embodied_sync.exporters.rerun import save_rerun_episode
        from embodied_sync.streams.synthetic import generate_synthetic_run

        output = Path(sys.argv[1])
        episode = align_run(
            generate_synthetic_run(duration_s=1.0, seed=0),
            target_rate_hz=10.0,
        )
        outcome = save_rerun_episode(episode, output)
        assert outcome == "rerun"
        assert output.is_file()
        assert output.stat().st_size > 0

        rerun_cli = Path(sys.executable).with_name("rerun")
        printed = subprocess.run(
            [str(rerun_cli), "rrd", "print", str(output)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        for name in episode.frames[0].samples:
            assert f"/streams/{name}/payload" in printed
            assert f"/streams/{name}/missing" in printed
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(output)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _forget_rerun_modules()
    assert result.returncode == 0, result.stdout + result.stderr
