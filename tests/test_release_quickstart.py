"""Regression test for the commands shown in the published README quickstart."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_pypi_quickstart_commands_run_cleanly(tmp_path: Path) -> None:
    """Exercise the install-facing synth → align → report path in isolation."""
    clean = tmp_path / "runs" / "clean"
    episode = tmp_path / "episodes" / "clean"
    html = tmp_path / "reports" / "clean.html"
    summary = tmp_path / "reports" / "clean.json"

    commands = [
        ["synth", "--out", str(clean), "--seed", "0", "--duration-s", "2"],
        ["align", str(clean), "--out", str(episode), "--target-rate-hz", "10"],
        [
            "report",
            str(episode),
            "--out",
            str(html),
            "--json-summary",
            str(summary),
        ],
    ]
    for command in commands:
        result = subprocess.run(
            [sys.executable, "-m", "embodied_sync.cli.main", *command],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    assert html.is_file()
    assert summary.is_file()
