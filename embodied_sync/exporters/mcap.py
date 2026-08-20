"""MCAP exporter (Milestone 4 synthetic contract path).

Optional-dependency discipline: this module imports at base-install
time (no top-level ``import mcap``), so ``import
embodied_sync.exporters.mcap`` succeeds without ``pip install
embodied-sync[mcap]``.

See ``docs/developer/exporter_contracts.md`` for the round-trip
contract this exporter must satisfy (byte-exact timestamp
preservation, quality-flag propagation, deterministic output).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_sync.core.episode import AlignedRun
from embodied_sync.core.sample import Sample
from embodied_sync.datasets.io import _frame_to_record, _sample_to_record

__all__ = ["save_mcap_episode", "save_mcap_run"]

_FORMAT = "embodied_sync.mcap.contract.v0"


def save_mcap_run(
    run: dict[str, list[Sample]], path: str | Path
) -> None:
    """Serialise a run of ``Sample``s to a deterministic MCAP contract file.

    The base-install CI path writes a small canonical JSON document with an
    ``.mcap``-friendly API surface. It is intentionally deterministic and
    lossless for the internal model, so adapter/exporter contract tests do not
    depend on ROS2 or the optional ``mcap`` wheel. External true-MCAP tests can
    layer on top of the same public function when the extra is installed.
    """
    document: dict[str, Any] = {
        "format": _FORMAT,
        "type": "run",
        "streams": {
            name: [_sample_to_record(sample) for sample in samples]
            for name, samples in run.items()
        },
    }
    _write_document(document, path)


def save_mcap_episode(
    episode: AlignedRun, path: str | Path
) -> None:
    """Serialise an aligned episode to a deterministic MCAP contract file.

    The run adapter intentionally reads only ``type == "run"`` documents.
    Episode export is present so downstream tools can archive aligned frames
    and reports through the same deterministic interchange surface.
    """
    document: dict[str, Any] = {
        "format": _FORMAT,
        "type": "aligned_episode",
        "frames": [_frame_to_record(frame) for frame in episode.frames],
        "report": {
            "missing_count": episode.report.missing_count,
            "ground_truth_missing_count": episode.report.ground_truth_missing_count,
            "median_skew_ns": episode.report.median_skew_ns,
        },
    }
    _write_document(document, path)


def _write_document(document: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )
