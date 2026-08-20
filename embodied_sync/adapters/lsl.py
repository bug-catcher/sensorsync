"""LSL/XDF adapter: native LabRecorder XDF reader + replay contract path.

Two entry points:

- :func:`load_xdf_file` reads a real LabRecorder ``.xdf`` session via
  ``pyxdf``. The optional dependency is imported lazily inside the
  function. By default the reader asks ``pyxdf`` for raw, pre-correction
  timestamps (``synchronize_clocks=False``, ``dejitter_timestamps=False``)
  and converts those seconds to integer nanoseconds with ``round(t * 1e9)``.
  XDF clock-offset measurements are preserved in ``dataset_info`` rather
  than applied to samples, so sync-quality analysis can inspect the raw
  timing and the recorder's correction metadata side by side (D-0034).
- :func:`load_lsl_replay` reads the deterministic JSON contract fixture
  used by CI without importing optional dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_sync.core.sample import Modality, Sample

__all__ = ["load_lsl_replay", "load_xdf_file", "open_lsl_inlet"]

_FORMAT = "embodied_sync.lsl.replay.v0"


def _lsl_modality(stream_type: str) -> Modality:
    normalized = stream_type.strip().lower()
    if normalized in {"markers", "marker", "events", "event"}:
        return Modality.EVENT
    if normalized in {"camera", "camera_features", "video", "image"}:
        return Modality.CAMERA
    if normalized in {"joint_state", "robot_state", "kinematics"}:
        return Modality.ROBOT_STATE
    if normalized in {"tactile", "force", "pressure"}:
        return Modality.TACTILE
    if normalized in {"audio", "audio_features"}:
        return Modality.AUDIO
    return Modality.OTHER


def _first_info_value(info: dict[str, Any], key: str, default: object = "") -> object:
    value = info.get(key, default)
    if isinstance(value, list) and value:
        return value[0]
    return value


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (str, bytes, bytearray)):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return default


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (str, bytes, bytearray)):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _clock_offset_table(stream: dict[str, Any]) -> list[dict[str, int]]:
    """Return XDF clock offsets as integer-ns ``{"time_ns", "offset_ns"}`` rows."""
    times = stream.get("clock_times") or []
    values = stream.get("clock_values") or []
    table: list[dict[str, int]] = []
    for t, value in zip(times, values):
        table.append(
            {
                "time_ns": round(float(t) * 1e9),
                "offset_ns": round(float(value) * 1e9),
            }
        )
    if table:
        return table

    footer_offsets = (
        stream.get("footer", {})
        .get("info", {})
        .get("clock_offsets", [])
    )
    for entry in footer_offsets:
        offsets = entry.get("offset", []) if isinstance(entry, dict) else []
        for offset in offsets:
            if not isinstance(offset, dict):
                continue
            time_values = offset.get("time") or []
            value_values = offset.get("value") or []
            if not time_values or not value_values:
                continue
            table.append(
                {
                    "time_ns": round(float(time_values[0]) * 1e9),
                    "offset_ns": round(float(value_values[0]) * 1e9),
                }
            )
    return table


def _series_value(series: object, index: int) -> object:
    value = series[index]  # type: ignore[index]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [v.item() if hasattr(v, "item") else v for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def load_xdf_file(
    path: str | Path,
    *,
    synchronize_clocks: bool = False,
    dejitter_timestamps: bool = False,
) -> tuple[dict[str, list[Sample]], dict[str, Any]]:
    """Load a LabRecorder XDF file as a canonical run.

    Returns ``(run, dataset_info)``. Each XDF stream becomes one run
    stream. Floating-point XDF timestamps are converted to integer
    nanoseconds with ``round(t * 1e9)`` and are not regridded. The
    default preserves raw pre-correction timing; callers may opt into
    ``pyxdf`` clock synchronization or dejittering explicitly.
    """
    import pyxdf  # noqa: PLC0415

    xdf_path = Path(path)
    streams, header = pyxdf.load_xdf(
        str(xdf_path),
        synchronize_clocks=synchronize_clocks,
        dejitter_timestamps=dejitter_timestamps,
    )

    run: dict[str, list[Sample]] = {}
    stream_info: list[dict[str, Any]] = []
    for stream_index, stream in enumerate(streams):
        info = stream["info"]
        name = str(_first_info_value(info, "name", f"stream_{stream_index}"))
        stream_type = str(_first_info_value(info, "type", ""))
        uid = str(
            _first_info_value(
                info,
                "uid",
                _first_info_value(info, "source_id", name),
            )
        )
        modality = _lsl_modality(stream_type)
        stamps = stream["time_stamps"]
        series = stream["time_series"]
        samples: list[Sample] = []
        for i, stamp in enumerate(stamps):
            ts_ns = round(float(stamp) * 1e9)
            samples.append(
                Sample(
                    stream_name=name,
                    modality=modality,
                    sequence_id=i,
                    acquisition_time_ns=ts_ns,
                    receive_time_ns=ts_ns,
                    source_clock_domain=uid or name,
                    payload=_series_value(series, i),
                )
            )
        run[name] = samples
        stream_info.append(
            {
                "name": name,
                "type": stream_type,
                "stream_id": info.get("stream_id"),
                "uid": uid,
                "channel_count": _as_int(_first_info_value(info, "channel_count", 0)),
                "nominal_srate": _as_float(_first_info_value(info, "nominal_srate", 0.0)),
                "effective_srate": _as_float(info.get("effective_srate", 0.0)),
                "samples": len(samples),
                "clock_offsets": _clock_offset_table(stream),
            }
        )

    dataset_info: dict[str, Any] = {
        "source_path": str(xdf_path),
        "format": "xdf",
        "timestamp_mode": {
            "synchronize_clocks": synchronize_clocks,
            "dejitter_timestamps": dejitter_timestamps,
            "units": "integer_ns_from_xdf_seconds",
        },
        "header": header,
        "streams": stream_info,
    }
    return run, dataset_info


def load_lsl_replay(path: str | Path) -> dict[str, list[Sample]]:
    """Load a deterministic LSL/XDF-like replay JSON fixture.

    Timestamps in the fixture are integer nanoseconds after XDF clock-offset
    correction. The original offset metadata is preserved in each sample's
    payload under ``xdf_clock_offset_ns`` when present.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != _FORMAT:
        raise ValueError(
            f"unsupported LSL replay format {document.get('format')!r} in {path}"
        )
    streams = document.get("streams")
    if not isinstance(streams, dict):
        raise ValueError(f"invalid LSL replay in {path}: missing streams")

    run: dict[str, list[Sample]] = {}
    for name, stream in streams.items():
        modality = Modality(stream.get("modality", Modality.OTHER.value))
        clock_domain = str(stream.get("source_clock_domain", "lsl"))
        clock_offset_ns = stream.get("xdf_clock_offset_ns")
        records = stream.get("samples")
        if not isinstance(records, list):
            raise ValueError(f"invalid LSL stream {name!r}: missing samples")
        samples: list[Sample] = []
        for index, record in enumerate(records):
            payload = record.get("payload")
            if clock_offset_ns is not None:
                payload = {
                    "value": payload,
                    "xdf_clock_offset_ns": int(clock_offset_ns),
                }
            samples.append(
                Sample(
                    stream_name=name,
                    modality=modality,
                    sequence_id=int(record.get("sequence_id", index)),
                    acquisition_time_ns=int(record["acquisition_time_ns"]),
                    receive_time_ns=int(record.get("receive_time_ns", record["acquisition_time_ns"])),
                    source_clock_domain=clock_domain,
                    payload=payload,
                    payload_ref=record.get("payload_ref"),
                    quality_flags=frozenset(record.get("quality_flags", ())),
                )
            )
        run[name] = samples
    return run


def open_lsl_inlet(stream_name: str) -> object:
    """Open a live LSL inlet.

    Live streaming is intentionally optional. Calling this function requires
    ``pip install embodied-sync[lsl]`` and raises the standard ``ImportError``
    when ``pylsl`` is absent.
    """
    import pylsl  # noqa: PLC0415

    matches = pylsl.resolve_byprop("name", stream_name, timeout=1.0)
    if not matches:
        raise ValueError(f"no LSL stream named {stream_name!r} found")
    return pylsl.StreamInlet(matches[0])
