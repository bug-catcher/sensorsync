"""QUT dataset-example adapter.

The Hugging Face ``nmarticorena/dataset_example`` layout stores robot state
rows under ``episodes/<id>/state.json`` and camera frames in ``images.h5`` as
``/<episode>/<camera>[frame]``.  The HDF5 image arrays and MP4s have one 10 Hz
frame per state row, making row index the dataset's synchronized camera/state
contract.  The irregular ``time`` field is retained as source metadata but is
not the media clock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_sync.core.sample import Modality, Sample

__all__ = ["QUT_CLOCK_DOMAIN", "load_qut_dataset"]

QUT_CLOCK_DOMAIN = "qut_dataset_example"
QUT_SOURCE_RATE_HZ = 10.0

_STATE_STREAMS = {
    "robot_q": Modality.ROBOT_STATE,
    "robot_X_BE": Modality.ROBOT_STATE,
    "gello_q": Modality.ACTION,
    "gripper_action": Modality.ACTION,
    "gripper_width": Modality.ROBOT_STATE,
    "K_F_ext_hat_K": Modality.TACTILE,
    "O_F_ext_hat_K": Modality.TACTILE,
    "O_T_EE": Modality.ROBOT_STATE,
    "control_command_success_rate": Modality.OTHER,
    "dq": Modality.ROBOT_STATE,
    "elbow": Modality.ROBOT_STATE,
    "q": Modality.ROBOT_STATE,
    "tau_J": Modality.TACTILE,
    "tau_ext_hat_filtered": Modality.TACTILE,
}


def _episode_dirs(root: Path) -> list[Path]:
    episodes = root / "episodes"
    if not episodes.is_dir():
        raise FileNotFoundError(f"not a QUT dataset-example directory (no episodes/): {root}")
    found = sorted(p for p in episodes.iterdir() if (p / "state.json").is_file())
    if not found:
        raise FileNotFoundError(f"no QUT episode state.json files under {episodes}")
    return found


def _time_ms(row: dict[str, Any]) -> float:
    raw = row.get("time")
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError(f"state row has invalid time field: {raw!r}")
    return float(raw)


def _to_builtin(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _h5_camera_names(image_h5: Path, episode_id: str) -> list[str]:
    if not image_h5.is_file():
        return []
    try:
        import h5py
    except ModuleNotFoundError:
        return []
    with h5py.File(image_h5, "r") as h5:
        if episode_id not in h5:
            return []
        return sorted(str(name) for name in h5[episode_id].keys())


def load_qut_dataset(
    path: str | Path,
    *,
    max_episodes: int | None = None,
) -> tuple[dict[str, list[Sample]], dict[str, Any]]:
    """Load the QUT Hugging Face dataset-example as a canonical run.

    State row and camera frame indices are interpreted at the dataset's native
    10 Hz rate. Episodes are composed onto a continuous run timeline.
    """

    root = Path(path)
    episode_paths = _episode_dirs(root)
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]
    if not episode_paths:
        raise ValueError(f"QUT dataset has no episodes selected: {root}")

    run: dict[str, list[Sample]] = {}
    seq: dict[str, int] = {}
    boundaries: list[dict[str, object]] = []
    run_offset_ns = 0
    boundary_gap_ns = round(1_000_000_000 / QUT_SOURCE_RATE_HZ)

    for episode_path in episode_paths:
        rows = json.loads((episode_path / "state.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{episode_path / 'state.json'} must contain a list of rows")
        if not rows:
            continue

        episode_id = episode_path.name
        times_ms = [_time_ms(row) for row in rows]
        episode_start_ms = times_ms[0]
        state_acq_ns = [
            run_offset_ns + frame_index * boundary_gap_ns
            for frame_index in range(len(rows))
        ]
        for state_name, modality in _STATE_STREAMS.items():
            samples = run.setdefault(state_name, [])
            seq.setdefault(state_name, len(samples))
            for frame_index, row in enumerate(rows):
                if state_name not in row:
                    continue
                samples.append(
                    Sample(
                        stream_name=state_name,
                        modality=modality,
                        sequence_id=seq[state_name],
                        acquisition_time_ns=state_acq_ns[frame_index],
                        receive_time_ns=state_acq_ns[frame_index],
                        source_clock_domain=QUT_CLOCK_DOMAIN,
                        payload=_to_builtin(row[state_name]),
                    )
                )
                seq[state_name] += 1

        image_h5 = root / "images.h5"
        camera_names = _h5_camera_names(image_h5, episode_id)
        if camera_names:
            camera_sources = [
                (camera_name, f"images.h5:/{episode_id}/{camera_name}[{{frame_index}}]")
                for camera_name in camera_names
            ]
        else:
            video_files = sorted((episode_path / "video").glob("*.mp4"))
            camera_sources = [
                (video_path.stem, f"{video_path.relative_to(root)}#frame={{frame_index}}")
                for video_path in video_files
            ]
        for camera_name, ref_template in camera_sources:
            stream_name = f"camera.{camera_name}"
            samples = run.setdefault(stream_name, [])
            seq.setdefault(stream_name, len(samples))
            for frame_index, time_ns in enumerate(state_acq_ns):
                samples.append(
                    Sample(
                        stream_name=stream_name,
                        modality=Modality.CAMERA,
                        sequence_id=seq[stream_name],
                        acquisition_time_ns=time_ns,
                        receive_time_ns=time_ns,
                        source_clock_domain=QUT_CLOCK_DOMAIN,
                        payload={
                            "episode_id": episode_id,
                            "frame_index": frame_index,
                            "source_time_ms": times_ms[frame_index],
                        },
                        payload_ref=ref_template.format(frame_index=frame_index),
                    )
                )
                seq[stream_name] += 1

        boundaries.append(
            {
                "episode_id": episode_id,
                "start_time_ns": run_offset_ns,
                "length": len(rows),
                "source_start_time_ms": episode_start_ms,
                "source_end_time_ms": times_ms[-1],
                "image_h5_present": bool(camera_names),
                "camera_count": len(camera_sources),
            }
        )
        run_offset_ns = state_acq_ns[-1] + boundary_gap_ns

    info: dict[str, Any] = {
        "source_path": str(root),
        "format": "qut.dataset_example.v0",
        "source_rate_hz": QUT_SOURCE_RATE_HZ,
        "imported_episodes": len(boundaries),
        "episodes": boundaries,
        "timestamp_mapping": {
            "source": "state.json row index and matching camera frame index at 10 Hz",
            "acquisition_time_ns": "episode start + frame_index * 100 ms",
            "receive_time_ns": "acquisition_time_ns",
            "camera_acquisition_time_ns": "same timestamp as state.json row with matching index",
            "state_time_ms": "retained as source metadata; not used as the media clock",
        },
        "camera_streams": sorted(name for name in run if name.startswith("camera.")),
        "state_streams": sorted(_STATE_STREAMS),
    }
    return run, info
