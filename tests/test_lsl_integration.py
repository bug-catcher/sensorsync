from __future__ import annotations

import json

from embodied_sync.adapters.lsl import load_lsl_replay


def test_lsl_replay_preserves_clock_offset_metadata(tmp_path) -> None:
    path = tmp_path / "lsl_replay.json"
    path.write_text(
        json.dumps(
            {
                "format": "embodied_sync.lsl.replay.v0",
                "streams": {
                    "imu": {
                        "modality": "tactile",
                        "source_clock_domain": "lsl",
                        "xdf_clock_offset_ns": -3,
                        "samples": [
                            {
                                "sequence_id": 0,
                                "acquisition_time_ns": 10,
                                "receive_time_ns": 14,
                                "payload": [1.0, 2.0, 3.0],
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    run = load_lsl_replay(path)

    sample = run["imu"][0]
    assert sample.acquisition_time_ns == 10
    assert sample.receive_time_ns == 14
    assert sample.payload == {"value": [1.0, 2.0, 3.0], "xdf_clock_offset_ns": -3}
