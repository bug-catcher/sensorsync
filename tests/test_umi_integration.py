from __future__ import annotations

import json

from embodied_sync.adapters.umi import load_umi_replay_buffer


def test_umi_replay_buffer_applies_explicit_latency_offsets(tmp_path) -> None:
    path = tmp_path / "umi_replay.json"
    path.write_text(
        json.dumps(
            {
                "format": "embodied_sync.umi.replay_buffer.v0",
                "streams": {
                    "wrist_camera": {
                        "modality": "camera",
                        "source_clock_domain": "umi_camera",
                        "latency_offset_ns": 12,
                        "samples": [
                            {
                                "sequence_id": 5,
                                "acquisition_time_ns": 100,
                                "payload_ref": "frames/000005.jpg",
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    run = load_umi_replay_buffer(path)

    sample = run["wrist_camera"][0]
    assert sample.acquisition_time_ns == 100
    assert sample.receive_time_ns == 112
    assert sample.payload_ref == "frames/000005.jpg"
