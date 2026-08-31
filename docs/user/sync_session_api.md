# Message-filters-style synchronization without ROS

`SyncSession` offers a small callback-and-polling boundary for live streams.
It is useful when an SDK already delivers callbacks but ROS is not part of the
application. It is not a ROS `message_filters` replacement and does not make
hardware clocks agree.

```python
import embodied_sync as embsync

with embsync.init(
    run_dir="runs/demo",
    streams={"camera": embsync.StreamConfig(rate_hz=30, tolerance_ms=20)},
    primary="camera",
) as sync:
    camera_sdk.on_frame(sync.attach("camera", timestamp=lambda f: f.device_ts_ns))
    while running:
        sync.push("robot", robot_sdk.read_state())
        bundle = sync.get()
        if bundle.ok:
            policy(bundle["camera"], bundle["robot"])
```

Bundles record per-stream skew and quality metadata. Missing streams, stale
holds, clock resets, and unmapped clock domains are explicit outcomes.
