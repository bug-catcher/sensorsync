#!/usr/bin/env python3
import math
import random
import signal
import time
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock


running = True


def stop_handler(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, stop_handler)
signal.signal(signal.SIGTERM, stop_handler)


@dataclass
class NumericStream:
    name: str
    stream_type: str
    channel_count: int
    nominal_rate_hz: float
    unit: str
    generator: Callable[[float], List[float]]

    def make_outlet(self) -> StreamOutlet:
        info = StreamInfo(
            name=self.name,
            type=self.stream_type,
            channel_count=self.channel_count,
            nominal_srate=self.nominal_rate_hz,
            channel_format="float32",
            source_id=f"embsync_{self.name}",
        )

        channels = info.desc().append_child("channels")
        for i in range(self.channel_count):
            channels.append_child("channel") \
                .append_child_value("label", f"ch{i}") \
                .append_child_value("unit", self.unit) \
                .append_child_value("type", self.stream_type)

        return StreamOutlet(info)


def robot_state(t: float) -> List[float]:
    # 7-DoF-ish joint vector
    return [math.sin(t + i * 0.2) for i in range(7)]


def tactile_array(t: float) -> List[float]:
    # 4x4 synthetic pressure/taxel grid flattened
    base = np.zeros((4, 4), dtype=np.float32)
    cx = 1.5 + math.sin(t) * 0.8
    cy = 1.5 + math.cos(t * 0.7) * 0.8
    for y in range(4):
        for x in range(4):
            base[y, x] = math.exp(-((x - cx) ** 2 + (y - cy) ** 2))
    return base.flatten().tolist()


def audio_energy(t: float) -> List[float]:
    # Pretend this is a 20 ms audio energy / spectral feature window
    return [
        0.5 + 0.5 * math.sin(2 * math.pi * 1.0 * t),
        0.5 + 0.5 * math.sin(2 * math.pi * 3.0 * t),
        random.random() * 0.05,
    ]


def camera_features_front(t: float) -> List[float]:
    # Stand-in for camera frame metadata/features, not raw images
    return [math.sin(t), math.cos(t), t % 1.0, random.random() * 0.01]


def camera_features_wrist(t: float) -> List[float]:
    return [math.sin(t + 0.1), math.cos(t + 0.1), t % 1.0, random.random() * 0.01]


def publish_numeric_stream(stream: NumericStream):
    outlet = stream.make_outlet()
    period = 1.0 / stream.nominal_rate_hz
    next_push = time.perf_counter()

    print(f"Publishing {stream.name} at {stream.nominal_rate_hz} Hz")

    while running:
        now_perf = time.perf_counter()
        if now_perf >= next_push:
            ts = local_clock()
            sample = stream.generator(ts)
            outlet.push_sample(sample, timestamp=ts)
            next_push += period
        else:
            time.sleep(min(0.001, next_push - now_perf))


def main():
    import threading

    streams = [
        NumericStream("camera_front_features", "camera_features", 4, 30.0, "a.u.", camera_features_front),
        NumericStream("camera_wrist_features", "camera_features", 4, 30.0, "a.u.", camera_features_wrist),
        NumericStream("robot_state", "joint_state", 7, 250.0, "rad", robot_state),
        NumericStream("tactile_array", "tactile", 16, 60.0, "pressure", tactile_array),
        NumericStream("audio_energy_window", "audio", 3, 50.0, "energy", audio_energy),
    ]

    threads = []
    for s in streams:
        th = threading.Thread(target=publish_numeric_stream, args=(s,), daemon=True)
        th.start()
        threads.append(th)

    # Marker stream: irregular contact / event labels
    marker_info = StreamInfo(
        name="event_markers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id="embsync_event_markers",
    )
    marker_outlet = StreamOutlet(marker_info)

    print("Publishing event_markers irregularly")
    print("Press Ctrl+C to stop.")

    event_id = 0
    while running:
        time.sleep(random.uniform(0.7, 2.0))
        ts = local_clock()
        label = f"contact_event_{event_id}"
        marker_outlet.push_sample([label], timestamp=ts)
        print(f"{ts:.3f}: {label}")
        event_id += 1

    print("Stopping publishers...")


if __name__ == "__main__":
    main()
