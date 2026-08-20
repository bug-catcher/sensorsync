#!/usr/bin/env python3
import sys
import pyxdf

path = sys.argv[1]
streams, header = pyxdf.load_xdf(path)

print(f"Loaded {len(streams)} streams from {path}\n")

for stream in streams:
    info = stream["info"]
    name = info["name"][0]
    stype = info["type"][0]
    nominal_srate = info["nominal_srate"][0]
    data = stream["time_series"]
    stamps = stream["time_stamps"]

    print(f"Stream: {name}")
    print(f"  type: {stype}")
    print(f"  nominal_srate: {nominal_srate}")
    print(f"  samples: {len(stamps)}")
    if len(stamps):
        print(f"  first timestamp: {stamps[0]:.6f}")
        print(f"  last timestamp:  {stamps[-1]:.6f}")
        print(f"  duration:        {stamps[-1] - stamps[0]:.3f}s")
    print()
