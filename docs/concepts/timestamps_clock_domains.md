# Timestamps and clock domains

Every synchronization bug in robot-learning data is, at bottom, a confusion
about one of two questions:

1. **When did this measurement happen?**
2. **According to whose clock?**

`embodied_sync` forces both questions to have explicit answers on every
sample. This doc explains the model using the actual `Sample` type
(`embodied_sync/core/sample.py`) and the synthetic rig
(`embodied_sync/streams/synthetic.py`) as concrete examples.

## Two timestamps per sample, not one

A `Sample` carries **two** timestamps:

- `acquisition_time_ns` — when the sensor observed the world.
- `receive_time_ns` — when the host received the data.

Real transport takes time, so in clean data `receive >= acquisition`. The
difference is exposed as `Sample.transport_latency_ns`:

```text
world event      sensor observes         host receives
     |                 |                       |
-----+-----------------+-----------------------+---------> time
                       ^                       ^
              acquisition_time_ns       receive_time_ns
                       |<-- transport latency -->|
```

Why keep both? Because they answer different questions:

- **Alignment** ("which camera frame goes with which joint state?") must use
  `acquisition_time_ns` — you want to pair what the sensors saw at the same
  instant, not what happened to arrive together.
- **Online policy code** ("what is the freshest observation I can act on?")
  is constrained by `receive_time_ns` — you cannot use data that has not
  arrived yet, no matter when it was acquired.

A system that stores only one timestamp has silently chosen one of these
answers and destroyed the other.

### Concrete example: the synthetic rig

The synthetic truth harness gives every stream a *fixed, documented*
transport latency (D-0006), so corruption profiles are the only source of
timing noise:

| stream | rate | transport latency |
| --- | --- | --- |
| `cam_front` | 30 Hz | 12 ms |
| `cam_wrist` | 30 Hz | 15 ms |
| `robot_state` | 250 Hz | 1 ms |
| `tactile` | 60 Hz | 2 ms |
| `audio` | 50 Hz | 20 ms |
| `actions` | 10 Hz | 0.5 ms |
| `events` | irregular | 0.1 ms |

Runnable (from a checkout):

```bash
embsync synth --out /tmp/demo_run --seed 0 --duration-s 1.0
```

```python
from embodied_sync.datasets.io import load_run

run = load_run("/tmp/demo_run")
s = run["cam_front"][3]        # frame 3: i * 1e9 / 30 Hz = 100_000_000 ns
s.acquisition_time_ns          # 100000000
s.receive_time_ns              # 112000000  (+ 12 ms transport)
s.transport_latency_ns         # 12000000
```

Note what this implies for cross-stream pairing: `cam_front` frame 3 and the
`robot_state` sample acquired at the same 100 ms instant arrive **11 ms
apart**. Pairing by arrival order would systematically match the camera with
robot state from the past. That is the classic sync bug this project exists
to detect, and it is visible only because acquisition and receive times are
kept separate.

## Integer nanoseconds, never floats

All stored timestamps are integer nanoseconds (D-0002). `Sample.__post_init__`
rejects anything else, including `bool`. Floats are allowed only in derived
statistics (rates, skew estimates).

The reason is not pedantry. Float64 has 53 bits of mantissa; nanosecond
epoch timestamps need more:

```python
>>> t = 2**53 + 1            # ~104 days in ns; any 2026 epoch-ns is far bigger
>>> float(t) == float(t - 1)
True                          # two different instants, one float
```

Unix epoch time in nanoseconds today is ~1.78e18 ≈ 2**60.6. At that
magnitude a float64 has a resolution of **256 ns** — worse than the timing
precision of every stream in the rig. Any pipeline step that round-trips
timestamps through float (JSON numbers via a float path, `numpy.float64`
arrays, seconds-as-float conversions) quantizes them, and the corruption is
silent and unrecoverable. This is why the run format (`datasets/io.py`)
serializes timestamps as JSON integer literals with no float path anywhere,
and why the round-trip test pins the exact value `2**53 + 1`.

## Clock domains: "according to whose clock?"

A timestamp is meaningless without knowing which clock produced it. Every
`Sample` names that clock in `source_clock_domain` (e.g. `"host_mono"`,
`"cam_front_hw"`, `"lsl"`).

Different clocks disagree in two ways:

- **Offset** — a camera's hardware clock might start at 0 at power-on while
  the host monotonic clock started at boot. Their readings can differ by
  hours while describing the same instant.
- **Drift** — independent oscillators tick at slightly different rates
  (tens of ppm is typical), so even a perfectly measured offset decays.
  50 ppm is 3 ms of error per minute — at 30 fps, that is a whole frame of
  misalignment in ~11 minutes.

The project rule (D-0003): **clock domains are never mixed silently.**
Comparing or subtracting timestamps from different domains requires an
explicit mapping (offset/drift model, `embodied_sync.time`, Milestone 2).
If no mapping is known, alignment still proceeds, but with lowered
confidence and an explicit warning recorded in the output — a wrong answer
with a confident face is the failure mode we refuse.

### Concrete failure case

A rig records camera frames stamped by the camera's hardware clock and robot
state stamped with host wall-clock time. Both columns are "nanoseconds", both
are monotonically increasing, and naive nearest-neighbor pairing on them
"works" — it produces pairs without any error message. Then NTP steps the
wall clock 80 ms backwards during a recording, or the camera is power-cycled
between episodes, and every downstream label is quietly wrong. With named
domains, that subtraction is illegal until someone supplies the
`cam_hw -> host_wall` mapping, and the sync report shows exactly what was
assumed.

In the synthetic rig every stream currently uses `"host_mono"` — one shared
domain, so cross-stream math is legal. `embsync corrupt`'s `clock_drift`
kind (below) simulates the effect on a synthetic run without changing its
declared domain; a *real* second clock domain — a second device with its
own oscillator — needs an actual offset/drift mapping between the two
domains, which is what `embodied_sync.calibrate` produces (see below).
Either way, the alignment engine uses the declared mapping or degrades
loudly; it never assumes the mapping is identity.

### Recovering a real cross-device mapping: `embodied_sync.calibrate`

The synthetic rig can *declare* every stream's domain because it wrote
every sample itself. A real two-device rig cannot: you have two clocks and
no ground truth linking them, only whatever impulsive event both devices
happened to observe — a clap two microphones both picked up, a QR code
with an embedded timestamp both cameras filmed, a matched pair of event
trains. `embodied_sync.calibrate.estimator.fit_clock_mapping` turns paired
event times from *either* clock into a `LatencyEstimate` (offset + drift +
confidence); `calibrate/clap.py` and `calibrate/visual_timestamp.py`
supply the paired event times for the two most common physical
calibrators, and `embsync calibrate clap` runs the audio path end to end.

That `LatencyEstimate` is exactly the "explicit mapping" this section
requires before cross-domain arithmetic is legal. `SyncSession` consumes
it directly: `register_clock_mapping(mapping)` records it,
`time_correction(stream)` returns the cached correction to add to a
stream-domain timestamp to reach the session domain, and
`mark_clock_reset(stream)` invalidates it the moment a device reconnects
— an unmapped domain does not silently become mapped-forever from one
calibration shot early in a session.

## How corruptions map onto this model

The corruption harness (`embodied_sync/corrupt/`, D-0009/D-0010) edits
exactly one thing at a time, which only makes sense in terms of the split
above:

- `fixed_latency`, `jitter`, and `clock_drift` all edit **`receive_time_ns`
  only** — delivery got slower, noisier, or linearly slower-over-time;
  the world was still observed on the clean `acquisition_time_ns` grid.
  This models transport/host-clock drift, not the sensor's own clock
  lying (a true `acquisition_time_ns`-domain drift is a separate,
  unimplemented corruption kind); recovering a *real* second acquisition
  clock domain is `embodied_sync.calibrate`'s job, above, not the
  corruption harness's.
- `dropped_frames` removes samples entirely; survivors keep their original
  `sequence_id`, because the gap is the observable symptom.

A real system can observe a sequence-id gap (so drops set the `gap_before`
quality flag) but cannot observe that a given sample was jittered (so jitter
sets no flag; the ground truth lives in the profile and the returned
corruption metadata). "What could an honest recorder have known?" is the
test for every flag.

## Rules of thumb

- Store acquisition *and* receive times; never collapse them.
- Integer nanoseconds end to end; convert at the edges, exactly once.
- Name the clock. If you cannot name it, that is itself information —
  record it as its own domain rather than assuming `host_mono`.
- Align on acquisition time; gate online decisions on receive time.
- Cross-domain arithmetic requires an explicit mapping or an explicit
  warning. Never both absent.
