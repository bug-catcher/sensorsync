# Latency, jitter, and drift

These three words get used interchangeably in robotics and streaming media
prose, and it is expensive when they get used interchangeably in code. Each
describes a different way a sample's timestamps can be wrong, each shows up
as a different signature in `receive_time_ns - acquisition_time_ns`, and
each is modelled by a distinct corruption in `embodied_sync/corrupt/`. Mix
two of them up and the alignment engine's tolerance windows sit in the
wrong place; the "fix" hides one symptom by exchanging it for a subtler
one.

## The three signatures

Recall the sample model from
[`timestamps_clock_domains.md`](./timestamps_clock_domains.md): every
sample carries `acquisition_time_ns` (when the sensor observed the world)
and `receive_time_ns` (when the host received it). Call the difference
`L_i = receive_time_ns[i] - acquisition_time_ns[i]` — the per-sample
transport latency. Latency, jitter, and drift are three distinct shapes of
`L_i` over `i`.

### Latency — a constant offset

Every sample takes the same time to reach the host:

```text
L_i = L_0             (constant across i)
```

```text
acquisition:   0    33    66   100   133   166   ...
receive:      12    45    78   112   145   178   ...
L_i:          12    12    12    12    12    12   ...
```

There is no per-sample uncertainty; every arrival is late by exactly the
same amount. Latency is *predictable*: if you know `L`, you know when the
observation actually happened.

Corruption kind: `fixed_latency` (D-0009). It adds a constant to
`receive_time_ns`; acquisition is untouched, no quality flag is set — a
recorder cannot detect a hidden constant delay from one sample alone.

### Jitter — random per-sample variation

Every sample's latency is a random draw around some mean:

```text
L_i ≈ L_0 + N(0, σ²)   (independent per i)
```

```text
acquisition:   0    33    66   100   133   166   ...
receive:      14    39    78   106   150   170   ...
L_i:          14     6    12     6    17     4   ...
```

Samples can even arrive *out of order* (the sixth one has a smaller `L`
than the fifth, so its receive time can slip below its predecessor's).
Jitter is *observable in aggregate* — the variance of `L_i` gives σ —
but not per-sample.

Corruption kind: `jitter` (D-0010). It adds Gaussian noise to
`receive_time_ns` per sample, with optional clip. Sample order is
preserved in the list, but observed receive times can go non-monotonic;
downstream must treat that as a timing anomaly, not sort it away.

### Drift — a slope in L over time

The sensor and host clocks tick at slightly different rates, so the gap
between them grows (or shrinks) linearly:

```text
L_i = L_0 + drift_ppb * (t_i - t_0) / 1e9
```

```text
acquisition:      0     100M     200M     300M    400M    ...
L_i (ppb=1e5):    0       10       20       30      40    ...
receive:          0    100M+10  200M+20  300M+30 400M+40  ...
```

For a clock 100 ppm fast, that is 6 ms of extra apparent latency per
minute. Drift is *invisible over short windows* — locally it looks like
constant latency — but ruins long alignments: after 10 minutes at 100 ppm
you are 60 ms off, which at 30 fps is nearly two frames.

Corruption kind: `clock_drift` (D-0014). It adds a linear offset to
`receive_time_ns` anchored at the stream's first acquisition. No quality
flag: smooth drift is not per-sample observable without downstream clock
modelling.

## Why the distinction matters

They *look* alike at any single sample: `L_i` is a positive number in all
three cases. The differences only show up in the *shape* of `L` over
time, and each shape breaks alignment differently.

| symptom | latency | jitter | drift |
| --- | --- | --- | --- |
| mean(`L`) | shifted | unchanged | slowly changing |
| var(`L`) | 0 | > 0, constant | growing (or constant if fit as line) |
| non-monotonic receive times | never | sometimes | never |
| effect over 10 minutes | none extra | none extra | growing skew |
| observable per-sample by an honest recorder | no | no | no |

The reason `embodied_sync` splits them into separate corruption kinds is
that each requires a different countermeasure:

- **Latency** is fixed by subtracting the known constant (or estimating
  it once). Nearest-neighbor alignment on `acquisition_time_ns` is
  immune to a constant per-stream `L`, because the constant cancels
  when you compare across streams that share the same shift.
- **Jitter** is fixed by widening the alignment tolerance window: if you
  know σ, you know how much slop to allow around each target time. Fit
  it too tight and you drop matches; fit it too loose and you mis-pair.
- **Drift** is fixed by *estimating a slope* between the two clocks and
  applying an ongoing correction — a fundamentally different repair
  from either of the other two.

## The failure case: confusing drift with latency

The bug goes like this. A rig ships with a hardware camera that has 12 ms
of pipeline delay, and the integrator "calibrates" it by subtracting a
fixed 12 ms from every camera `acquisition_time_ns`. Alignment against
robot state looks perfect for the first minute. Over the course of a
30-minute demo the camera clock, running 80 ppm fast, drifts 144 ms
ahead. The correction — a constant — cannot track a slope, so the paired
frames are gradually mis-selected. Because there is no anomaly at any
single sample, and every step's math checks out, the mis-pairing is
silent. The label numbers on the resulting dataset look plausible; the
learned policy underperforms on the real robot, and the failure diagnosis
takes weeks.

Concretely, in the synthetic rig:

```python
from embodied_sync.corrupt import (
    ClockDriftCorruption,
    CorruptionProfile,
    FixedLatencyCorruption,
    apply_profile,
)
from embodied_sync.streams.synthetic import generate_synthetic_run

run = generate_synthetic_run(duration_s=60.0, seed=0)

# What the integrator thinks they applied: a constant latency correction.
fixed = apply_profile(
    run,
    CorruptionProfile(
        seed=0,
        corruptions=(FixedLatencyCorruption(stream="cam_front", offset_ns=12_000_000),),
    ),
).run["cam_front"]

# What is actually happening in the rig: linear drift.
drifted = apply_profile(
    run,
    CorruptionProfile(
        seed=0,
        corruptions=(ClockDriftCorruption(stream="cam_front", drift_ppb=80_000),),
    ),
).run["cam_front"]

# Latency shift is constant across the run …
print(fixed[-1].receive_time_ns - fixed[0].receive_time_ns)
# … drift accumulates over the same span.
print(drifted[-1].receive_time_ns - drifted[0].receive_time_ns)
```

Fitting a single constant `L` to the drifted stream gets the *mean*
right and the *tails* wrong; alignment errors at the ends of the demo
can exceed a full frame period. The two corruptions look identical for
one sample and diverge over the run — which is exactly what makes the
mix-up so easy and so damaging.

## Rules of thumb

- Model **latency** as a per-stream constant. Estimate it once, subtract
  once, and expect it to be right for the whole recording.
- Model **jitter** as a per-stream standard deviation. Use it to size
  alignment tolerance windows and to threshold "stale observation"
  detectors. Never sort receive times to smooth it away.
- Model **drift** as a *rate*, not an offset. Estimate it against a
  reference clock periodically; if you cannot estimate it, name the
  clock domains separately so cross-domain math is legal only through
  an explicit mapping.
- If a symptom you diagnosed as one of the three does not repeat on a
  new recording, you probably named the wrong one. Rerun the synthetic
  harness with the specific corruption kind you suspect and check the
  signature of `L_i` — that is what the harness exists for.
