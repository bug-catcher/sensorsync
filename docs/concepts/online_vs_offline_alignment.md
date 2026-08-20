# Online vs offline alignment

There are two problems a robot-learning stack solves with the same word
"alignment", and they have almost nothing in common. Offline alignment
is a *dataset* problem: given a directory of recorded streams, produce
policy-ready frames. Online alignment is a *control* problem: given a
live stream of samples arriving now, hand the policy a "current
observation" that respects a deadline. Confuse them and you build a
recorder that hides your latency, or a policy loop that stalls waiting
for a frame that will never arrive.

## Definitions

**Offline alignment** (implemented today, D-0020 / D-0022 / D-0025):
inputs are complete, in memory or on disk, and known in advance. The
engine sees every sample from every stream before picking the frame
grid. The `align_run` API in `embodied_sync/align/engine.py` is
offline; it needs the full run, and the frame grid is clipped to the
intersection of the streams' acquisition-time windows.

**Online alignment** (implemented, D-0026 / D-0027): inputs arrive one
sample at a time, and every request has a *deadline*.
`embodied_sync/align/ring_buffer.py::StreamRingBuffer` and
`embodied_sync/align/online.py::MultiStreamAligner` maintain bounded
ring buffers per stream and answer "what is the best observation for
target time T that I can produce by time T + deadline?" without ever
seeing the future. `get_aligned_observation(target_ns, deadline_ns=0)`
is the per-stream API; `get_latest_policy_observation(now_ns)` and
`MultiStreamAligner.get_latest_policy_frame(now_ns)` are the
deadline-zero policy-tick wrappers.

The distinction is not "batch vs streaming". Both operate on samples
with timestamps; both use the same tolerance rules; both can pick
nearest-neighbor, ZoH, or interpolation. The distinction is
**access to future samples**.

## Correctness constraints

Offline alignment is a pure function. Given a run, `align_run`
produces the same aligned episode every time — the only knobs are
`target_rate_hz`, `method`, and `ground_truth`. It is free to look
ahead: for target time T it can pick the sample just after T if that
sample lies closer than any earlier one. That is what nearest-neighbor
does. Linear interpolation actively requires it: the bracketing pair
straddles T.

Online alignment cannot look ahead, ever. At time T you have exactly
what has arrived by time T + deadline. The tightest deadline is zero
(no waiting): only samples with `receive_time_ns <= T` are eligible.
Any policy that wants to use a sample with `acquisition_time_ns > T`
is asking the future to have already happened; the engine must refuse.

Two invariants follow.

1. **Causality**: an online pick with `deadline_ns == 0` must satisfy
   `receive_time_ns <= T`. Nearest-neighbor on `acquisition_time_ns`
   is *not* causal — it can pick a sample whose acquisition is later
   than T (arbitrarily close in time; still in the future from the
   engine's perspective). Zero-order hold is causal by construction:
   `acquisition_time_ns <= T` and `receive_time_ns <= T` are the same
   inequality for a clean stream.

2. **Bounded memory**: an online engine cannot grow unboundedly with
   the length of the recording. It carries a per-stream ring buffer
   sized by the tolerance window (plus safety margin for jitter). The
   offline `align_run` has no such bound — it holds every sample of
   every stream during the pass. Porting an offline algorithm to
   online means both dropping the look-ahead *and* fitting the state
   into a bounded buffer.

Two things that are the same across both.

- **Tolerance rule** — half the median inter-sample acquisition
  interval — is the same in both settings. A stream's own regularity
  bounds how far a plausible pick can be from the target, regardless
  of whether the picker knows the future.
- **Skew convention** — `skew_ns = source - target` — is the same. In
  online with deadline 0, ZoH produces `skew_ns <= 0` (source must
  precede target); with a deadline, `skew_ns` can become positive if
  the engine chooses to wait for a nearer future sample.

## A worked failure case

A demo team runs a bimanual grasping policy at 10 Hz. Training data
was aligned offline with nearest-neighbor, `target_rate_hz=10.0`.
Every training frame has a robot-state sample within 2 ms of its
target time — some just before, some just after — because
nearest-neighbor picks the closest by absolute skew. Median absolute
skew per frame: 1.9 ms. The learned policy generalises fine on the
dataset.

Deployed online, the same rig runs at 10 Hz. The team writes:

```python
# WRONG: reuses the offline nearest-neighbor picker at the policy tick.
target_ns = now_ns()
sample = nearest_neighbor(robot_state_buffer, target_ns)
action = policy(sample.payload)
send(action)
```

At the moment `target_ns = now_ns()` fires, no robot-state sample with
`acquisition_time_ns > target_ns` exists yet — the future hasn't
arrived. So `nearest_neighbor` collapses to the last received sample,
which is up to 4 ms stale (one inter-sample interval at 250 Hz).
Meanwhile *training* frames had, on average, samples with
`acquisition_time_ns = target_ns + 2 ms`. The policy has learned to
condition on state that is *just after* the target, and at inference
it gets state that is *just before*. The distribution shift is small
(about 4 ms per feature) but nonzero, and it points in a consistent
direction — the policy responds too early, actions bias into
overshoot, grip force is misjudged. Nothing raises a fault; the
policy just underperforms.

Two possible fixes, both online-correct.

**Deadline-zero ZoH.** Replace the offline nearest-neighbor picker
with ZoH at deadline 0. Every frame carries a strictly non-positive
skew, matching what a causal policy actually sees. Re-align the
*training* data with ZoH too — otherwise you have the same
distribution shift in reverse. This is the safe default.

**Deadline-shifted target.** Fire the tick at
`target_ns = now_ns() - offline_median_skew`; then use ZoH. This
matches the offline picker's expected skew shape without violating
causality. Requires that the offline pipeline record the per-stream
median skew alongside the episode manifest — currently the sync-
quality report exposes it, so this fix is available today; the
online engine just needs to know it.

The failure is diagnosable in the sync-quality report: median skew of
the training set is +2 ms; median skew of the deployment stream (if
you record and align it) is around -2 ms. The 4 ms gap is the bug's
fingerprint. Because both numbers are small, it is easy to overlook —
until you notice they consistently sit on *opposite* sides of zero.

## Where `SyncSession` fits

`StreamRingBuffer` and `MultiStreamAligner` above are the engine; most
code should not call them directly. `embodied_sync.session.SyncSession`
(D-0037) wraps them into the API a live consumer actually wants:
`push`/`attach` per stream, `get()` a `SyncBundle` back. `bundle.ok`
*is* "every configured stream present and within tolerance" — the
causality and bounded-memory invariants above still hold underneath,
but the caller no longer has to reconstruct "was this pick good enough"
from a raw skew number on every tick.

Two bundling modes cover the two shapes online alignment comes in:

- **`policy="latest"` (default)** is deadline-zero ZoH per stream —
  exactly `get_latest_policy_frame(now_ns)` above, one call per
  configured target time. Use this for a policy tick: "give me the
  best available-now observation."
- **`policy="approximate"` (D-0040)** is a different shape entirely: a
  true pivot-and-span-minimizing `ApproximateTime` bundler (ROS's
  `ApproximateTimeSynchronizer` contract — each message used at most
  once, sets published in order, span minimized), surfaced through
  `poll_bundles()` when the data allows a bundle to close, not on a
  fixed tick. Use this when "these N messages arrived close together"
  matters more than "what do we know right now" — multi-camera capture
  is the canonical case.

`SyncSession` also owns what this document assumes is available but
doesn't itself provide: the `SyncViolation` stream for every degraded
pick (a stale hold, an unmapped clock domain, a stream matching
nothing), live `quality()` windows, and clock-domain corrections via
`embodied_sync.calibrate` (see
[`timestamps_clock_domains.md`](timestamps_clock_domains.md)). The
worked failure case above — training and deployment silently
disagreeing on picker semantics — is exactly what `SyncViolation` and
the recorded `median_skew_ns` are for catching before it reaches
production.

## Rules of thumb

- Offline picks the "best" sample for a target time; online picks the
  "best available *now*" sample for a target time. Anything that can
  reach for a sample not yet received is offline.
- If you plan to deploy the same alignment policy online, choose it
  offline too — ZoH for policy observations, nearest-neighbor only
  for post-hoc analysis where causality is not required.
- Nearest-neighbor on `acquisition_time_ns` is offline-only. There is
  no correct online implementation without a deadline > 0, and even
  then you burn latency budget waiting for a future sample.
- Linear interpolation is offline-only. Any interpolation between a
  past and a future sample needs the future.
- Reports produced from offline episodes should carry the picker's
  method in metadata (they do, per D-0020 / D-0022 / D-0025) so
  online consumers can compare and detect the "training used
  nearest, deployment used ZoH" trap.
