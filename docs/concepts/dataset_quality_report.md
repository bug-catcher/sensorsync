# Dataset-quality reports: the sync slice

A dataset can be broken in many ways. Labels can be wrong; the coverage
can miss whole regimes of the task; the action space can be recorded at
the wrong resolution; and the *timing* can be wrong in the several ways
[`latency_vs_jitter_vs_drift.md`](./latency_vs_jitter_vs_drift.md)
enumerates. Any serious dataset-quality tool eventually grows a
diagnostic for each of these failure modes. `embodied_sync` v0 ships
exactly one of them — the sync-quality report — because timing is the
failure mode the alignment pipeline is *directly* responsible for; the
others rest on assumptions the alignment pipeline cannot make on the
user's behalf.

This doc names the shape of that one report, contrasts the three axes
it actually surfaces, and points at
[`../user/interpreting_sync_reports.md`](../user/interpreting_sync_reports.md)
for the "how do I read the numbers" walk-through. It is deliberately
not a field reference.

This is the *offline* report, computed once over a finished episode. A
live session has a running equivalent: `SyncSession.quality()` returns
the same per-stream rate/skew/confidence shape over a bounded recent
window, so a control loop can watch for the report's failure signatures
(stale holds, out-of-tolerance skew) while still recording, instead of
finding out only after the fact from the HTML page.

## The family and where sync sits in it

The dataset-quality diagnostics a robot-learning stack eventually needs:

- **Timing quality** — did the aligned frames actually pair samples
  from the same instant? This is what the sync-quality report answers.
- **Label quality** — are ground-truth labels correct, self-consistent,
  and in sync with the frame they annotate? Out of scope for v0; the
  alignment engine has no notion of a label.
- **Coverage and diversity** — do the recordings span the task's
  operating envelope, or are they clustered in an easy corner? Out of
  scope for v0; you cannot infer this from timestamps alone.
- **Action-space quality** — are recorded actions expressive enough
  and free of quantisation or clipping artefacts? Out of scope for v0.

The sync-quality report is the *timing-quality slice* of a broader
diagnostic family. Framing it that way matters for two reasons. First,
you should not read a clean sync report as a clean dataset — it only
rules out one class of failure. Second, when the other slices arrive,
they will follow the same pattern: a small structured summary with
version-stamped fields (see D-0024) plus a self-contained HTML
rendering, so any one of them can be dropped into an email or a repo
diff (D-0023).

## The three axes the sync report surfaces

`embodied_sync.reports.sync_quality.build_report` returns a
`SyncQualityReport` with `frame_count` plus per-stream `StreamStats`.
Each `StreamStats` carries:

```text
frame_count, missing_count, missing_rate,
median_skew_ns, median_abs_skew_ns,
median_confidence, method,
ground_truth_missing_count
```

Three of those fields drive three distinct diagnostic axes. They ask
different questions and fail in different shapes; a report that looks
fine on one axis can hide a bug that shows up loudly on another.

### Axis 1 — `missing_rate`: is the stream present at all?

The coarsest signal. For each stream, what fraction of aligned frames
have no sample within `tolerance_ns` of the target time?

```text
missing_rate = missing_count / frame_count
```

A non-zero `missing_rate` says: at those frame times, the stream had
nothing acceptable to pair. It does not say *why*. Three common causes:

- Hardware drops — samples never reached the recorder. The corruption
  side rails call this `random_drop` (D-0007). The
  `ground_truth_missing_count` column tells you how many drops the
  corruption sidecar knows about, so you can subtract known-lost
  frames from the total and see whether the remaining gap is real.
- Buffer overflow during the run — the recorder ran hot and evicted
  samples before writing them. Same downstream signature as hardware
  drops.
- Target-rate mismatch — the alignment `target_rate_hz` is set higher
  than the stream's real sample rate can support at the current
  tolerance. The stream is fine; you asked for frames it cannot
  produce.

The three collapse into the same number. To tell them apart you need
either a corruption sidecar (case one) or knowledge of the target
rate (case three). What the report guarantees is that a stream with
`missing_rate` well above zero is *not usable* for that many frames,
regardless of which cause you eventually attribute it to.

### Axis 2 — `median_skew_ns`: which side of the target are we picking?

A signed nanosecond count: the median of
`sample.acquisition_time_ns - target_time_ns` across non-missing
frames of that stream.

- **Positive** — the picker preferred samples in the *future* of the
  target. Only nearest-neighbor and linear interpolation can do this;
  ZoH cannot (it is causal by construction). A `+2 ms` median skew is
  a nearest-neighbor picker consistently reaching forward.
- **Negative** — the picker preferred samples in the *past* of the
  target. ZoH will always produce this; nearest-neighbor may too when
  a past sample happens to be closer.
- **Near zero** — the picker split cleanly across the target. The
  usual signature of nearest-neighbor on a stream whose rate divides
  the target rate.

`median_abs_skew_ns` is the same statistic without the sign; it tells
you how *far* the picker had to reach without saying which direction.
For a report that mixes streams aligned by different pickers, keep
both.

The sign of `median_skew_ns` is a distinct diagnostic from its
magnitude, and this is where the "4 ms distribution shift" failure
case walked through in
[`online_vs_offline_alignment.md`](./online_vs_offline_alignment.md)
lives. A training set aligned nearest-neighbor has near-zero
`median_skew_ns`; the same policy deployed online with ZoH has
`median_skew_ns` around `-4 ms` on a 250 Hz stream. Both look small
in isolation. The *gap* between them — the sign flip — is the bug's
signature. The report field to compare across the two aligned
episodes is exactly this one.

### Axis 3 — `median_confidence`: how stale is the median pick?

Confidence is `1 - abs(skew_ns) / tolerance_ns` clamped into `[0, 1]`:
a picker that lands on the target has confidence 1.0; one that lands
right at the tolerance edge has confidence 0.0. `median_confidence`
is that median across non-missing frames of the stream.

The tempting reading is "median confidence near 1.0 means low skew".
It does not. It means low skew *relative to `tolerance_ns`*. If the
tolerance is 16 ms (half a 30 fps interval) and the median pick is
1 ms off, confidence is `~0.94` — comfortably near 1.0. If the
tolerance is 2 ms (half a 250 Hz interval) and the median pick is
1 ms off, confidence is `~0.5`. Same skew, wildly different
confidence, because "how stale is too stale" is a per-stream choice.

Two shapes to watch for:

- **High median confidence with `median_skew_ns` clamped near a
  tolerance edge.** The picker is systematically reaching to one
  side of the target and just barely staying in tolerance. This is a
  *systematic offset*, not random noise — usually a hidden
  `fixed_latency` (see D-0009) or an early-stage `clock_drift` (see
  D-0014) that has not yet grown large enough to blow the tolerance.
  Widening the tolerance hides it further; the fix is to identify
  the offset upstream.
- **Median confidence sliding towards 0.5 with `missing_rate` still
  zero.** The picker is finding a sample every time, but the samples
  are far from the target. Usually a target-rate/stream-rate
  mismatch, or drift that has grown enough to eat the tolerance
  budget but not enough to overflow it yet. A leading indicator that
  `missing_rate` is about to rise if the recording continues.

Confidence quantiles are a *scale-free* view of skew: they answer
"how much of the tolerance budget did we typically spend?" and
compose across streams whose absolute skews are not comparable.

## A worked case: three axes, one bug

A rig records robot state at 250 Hz and a front camera at 30 Hz. The
camera has 12 ms of unmodelled pipeline latency (an unfixed constant
offset — see the failure case in
[`latency_vs_jitter_vs_drift.md`](./latency_vs_jitter_vs_drift.md)).
The dataset is aligned with ZoH at `target_rate_hz=30.0`,
tolerance 16 ms per stream. The sync-quality report looks like this:

```text
Stream        Frames  Missing        Median skew   Median |skew|  Median conf.
robot_state    18000  0 (0.0%)       -1.994 ms     1.994 ms       0.876
cam_front      18000  0 (0.0%)      -13.001 ms    13.001 ms       0.187
```

Read one axis at a time:

- `missing_rate` on both streams is zero. Axis 1 says nothing is
  wrong.
- `median_skew_ns` on `cam_front` is `-13 ms`. That is inside the
  16 ms tolerance, but it is *pinned near the edge*, and it is on the
  causal side (ZoH always is). Axis 2 says the picker is reaching
  well into the past on every frame.
- `median_confidence` on `cam_front` is 0.187. Axis 3 says the picker
  spent 81 percent of its tolerance budget on the median frame.

Any single axis in isolation looks survivable. The three together
say: this stream has a systematic offset that ZoH is silently
absorbing into skew, and if a real drift or jitter component grows on
top of it, `missing_rate` will start climbing without warning. The
same rig aligned nearest-neighbor would show `median_skew_ns` around
`+3 ms` (nearest-neighbor reaches for the closest sample regardless
of direction), and the picker-vs-picker comparison would flag the
constant offset the same way the online-vs-offline walk-through does.

None of this requires reading the frames themselves. Three numbers
per stream carry the diagnostic.

## Rules of thumb

- Treat the sync-quality report as necessary, not sufficient. A clean
  report rules out timing failures. Label, coverage, and action-space
  failures are not visible in it.
- Read the three axes in order — `missing_rate`, then
  `median_skew_ns`, then `median_confidence`. If `missing_rate` is
  above your tolerance, the other two are conditioned on a biased
  subset of frames and should be interpreted cautiously.
- The sign of `median_skew_ns` is a distinct fact from its magnitude.
  Compare signs across streams and across training-vs-deployment
  episodes before comparing magnitudes.
- A `median_confidence` near 1.0 does not mean low skew — it means
  low skew *relative to `tolerance_ns`*. Two runs with different
  tolerances are not directly comparable on this axis without
  renormalising.
- The report and the episode `manifest.json` both carry
  `median_skew_ns`. For downstream diagnostics that want to compare
  training and deployment episodes, prefer the manifest — see
  [`robot_policy_observations.md`](./robot_policy_observations.md)'s
  "Reading `median_skew_ns` in the manifest" section for the exact
  shape of the check.

## Related material

- Concept: [`latency_vs_jitter_vs_drift.md`](./latency_vs_jitter_vs_drift.md)
  — the three timing-error shapes that produce the skew and missing-rate
  signatures the report shows.
- Concept: [`online_vs_offline_alignment.md`](./online_vs_offline_alignment.md)
  — the "4 ms distribution shift" failure case the report's
  `median_skew_ns` axis is designed to catch.
- Concept: [`robot_policy_observations.md`](./robot_policy_observations.md)
  — the "Reading `median_skew_ns` in the manifest" section, which is
  the concrete downstream use of the same statistic.
- User: [`../user/interpreting_sync_reports.md`](../user/interpreting_sync_reports.md)
  — the "how do I read this specific table" companion. Column-by-column;
  this doc is the "why the columns exist" companion.
- Decisions: D-0023 (sync-quality report shape and HTML target) and
  D-0024 (canonical types split for downstream consumption without the
  reports subpackage).
