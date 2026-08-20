# Optional alignment verification

`embodied-sync` can ask a separately deployed verifier for a coarse second
opinion without importing a model package or uploading media itself. The
classical clock mapping remains authoritative; disagreement is added to the
human inspector as a warning.

## CLI

Run a compatible service, then configure its URL and optional bearer token:

```bash
export EMBODIED_SYNC_VERIFY_URL=http://127.0.0.1:8765
export EMBODIED_SYNC_VERIFY_TOKEN='token-from-the-service-operator'

embsync verify file:///data/video.mp4 file:///data/audio.wav \
  --offset-ms 20 \
  --search-radius-ms 400 \
  --tolerance-ms 200 \
  --metadata scene=pick_001 \
  --out verification.json
```

The public client sends the two URI strings, the classical proposal, search
gate, and string metadata as JSON. It does not read or transmit either media
file. A local service therefore needs access to the same paths; a remote
deployment can define its own content-ID or signed-URL resolver.

The command succeeds when the service produced a valid review, even when
`needs_inspection` is true. That field is a routing decision, not a transport
failure.

## Library and inspector

```python
from embodied_sync.inspect import (
    HTTPAlignmentVerifier,
    VerificationRequest,
    apply_verification_review,
    verify_alignment,
)

request = VerificationRequest(
    reference_uri="file:///data/video.mp4",
    candidate_uri="file:///data/audio.wav",
    proposed_offset_ns=mapping.offset_ns,
    search_radius_ns=400_000_000,
    metadata=(("scene", "pick_001"),),
)
review = verify_alignment(
    mapping,
    request,
    HTTPAlignmentVerifier("http://127.0.0.1:8765", token=token),
    tolerance_ns=200_000_000,
)
context = apply_verification_review(context, review)
page = build_page(inspection, evidence, context)
```

`apply_verification_review` adds the verifier identity, offset, confidence,
and disagreement to the fitted-mapping panel. When disagreement exceeds the
tolerance, it also adds an inspector banner that explicitly says the
classical fit was not replaced.

## Wire contract

The client calls `POST /v1/verify` with schema version 1. Responses carry a
verifier ID, integer-nanosecond offset, optional confidence in `[0, 1]`, and
string details. The client rejects malformed responses and offsets outside
the requested search radius.

Confidence is verifier-specific. It must not be interpreted as calibrated
probability unless that verifier's own documentation establishes calibration.
