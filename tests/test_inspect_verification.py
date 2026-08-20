from __future__ import annotations

import pytest

from embodied_sync.inspect import (
    AlignmentVerifier,
    Inspection,
    PageContext,
    StreamInfo,
    VerificationRequest,
    VerificationResult,
    apply_verification_review,
    build_page,
    review_verification,
    verify_alignment,
)
from embodied_sync.time.clock_domain import ClockDomain, ClockKind, LatencyEstimate


def _mapping(offset_ns: int = 20_000_000) -> LatencyEstimate:
    return LatencyEstimate(
        source=ClockDomain("reference", ClockKind.HARDWARE),
        target=ClockDomain("candidate", ClockKind.HARDWARE),
        offset_ns=offset_ns,
        variance_ns=1_000_000,
    )


def test_verifier_is_a_structural_api_boundary() -> None:
    class Client:
        def verify(self, request: VerificationRequest) -> VerificationResult:
            return VerificationResult("private-api", request.proposed_offset_ns, 0.8)

    assert isinstance(Client(), AlignmentVerifier)


def test_agreement_on_threshold_does_not_replace_mapping() -> None:
    result = VerificationResult("private-api", 25_000_000, 0.75)
    review = review_verification(_mapping(), result, tolerance_ns=5_000_000)

    assert review.classical_offset_ns == 20_000_000
    assert review.disagreement_ns == 5_000_000
    assert not review.needs_inspection
    assert review.warnings == ()
    assert ("Verifier", "private-api") in review.summary_rows


def test_disagreement_routes_to_visual_inspection() -> None:
    result = VerificationResult("private-api", -30_000_000)
    review = review_verification(_mapping(), result, tolerance_ns=10_000_000)

    assert review.needs_inspection
    assert review.disagreement_ns == 50_000_000
    assert "classical fit has not been replaced" in review.warnings[0]


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_verification_result_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        VerificationResult("private-api", 0, confidence)


def test_verification_request_rejects_negative_search_radius() -> None:
    with pytest.raises(ValueError, match="search_radius_ns"):
        VerificationRequest("ref", "candidate", 0, -1)


def test_library_orchestration_calls_verifier_without_replacing_mapping() -> None:
    class Client:
        def verify(self, request: VerificationRequest) -> VerificationResult:
            return VerificationResult("deep-api", request.proposed_offset_ns + 5_000_000, 0.7)

    request = VerificationRequest("video", "audio", 20_000_000, 10_000_000)
    review = verify_alignment(_mapping(), request, Client(), tolerance_ns=2_000_000)

    assert review.classical_offset_ns == 20_000_000
    assert review.result.proposed_offset_ns == 25_000_000
    assert review.needs_inspection


def test_verification_review_is_rendered_in_inspector() -> None:
    review = review_verification(
        _mapping(), VerificationResult("deep-api", -30_000_000, 0.61), tolerance_ns=10_000_000
    )
    context = apply_verification_review(
        PageContext(
            title="Integration",
            reference=StreamInfo("video"),
            candidate=StreamInfo("audio"),
        ),
        review,
    )
    inspection = Inspection(
        events_a_ns=(),
        events_b_ns=(),
        mapping=_mapping(),
        matched=(),
        residuals_ns=(),
    )
    page = build_page(inspection, [], context)

    assert "deep-api" in page
    assert "Verifier disagreement" in page
    assert "classical fit has not been replaced" in page
