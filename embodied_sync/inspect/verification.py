"""Model-agnostic boundary and HTTP client for an optional verifier.

The public package deliberately owns the contract, not an implementation.
A verifier may run in another process, behind an API, or in a separately
licensed package.  It returns a coarse second opinion; the classical clock
mapping remains the estimate and disagreement routes the result to the human
inspector.

The public client sends only opaque URIs and scalar metadata. It never opens or
uploads media; URI resolution remains a service-side policy decision.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from embodied_sync.time.clock_domain import LatencyEstimate

if TYPE_CHECKING:
    from embodied_sync.inspect.render import PageContext

__all__ = [
    "AlignmentVerifier",
    "HTTPAlignmentVerifier",
    "VerificationServiceError",
    "VerificationRequest",
    "VerificationResult",
    "VerificationReview",
    "apply_verification_review",
    "review_verification",
    "verification_document",
    "verify_alignment",
]

_MS_NS = 1_000_000
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    """Portable description of media to be checked by an optional service.

    ``proposed_offset_ns`` uses the same convention as
    :class:`~embodied_sync.time.clock_domain.LatencyEstimate`: candidate time
    equals reference time plus offset.  URIs are opaque to the public package;
    they can be signed URLs, content IDs, or paths understood by a local
    adapter.
    """

    reference_uri: str
    candidate_uri: str
    proposed_offset_ns: int
    search_radius_ns: int
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.reference_uri or not self.candidate_uri:
            raise ValueError("reference_uri and candidate_uri must be non-empty")
        if self.search_radius_ns < 0:
            raise ValueError("search_radius_ns must be non-negative")


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """A verifier's coarse second opinion, independent of its implementation."""

    verifier_id: str
    proposed_offset_ns: int
    confidence: float | None = None
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.verifier_id:
            raise ValueError("verifier_id must be non-empty")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and between 0 and 1")


@runtime_checkable
class AlignmentVerifier(Protocol):
    """Structural interface implemented by local packages or API clients."""

    def verify(self, request: VerificationRequest) -> VerificationResult:
        """Return a second opinion without changing the classical estimate."""
        ...


class VerificationServiceError(RuntimeError):
    """The verifier endpoint failed or violated the public wire contract."""


class HTTPAlignmentVerifier:
    """Dependency-free client for the private ``POST /v1/verify`` endpoint."""

    __slots__ = ("endpoint", "timeout_s", "token")

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 120.0,
        token: str | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must be non-empty")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.token = token

    @property
    def verify_url(self) -> str:
        return (
            self.endpoint
            if self.endpoint.endswith("/v1/verify")
            else f"{self.endpoint}/v1/verify"
        )

    def verify(self, request: VerificationRequest) -> VerificationResult:
        body = json.dumps(
            {
                "schema_version": _SCHEMA_VERSION,
                "request": verification_request_dict(request),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        http_request = Request(self.verify_url, data=body, headers=headers, method="POST")
        try:
            with urlopen(http_request, timeout=self.timeout_s) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace").strip()
            raise VerificationServiceError(
                f"verifier returned HTTP {exc.code}: {message or exc.reason}"
            ) from exc
        except (URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VerificationServiceError(f"verifier request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise VerificationServiceError("verifier response must be a JSON object")
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise VerificationServiceError("unsupported verifier response schema_version")
        raw_result = payload.get("result")
        if not isinstance(raw_result, dict):
            raise VerificationServiceError("verifier response has no result object")
        return verification_result_from_dict(raw_result)


@dataclass(frozen=True, slots=True)
class VerificationReview:
    """How a verifier result should be presented alongside an alignment."""

    result: VerificationResult
    classical_offset_ns: int
    disagreement_ns: int
    tolerance_ns: int
    needs_inspection: bool

    @property
    def summary_rows(self) -> tuple[tuple[str, str], ...]:
        """Rows ready to append to :class:`~embodied_sync.inspect.PageContext`."""
        confidence = (
            "unreported"
            if self.result.confidence is None
            else f"{self.result.confidence:.3f}"
        )
        return (
            ("Verifier", self.result.verifier_id),
            ("Verifier offset", f"{self.result.proposed_offset_ns / _MS_NS:+.1f} ms"),
            ("Verifier confidence", confidence),
            ("Verifier disagreement", f"{self.disagreement_ns / _MS_NS:.1f} ms"),
        )

    @property
    def warnings(self) -> tuple[str, ...]:
        """A warning banner when the independent result needs a person."""
        if not self.needs_inspection:
            return ()
        return (
            (
                f"Independent verifier {self.result.verifier_id!r} disagrees with "
                f"the classical mapping by {self.disagreement_ns / _MS_NS:.1f} ms "
                f"(review threshold {self.tolerance_ns / _MS_NS:.1f} ms). "
                "The classical fit has not been replaced; inspect the competing "
                "pairings below."
            ),
        )


def review_verification(
    mapping: LatencyEstimate,
    result: VerificationResult,
    *,
    tolerance_ns: int,
) -> VerificationReview:
    """Compare a second opinion with a fit and route disagreement to review.

    Equality is agreement: a difference exactly on the caller's threshold does
    not produce a warning.  This function never averages offsets or mutates the
    fit; any accepted final mapping must still be produced by the classical
    estimator.
    """
    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be non-negative")
    disagreement_ns = abs(int(result.proposed_offset_ns) - int(mapping.offset_ns))
    return VerificationReview(
        result=result,
        classical_offset_ns=int(mapping.offset_ns),
        disagreement_ns=disagreement_ns,
        tolerance_ns=int(tolerance_ns),
        needs_inspection=disagreement_ns > tolerance_ns,
    )


def verify_alignment(
    mapping: LatencyEstimate,
    request: VerificationRequest,
    verifier: AlignmentVerifier,
    *,
    tolerance_ns: int,
) -> VerificationReview:
    """Call a verifier and compare its bounded response with ``mapping``."""
    result = verifier.verify(request)
    if abs(result.proposed_offset_ns - request.proposed_offset_ns) > request.search_radius_ns:
        raise VerificationServiceError(
            "verifier result lies outside the requested search radius"
        )
    return review_verification(mapping, result, tolerance_ns=tolerance_ns)


def verification_request_dict(request: VerificationRequest) -> dict[str, Any]:
    return {
        "reference_uri": request.reference_uri,
        "candidate_uri": request.candidate_uri,
        "proposed_offset_ns": request.proposed_offset_ns,
        "search_radius_ns": request.search_radius_ns,
        "metadata": dict(request.metadata),
    }


def verification_result_from_dict(payload: dict[str, Any]) -> VerificationResult:
    try:
        verifier_id = payload["verifier_id"]
        proposed_offset_ns = payload["proposed_offset_ns"]
    except KeyError as exc:
        raise VerificationServiceError(f"verifier result is missing {exc.args[0]!r}") from exc
    confidence = payload.get("confidence")
    details = payload.get("details", {})
    if not isinstance(verifier_id, str):
        raise VerificationServiceError("verifier_id must be a string")
    if not isinstance(proposed_offset_ns, int) or isinstance(proposed_offset_ns, bool):
        raise VerificationServiceError("proposed_offset_ns must be an integer")
    if confidence is not None and (
        not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
    ):
        raise VerificationServiceError("confidence must be a number or null")
    if not isinstance(details, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in details.items()
    ):
        raise VerificationServiceError("details must map strings to strings")
    try:
        return VerificationResult(
            verifier_id=verifier_id,
            proposed_offset_ns=proposed_offset_ns,
            confidence=None if confidence is None else float(confidence),
            details=tuple(details.items()),
        )
    except ValueError as exc:
        raise VerificationServiceError(f"invalid verifier result: {exc}") from exc


def verification_document(
    request: VerificationRequest, review: VerificationReview
) -> dict[str, Any]:
    """Stable JSON-ready document shared by the CLI and library callers."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "request": verification_request_dict(request),
        "result": {
            "verifier_id": review.result.verifier_id,
            "proposed_offset_ns": review.result.proposed_offset_ns,
            "confidence": review.result.confidence,
            "details": dict(review.result.details),
        },
        "review": {
            "classical_offset_ns": review.classical_offset_ns,
            "disagreement_ns": review.disagreement_ns,
            "tolerance_ns": review.tolerance_ns,
            "needs_inspection": review.needs_inspection,
        },
    }


def apply_verification_review(
    context: PageContext, review: VerificationReview
) -> PageContext:
    """Append verifier evidence to an inspector ``PageContext``.

    The local import avoids making the wire-level contract depend on the HTML
    renderer while still providing one canonical inspector integration path.
    """
    from embodied_sync.inspect.render import PageContext

    if not isinstance(context, PageContext):
        raise TypeError("context must be a PageContext")
    return replace(
        context,
        extra_rows=context.extra_rows + review.summary_rows,
        warnings=context.warnings + review.warnings,
    )
