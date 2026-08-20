"""Cross-domain alignment helpers (Milestone 2, D-0029).

Once :class:`~embodied_sync.time.clock_domain.LatencyEstimate` values
exist, alignment must (a) translate source-domain timestamps into the
target's domain before comparing them and (b) lower confidence when
the mapping itself is uncertain. This module is the thin numeric layer
the offline/online engines call into.

Confidence lowering: :func:`cross_domain_confidence_factor` returns
``1.0`` when ``variance_ns == 0`` (perfectly known mapping — no
lowering) and asymptotically approaches ``0`` as variance grows large
relative to the aligner's tolerance. The formula is
``tolerance / (tolerance + variance)`` — same shape as the standard
"noise trades off with tolerance" first-order model, monotone and
bounded so the multiplier is always safe to apply to a scored
confidence in ``[0, 1]``.
"""

from __future__ import annotations

from embodied_sync.time.clock_domain import LatencyEstimate

__all__ = ["cross_domain_confidence_factor"]


def cross_domain_confidence_factor(mapping: LatencyEstimate, tolerance_ns: int) -> float:
    """Confidence multiplier in ``[0, 1]`` for the given mapping.

    ``variance_ns == 0`` returns ``1.0`` (perfect knowledge). Larger
    variance drives the factor toward zero; ``variance_ns ==
    tolerance_ns`` returns ``0.5``. ``tolerance_ns <= 0`` returns
    ``0.0`` because no tolerance means every observation is
    intolerant of any variance.
    """
    if tolerance_ns <= 0:
        return 0.0
    if mapping.variance_ns <= 0:
        return 1.0
    return tolerance_ns / (tolerance_ns + mapping.variance_ns)
