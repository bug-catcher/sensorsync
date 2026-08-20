"""Serializable evidence, profile, and import-plan types.

The inference layer never executes free-form generated code. It emits an
``ImportPlan`` naming a registered deterministic executor plus JSON-compatible
parameters. Plans can therefore be reviewed, versioned, and replayed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DatasetProfile",
    "Evidence",
    "ImportPlan",
    "InferenceResult",
    "load_import_plan",
    "load_inference_result",
    "save_json_document",
]

INGEST_FORMAT_VERSION = 0


def _require_dict(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _require_string_list(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a JSON array of strings")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class Evidence:
    """One inspectable reason contributing to an import interpretation."""

    code: str
    message: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "score": self.score}

    @classmethod
    def from_dict(cls, document: object) -> "Evidence":
        data = _require_dict(document, name="evidence")
        return cls(
            code=str(data["code"]),
            message=str(data["message"]),
            score=float(data["score"]),
        )


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Read-only structural observations about a local dataset."""

    root: str
    path_kind: str
    signatures: tuple[str, ...]
    facts: dict[str, Any]
    warnings: tuple[str, ...] = ()
    format_version: int = INGEST_FORMAT_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "type": "dataset_profile",
            "root": self.root,
            "path_kind": self.path_kind,
            "signatures": list(self.signatures),
            "facts": self.facts,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, document: object) -> "DatasetProfile":
        data = _require_dict(document, name="dataset profile")
        if data.get("format_version") != INGEST_FORMAT_VERSION:
            raise ValueError(
                f"unsupported dataset profile format_version "
                f"{data.get('format_version')!r}"
            )
        if data.get("type") != "dataset_profile":
            raise ValueError(f"expected dataset_profile, got {data.get('type')!r}")
        return cls(
            root=str(data["root"]),
            path_kind=str(data["path_kind"]),
            signatures=_require_string_list(data.get("signatures", []), name="signatures"),
            facts=_require_dict(data.get("facts", {}), name="facts"),
            warnings=_require_string_list(data.get("warnings", []), name="warnings"),
        )


@dataclass(frozen=True, slots=True)
class ImportPlan:
    """A deterministic executor choice with evidence and parameters."""

    executor: str
    confidence: float
    parameters: dict[str, Any]
    evidence: tuple[Evidence, ...]
    warnings: tuple[str, ...] = ()
    format_version: int = INGEST_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "type": "import_plan",
            "executor": self.executor,
            "confidence": self.confidence,
            "parameters": self.parameters,
            "evidence": [item.to_dict() for item in self.evidence],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, document: object) -> "ImportPlan":
        data = _require_dict(document, name="import plan")
        if data.get("format_version") != INGEST_FORMAT_VERSION:
            raise ValueError(
                f"unsupported import plan format_version {data.get('format_version')!r}"
            )
        if data.get("type") != "import_plan":
            raise ValueError(f"expected import_plan, got {data.get('type')!r}")
        raw_evidence = data.get("evidence", [])
        if not isinstance(raw_evidence, list):
            raise ValueError("import plan evidence must be a JSON array")
        return cls(
            executor=str(data["executor"]),
            confidence=float(data["confidence"]),
            parameters=_require_dict(data.get("parameters", {}), name="parameters"),
            evidence=tuple(Evidence.from_dict(item) for item in raw_evidence),
            warnings=_require_string_list(data.get("warnings", []), name="warnings"),
        )


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Ranked candidate plans and the optional confidence-gated selection."""

    profile: DatasetProfile
    candidates: tuple[ImportPlan, ...]
    selected: ImportPlan | None
    decision: str
    format_version: int = INGEST_FORMAT_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "type": "import_inference",
            "profile": self.profile.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected": self.selected.to_dict() if self.selected is not None else None,
            "decision": self.decision,
        }

    @classmethod
    def from_dict(cls, document: object) -> "InferenceResult":
        data = _require_dict(document, name="import inference")
        if data.get("format_version") != INGEST_FORMAT_VERSION:
            raise ValueError(
                f"unsupported import inference format_version "
                f"{data.get('format_version')!r}"
            )
        if data.get("type") != "import_inference":
            raise ValueError(f"expected import_inference, got {data.get('type')!r}")
        raw_candidates = data.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise ValueError("import inference candidates must be a JSON array")
        raw_selected = data.get("selected")
        return cls(
            profile=DatasetProfile.from_dict(data["profile"]),
            candidates=tuple(ImportPlan.from_dict(item) for item in raw_candidates),
            selected=(
                ImportPlan.from_dict(raw_selected) if raw_selected is not None else None
            ),
            decision=str(data.get("decision", "")),
        )


def save_json_document(document: DatasetProfile | ImportPlan | InferenceResult, path: str | Path) -> Path:
    """Write an inference-layer document as stable, human-readable JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _load_json(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return _require_dict(document, name=str(path))


def load_inference_result(path: str | Path) -> InferenceResult:
    return InferenceResult.from_dict(_load_json(path))


def load_import_plan(path: str | Path) -> ImportPlan:
    """Load either a direct plan or the selected plan in an inference report."""

    document = _load_json(path)
    if document.get("type") == "import_plan":
        return ImportPlan.from_dict(document)
    inference = InferenceResult.from_dict(document)
    if inference.selected is None:
        raise ValueError(
            f"{path!s} has no selected import plan; review its candidates or "
            "run import-auto with --accept-ambiguous"
        )
    return inference.selected
