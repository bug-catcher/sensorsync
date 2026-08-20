"""Confidence-gated orchestration for dataset inspection and import."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_sync.core.sample import Sample
from embodied_sync.ingest.execute import execute_import_plan
from embodied_sync.ingest.infer import infer_import
from embodied_sync.ingest.model import ImportPlan, InferenceResult
from embodied_sync.ingest.probe import inspect_dataset

__all__ = ["AmbiguousImportError", "DatasetImportAgent"]


class AmbiguousImportError(ValueError):
    """Raised when evidence is insufficient to select an interpretation."""


@dataclass(frozen=True, slots=True)
class DatasetImportAgent:
    """Inspect, infer, confidence-gate, then deterministically execute."""

    min_confidence: float = 0.75
    min_margin: float = 0.12

    def analyze(self, path: str | Path, *, rate_hz: float | None = None) -> InferenceResult:
        profile = inspect_dataset(path)
        return infer_import(
            profile,
            rate_hz=rate_hz,
            min_confidence=self.min_confidence,
            min_margin=self.min_margin,
        )

    def import_dataset(
        self,
        path: str | Path,
        *,
        plan: ImportPlan | None = None,
        rate_hz: float | None = None,
        accept_ambiguous: bool = False,
        max_episodes: int | None = None,
    ) -> tuple[dict[str, list[Sample]], dict[str, Any], InferenceResult | None]:
        inference: InferenceResult | None = None
        selected = plan
        if selected is None:
            inference = self.analyze(path, rate_hz=rate_hz)
            selected = inference.selected
            if selected is None and accept_ambiguous and inference.candidates:
                selected = inference.candidates[0]
            if selected is None:
                raise AmbiguousImportError(inference.decision)
        run, info = execute_import_plan(path, selected, max_episodes=max_episodes)
        info["import_plan"] = selected.to_dict()
        return run, info, inference
