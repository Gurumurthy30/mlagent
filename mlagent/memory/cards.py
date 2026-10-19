"""Typed memory card models for mlagent v3.

Every state write is captured as an immutable, typed memory card with a compressed summary,
structured payload, artifact pointers, and priority ranking.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

# Base Pydantic wrapper from state schema
from mlagent.graph.state import BaseModel

CardPriority = Literal["core", "high", "medium", "low"]


class MemoryCard(BaseModel):
    """Base class for all typed memory cards."""

    tag: str
    card_id: str
    owner: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    iteration: int = 0
    memory_summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: list[str] = Field(default_factory=list)
    priority: CardPriority = "medium"

    @field_validator("payload", mode="after")
    @classmethod
    def validate_non_empty_payload(cls, val: dict[str, Any]) -> dict[str, Any]:
        # Enforce that cards carry structured payload data
        if not val:
            raise ValueError(f"Payload cannot be empty for card {cls.__name__}")
        return val


class ProjectBriefCard(MemoryCard):
    """Compressed problem definition and dataset brief."""

    tag: str = "project_brief"
    priority: CardPriority = "core"


class EdaFindingCard(MemoryCard):
    """Empirical quantitative observation or anomaly."""

    tag: str = "eda_finding"
    priority: CardPriority = "medium"


class PrepSpecCard(MemoryCard):
    """Preprocessing blueprint, duplicate rules, and column specs."""

    tag: str = "prep_spec"
    priority: CardPriority = "high"


class ValidationPlanCard(MemoryCard):
    """Locked validation protocol, partitioning scheme, and baseline plan."""

    tag: str = "validation_plan"
    priority: CardPriority = "core"


class FeatureVersionCard(MemoryCard):
    """Feature engineering transformation version and column manifest."""

    tag: str = "feature_version"
    priority: CardPriority = "high"


class ExperimentRunCard(MemoryCard):
    """Trained model metrics, CV scores, and diagnostic report summary."""

    tag: str = "experiment_run"
    priority: CardPriority = "high"


class EvaluationVerdictCard(MemoryCard):
    """Independent generalization audit and Champion ratchet verdict."""

    tag: str = "evaluation_verdict"
    priority: CardPriority = "high"


class ChampionPointerCard(MemoryCard):
    """Active winning model pointer. The single mutable card in the memory store."""

    tag: str = "champion_pointer"
    priority: CardPriority = "core"


class HypothesisCard(MemoryCard):
    """Iterative ML Lead hypothesis formulation."""

    tag: str = "hypothesis"
    priority: CardPriority = "high"


class LessonLearnedCard(MemoryCard):
    """Empirical truth established from an approved or refuted hypothesis."""

    tag: str = "lesson_learned"
    priority: CardPriority = "core"


class FinalDeliverableCard(MemoryCard):
    """Final packaged model or ensemble, verification checks, and drift specs."""

    tag: str = "final_deliverable"
    priority: CardPriority = "core"


class WorkingScratchpadCard(MemoryCard):
    """Ephemeral per-node scratchpad for multi-step tasks."""

    tag: str = "working_scratchpad"
    priority: CardPriority = "low"


TAG_TO_CARD_CLS: dict[str, type[MemoryCard]] = {
    "project_brief": ProjectBriefCard,
    "eda_finding": EdaFindingCard,
    "prep_spec": PrepSpecCard,
    "validation_plan": ValidationPlanCard,
    "feature_version": FeatureVersionCard,
    "experiment_run": ExperimentRunCard,
    "evaluation_verdict": EvaluationVerdictCard,
    "champion_pointer": ChampionPointerCard,
    "hypothesis": HypothesisCard,
    "lesson_learned": LessonLearnedCard,
    "final_deliverable": FinalDeliverableCard,
    "working_scratchpad": WorkingScratchpadCard,
}
