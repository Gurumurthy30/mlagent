from __future__ import annotations

import functools
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from langgraph.graph import END

from mlagent.agents.base import AgentContext
from mlagent.config import AppConfig, LimitsConfig
from mlagent.events import EventBus
from mlagent.graph import (
    build_failure_updates,
    build_pipeline,
    route_after_data_analyst,
    route_after_features,
    route_after_model,
    route_after_stop_check,
    route_after_tune,
    route_from_phase,
)
from mlagent.memory import Workspace
from mlagent.state import ExperimentRun, Ledger, Review
from mlagent.stop_rules import (
    apply_reviewer_guards,
    decide_stop_reason,
)
from tests.conftest import FakeLlmClient, st_ledger


@functools.lru_cache(maxsize=1)
def get_compiled_graph_nodes() -> set[str]:
    """Dynamically get all node names from the compiled StateGraph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Workspace(Path(tmpdir) / "ws")
        ws.create_folders()
        from mlagent.environment_probe import EnvironmentInfo

        ctx = AgentContext(
            workspace=ws,
            event_bus=EventBus(),
            code_runner=None,  # type: ignore[arg-type]
            llm_client=FakeLlmClient(),  # type: ignore[arg-type]
            environment=EnvironmentInfo(),
            config=AppConfig(),
        )
        pipeline = build_pipeline(ctx)
        nodes = set(pipeline.nodes.keys())
        nodes.add(END)
        return nodes


@pytest.mark.unit
@given(ledger=st_ledger())
def test_all_routers_return_valid_graph_nodes(ledger: Ledger) -> None:
    """Every router function returns a node name that exists in the compiled graph."""
    compiled_nodes = get_compiled_graph_nodes()

    target = route_from_phase(ledger)
    assert target in compiled_nodes

    target = route_after_stop_check(ledger)
    assert target in compiled_nodes

    target = route_after_features(ledger)
    assert target in compiled_nodes

    target = route_after_model(ledger)
    assert target in compiled_nodes

    target = route_after_tune(ledger)
    assert target in compiled_nodes

    target = route_after_data_analyst(ledger)
    assert target in compiled_nodes


@pytest.mark.unit
def test_stop_reason_budget_limit() -> None:
    """Stop rule ends the run when iteration reaches configured loop limit."""
    limits = LimitsConfig(max_experiment_loops=5)
    ledger = Ledger(user_task="test", iteration=5)
    assert decide_stop_reason(ledger, limits) == "budget"

    ledger.iteration = 4
    assert decide_stop_reason(ledger, limits) != "budget"


@pytest.mark.unit
def test_stop_reason_crash_limit() -> None:
    """Stop rule ends the run when trailing crashes reach configured limit."""
    limits = LimitsConfig(max_consecutive_crashes=3)
    crashed_runs = [
        ExperimentRun(experiment_id=f"run_{i}", error=f"error_{i}")
        for i in range(1, 4)
    ]
    ledger = Ledger(user_task="test", runs=crashed_runs)
    assert decide_stop_reason(ledger, limits) == "crashes"


@pytest.mark.unit
def test_stop_reason_plateau_patience() -> None:
    """Stop rule ends the run when rounds with no gain reach plateau patience."""
    limits = LimitsConfig(plateau_rounds=3)
    reviews = [
        Review(iteration=1, focus="features"),
        Review(iteration=2, focus="model"),
        Review(iteration=3, focus="features"),
    ]
    # No runs marked counted_gain=True
    runs = [
        ExperimentRun(experiment_id="run_1", iteration=1, counted_gain=False),
        ExperimentRun(experiment_id="run_2", iteration=2, counted_gain=False),
        ExperimentRun(experiment_id="run_3", iteration=3, counted_gain=False),
    ]
    ledger = Ledger(user_task="test", reviews=reviews, runs=runs)
    assert decide_stop_reason(ledger, limits) == "plateau"


@pytest.mark.unit
def test_stop_reason_target_score_reached() -> None:
    """Stop rule ends the run when target score is achieved."""
    ledger = Ledger(user_task="test")
    ledger.plan.target_score = 0.90
    ledger.brief.higher_is_better = True

    # Run achieving 0.92
    ledger.runs = [
        ExperimentRun(
            experiment_id="run_1",
            score_mean=0.92,
            verdict="approve",
        )
    ]
    assert decide_stop_reason(ledger) == "target"


@pytest.mark.unit
def test_failed_node_diverts_to_submit_if_approved_runs_exist() -> None:
    """When an error happens, pipeline diverts to submit if approved work exists."""
    approved_run = ExperimentRun(
        experiment_id="run_001",
        score_mean=0.85,
        verdict="approve",
    )
    ledger_with_approved = Ledger(user_task="test", runs=[approved_run])
    updates = build_failure_updates(
        ledger_with_approved,
        node_name="model",
        stop_reason="llm_unavailable",
        message="Ollama dropped connection",
    )
    assert updates["phase"] == "submit"
    assert updates["stop_reason"] == "llm_unavailable"
    assert "status" not in updates or updates["status"] != "failed"

    # If no approved runs exist, it must record failure
    ledger_without_approved = Ledger(user_task="test", runs=[])
    failed_updates = build_failure_updates(
        ledger_without_approved,
        node_name="model",
        stop_reason="llm_unavailable",
        message="Ollama dropped connection",
    )
    assert failed_updates["status"] == "failed"
    assert failed_updates.get("phase") != "submit"


@pytest.mark.unit
def test_reviewer_guard_prevents_early_finish() -> None:
    """Reviewer cannot choose 'finish' before the minimum round limit."""
    ledger = Ledger(
        user_task="test",
        reviews=[Review(iteration=1, focus="features", reviewed_focus="features")],
    )
    review = Review(iteration=2, focus="finish")

    guarded = apply_reviewer_guards(ledger, review)
    assert guarded.focus != "finish"
    assert "too early" in guarded.guard_note
