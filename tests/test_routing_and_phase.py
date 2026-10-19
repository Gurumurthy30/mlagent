"""Unit tests for pipeline graph routing, termination boundaries, and Phase literals."""

from mlagent.graph.routing import route_after_evaluator
from mlagent.graph.state import ExperimentRun, Ledger


def test_ledger_phase_includes_summarizer() -> None:
    """Ledger allows phase='summarizer' and roundtrips without Pydantic ValidationError."""
    ledger = Ledger(user_task="Predict prices", phase="summarizer")
    assert ledger.phase == "summarizer"

    # Verify model_validate works on dict with phase="summarizer"
    data = ledger.model_dump()
    reloaded = Ledger.model_validate(data)
    assert reloaded.phase == "summarizer"


def test_route_after_evaluator_stops_strictly_at_max_loops() -> None:
    """When max_experiment_loops=12, after 12 runs it routes strictly to finalizer, not running a 13th."""
    ledger = Ledger(user_task="House Prices")

    # Populate 12 runs (from iteration 0 to 11)
    for it in range(12):
        ledger.iteration = it
        run = ExperimentRun(
            experiment_id=f"experiment_{it+1:03d}",
            iteration=it,
            model_family="ridge",
            cv_mean=35000.0 - it * 100,
        )
        ledger.runs.append(run)

    assert len(ledger.runs) == 12
    assert ledger.iteration == 11

    # Route decision after run 12
    decision = route_after_evaluator(ledger, max_experiment_loops=12)
    assert decision == "finalizer"
    assert ledger.stop_reason == "max_experiment_loops_reached"


def test_route_after_evaluator_allows_running_up_to_max() -> None:
    """When runs are below max_experiment_loops, it continues to ml_lead."""
    ledger = Ledger(user_task="House Prices")

    for it in range(11):
        ledger.iteration = it
        run = ExperimentRun(
            experiment_id=f"experiment_{it+1:03d}",
            iteration=it,
            model_family="ridge",
            cv_mean=35000.0,
        )
        ledger.runs.append(run)

    assert len(ledger.runs) == 11
    decision = route_after_evaluator(ledger, max_experiment_loops=12)
    assert decision == "ml_lead"
