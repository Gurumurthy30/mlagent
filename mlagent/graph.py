"""The LangGraph pipeline that runs a whole project: first look at the data, then the
experiment loop, then the report.

Every node reads the ledger, calls one agent, and returns only the ledger fields that agent
owns. A wrapper saves the ledger after every node, so a run can always be resumed.
The routing functions are plain functions, so each decision can be tested alone.
"""

import traceback
from collections.abc import Callable
from pathlib import Path

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from mlagent.agents.base import AgentContext, AgentFailedError
from mlagent.agents.data_analyst import run_data_analyst
from mlagent.agents.data_engineer import run_data_engineer
from mlagent.agents.evaluator import run_evaluator
from mlagent.agents.feature_engineer import run_feature_engineer
from mlagent.agents.ml_lead import run_ml_lead
from mlagent.agents.modeler import run_modeler
from mlagent.agents.reviewer import run_reviewer
from mlagent.agents.submitter import copy_predictions, run_submitter
from mlagent.agents.tuner import run_tuner
from mlagent.digest import build_progress_text
from mlagent.events import EventType
from mlagent.llm_client import LlmUnavailableError
from mlagent.memory import load_ledger, save_ledger, write_progress_file
from mlagent.state import Ledger, StopReason
from mlagent.stop_rules import decide_stop_reason

DATA_ENGINEER = "data_engineer"
DATA_ANALYST = "data_analyst"
ML_LEAD = "ml_lead"
STOP_CHECK = "stop_check"
FEATURE_ENGINEER = "feature_engineer"
MODELER = "modeler"
TUNER = "tuner"
EVALUATOR = "evaluator"
REVIEWER = "reviewer"
DATA_ANALYST_REVISIT = "data_analyst_revisit"
SUBMITTER = "submitter"

ALL_NODE_NAMES = [
    DATA_ENGINEER,
    DATA_ANALYST,
    ML_LEAD,
    STOP_CHECK,
    FEATURE_ENGINEER,
    MODELER,
    TUNER,
    EVALUATOR,
    REVIEWER,
    DATA_ANALYST_REVISIT,
    SUBMITTER,
]

NODE_AFTER_STOP_CHECK_BY_FOCUS = {
    "start": FEATURE_ENGINEER,
    "features": FEATURE_ENGINEER,
    "model": MODELER,
    "tune": TUNER,
    "data": DATA_ANALYST_REVISIT,
}

NODE_BY_PHASE = {
    "data_engineer": DATA_ENGINEER,
    "data_analyst": DATA_ANALYST,
    "ml_lead": ML_LEAD,
    "experiment_loop": STOP_CHECK,
    "submitter": SUBMITTER,
    "done": END,
}

STEPS_PER_ROUND_ALLOWANCE = 12
EXTRA_STEPS_ALLOWANCE = 60

NodeFunction = Callable[[Ledger], dict]
Router = Callable[[Ledger], str]


def run_is_over(ledger: Ledger) -> bool:
    return ledger.status == "failed" or ledger.phase == "done"


def stop_when_over(choose_next_node: Router) -> Router:
    """Wrap a router: a failed or finished run ends, and a stopped loop goes to the Submitter."""

    def route(ledger: Ledger) -> str:
        if run_is_over(ledger):
            return END
        if ledger.phase == "submitter":
            return SUBMITTER
        return choose_next_node(ledger)

    return route


def route_from_phase(ledger: Ledger) -> str:
    """Where a run starts or resumes: the step its saved phase says is next."""
    return NODE_BY_PHASE[ledger.phase]


def route_after_stop_check(ledger: Ledger) -> str:
    """The next agent for the focus. It also picks up a round that was interrupted halfway."""
    runs_of_this_round = [run for run in ledger.runs if run.iteration == ledger.iteration]
    if runs_of_this_round:
        has_unchecked_run = any(
            not run.crashed and ledger.find_evaluation(run.experiment_id) is None
            for run in runs_of_this_round
        )
        return EVALUATOR if has_unchecked_run else REVIEWER
    if ledger.focus == "start" and ledger.current_feature_version >= 1:
        return MODELER
    return NODE_AFTER_STOP_CHECK_BY_FOCUS[ledger.focus]


def route_after_features(ledger: Ledger) -> str:
    """A features round that kept nothing has no new run to train, so the Reviewer goes next."""
    return MODELER if ledger.last_features_changed else REVIEWER


def route_after_modeler(ledger: Ledger) -> str:
    newest_run = ledger.runs[-1] if ledger.runs else None
    if newest_run is None or newest_run.crashed:
        return REVIEWER
    return EVALUATOR


def route_after_tuner(ledger: Ledger) -> str:
    runs_of_this_round = [run for run in ledger.runs if run.iteration == ledger.iteration]
    if any(not run.crashed for run in runs_of_this_round):
        return EVALUATOR
    return REVIEWER


def build_failure_updates(
    ledger: Ledger, node_name: str, stop_reason: StopReason, message: str
) -> dict:
    """What to record when a node cannot continue. Approved work is still handed in."""
    updates: dict = {"stop_reason": stop_reason, "error_message": message}
    if node_name != SUBMITTER and ledger.approved_runs():
        updates["phase"] = "submitter"
    else:
        updates["status"] = "failed"
    return updates


def make_node(
    node_name: str, node_function: NodeFunction, agent_context: AgentContext
) -> NodeFunction:
    """Wrap a node: turn agent failures into recorded updates and save the ledger afterwards."""

    def run_node(ledger: Ledger) -> dict:
        try:
            updates = node_function(ledger)
        except LlmUnavailableError as error:
            updates = build_failure_updates(ledger, node_name, "llm_unavailable", str(error))
            agent_context.event_bus.emit(EventType.ERROR, node_name, str(error))
        except AgentFailedError as error:
            updates = build_failure_updates(ledger, node_name, "agent_failed", str(error))
            agent_context.event_bus.emit(EventType.ERROR, node_name, str(error))

        saved_ledger = ledger.model_copy(update=updates)
        save_ledger(agent_context.workspace, saved_ledger)
        write_progress_file(agent_context.workspace, build_progress_text(saved_ledger))
        return updates

    return run_node


def build_nodes(agent_context: AgentContext) -> dict[str, NodeFunction]:
    max_experiment_loops = agent_context.config.limits.max_experiment_loops
    event_bus = agent_context.event_bus

    def data_engineer_node(ledger: Ledger) -> dict:
        return {"brief": run_data_engineer(ledger, agent_context), "phase": "data_analyst"}

    def data_analyst_node(ledger: Ledger) -> dict:
        return {"eda": run_data_analyst(ledger, agent_context), "phase": "ml_lead"}

    def ml_lead_node(ledger: Ledger) -> dict:
        return {"plan": run_ml_lead(ledger, agent_context), "phase": "experiment_loop"}

    def stop_check_node(ledger: Ledger) -> dict:
        stop_reason = decide_stop_reason(ledger, max_experiment_loops)
        if stop_reason is not None:
            event_bus.emit(EventType.DECISION, "Controller", f"stopping the loop: {stop_reason}")
            return {"stop_reason": stop_reason, "phase": "submitter"}
        event_bus.emit(
            EventType.INFO,
            "Controller",
            f"round {ledger.iteration + 1} of {max_experiment_loops}, focus {ledger.focus}",
        )
        return {}

    def feature_engineer_node(ledger: Ledger) -> dict:
        result = run_feature_engineer(ledger, agent_context)
        return {
            "features": result.features,
            "current_feature_version": result.current_feature_version,
            "last_features_changed": result.changed,
        }

    def modeler_node(ledger: Ledger) -> dict:
        return {"runs": [*ledger.runs, run_modeler(ledger, agent_context)]}

    def tuner_node(ledger: Ledger) -> dict:
        tuned_runs = run_tuner(ledger, agent_context)
        return {"runs": [*ledger.runs, *tuned_runs], "last_tuned_iteration": ledger.iteration}

    def evaluator_node(ledger: Ledger) -> dict:
        best_run_before = ledger.best_approved_run()
        working_ledger = ledger.model_copy(deep=True)
        unchecked_runs = [
            run
            for run in ledger.runs
            if run.iteration == ledger.iteration
            and not run.crashed
            and ledger.find_evaluation(run.experiment_id) is None
        ]
        for run in unchecked_runs:
            evaluation = run_evaluator(working_ledger, agent_context, run.experiment_id)
            working_ledger.evaluations.append(evaluation)

        best_run_after = working_ledger.best_approved_run()
        has_new_best = best_run_after is not None and (
            best_run_before is None or best_run_after.experiment_id != best_run_before.experiment_id
        )
        if has_new_best:
            saved_path = copy_predictions(Path(best_run_after.test_predictions_path), agent_context)
            event_bus.emit(
                EventType.INFO,
                "Controller",
                f"new best {best_run_after.experiment_id}, predictions saved to {saved_path.name}",
            )
        return {"evaluations": working_ledger.evaluations}

    def reviewer_node(ledger: Ledger) -> dict:
        review = run_reviewer(ledger, agent_context)
        return {
            "reviews": [*ledger.reviews, review],
            "focus": review.focus,
            "iteration": review.iteration,
        }

    def data_analyst_revisit_node(ledger: Ledger) -> dict:
        question = ledger.reviews[-1].question if ledger.reviews else ""
        eda = run_data_analyst(ledger, agent_context, question=question)
        return {
            "eda": eda,
            "data_reentries_used": ledger.data_reentries_used + 1,
            "focus": "features",
        }

    def submitter_node(ledger: Ledger) -> dict:
        return {
            "final": run_submitter(ledger, agent_context),
            "status": "done",
            "phase": "done",
        }

    node_functions = {
        DATA_ENGINEER: data_engineer_node,
        DATA_ANALYST: data_analyst_node,
        ML_LEAD: ml_lead_node,
        STOP_CHECK: stop_check_node,
        FEATURE_ENGINEER: feature_engineer_node,
        MODELER: modeler_node,
        TUNER: tuner_node,
        EVALUATOR: evaluator_node,
        REVIEWER: reviewer_node,
        DATA_ANALYST_REVISIT: data_analyst_revisit_node,
        SUBMITTER: submitter_node,
    }
    return {
        name: make_node(name, function, agent_context) for name, function in node_functions.items()
    }


def targets_with_exits(*node_names: str) -> dict[str, str]:
    """A router's possible targets. Every router can also end the run or go to the Submitter."""
    return {name: name for name in [*node_names, SUBMITTER, END]}


def build_pipeline(agent_context: AgentContext):
    graph = StateGraph(Ledger)
    for node_name, node_function in build_nodes(agent_context).items():
        graph.add_node(node_name, node_function)

    graph.add_conditional_edges(
        START, route_from_phase, {name: name for name in NODE_BY_PHASE.values()}
    )

    fixed_next_node = {
        DATA_ENGINEER: DATA_ANALYST,
        DATA_ANALYST: ML_LEAD,
        ML_LEAD: STOP_CHECK,
        EVALUATOR: REVIEWER,
        REVIEWER: STOP_CHECK,
        DATA_ANALYST_REVISIT: FEATURE_ENGINEER,
    }
    for source_name, next_name in fixed_next_node.items():
        router = stop_when_over(lambda ledger, next_name=next_name: next_name)
        graph.add_conditional_edges(source_name, router, targets_with_exits(next_name))

    chosen_next_node = {
        STOP_CHECK: (route_after_stop_check, list(NODE_AFTER_STOP_CHECK_BY_FOCUS.values())),
        FEATURE_ENGINEER: (route_after_features, [MODELER, REVIEWER]),
        MODELER: (route_after_modeler, [EVALUATOR, REVIEWER]),
        TUNER: (route_after_tuner, [EVALUATOR, REVIEWER]),
    }
    for source_name, (router, target_names) in chosen_next_node.items():
        graph.add_conditional_edges(
            source_name, stop_when_over(router), targets_with_exits(*target_names)
        )

    graph.add_edge(SUBMITTER, END)
    return graph.compile()


def record_unexpected_error(error: Exception, agent_context: AgentContext) -> None:
    """Keep the last saved ledger, mark it failed, and write the full traceback to the logs."""
    workspace = agent_context.workspace
    traceback_path = workspace.logs_folder / "error_traceback.txt"
    traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
    message = f"Unexpected error: {error!r}. The traceback is in {traceback_path}."
    agent_context.event_bus.emit(EventType.ERROR, "Controller", message)
    try:
        last_ledger = load_ledger(workspace)
    except Exception:
        return
    save_ledger(
        workspace, last_ledger.model_copy(update={"status": "failed", "error_message": message})
    )


def run_pipeline(ledger: Ledger, agent_context: AgentContext) -> Ledger:
    """Run or resume the project from the phase saved in the ledger. Returns the final ledger."""
    max_experiment_loops = agent_context.config.limits.max_experiment_loops
    step_limit = max_experiment_loops * STEPS_PER_ROUND_ALLOWANCE + EXTRA_STEPS_ALLOWANCE
    pipeline = build_pipeline(agent_context)
    try:
        pipeline.invoke(ledger, config={"recursion_limit": step_limit})
    except GraphRecursionError as error:
        record_unexpected_error(error, agent_context)
    except Exception as error:
        record_unexpected_error(error, agent_context)
        raise
    return load_ledger(agent_context.workspace)
