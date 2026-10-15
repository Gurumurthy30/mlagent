"""The LangGraph pipeline that runs the whole project.

Loop architecture:
data_engineer -> data_analyst (self-loop until eda.done) -> ml_lead -> stop_check
experiment loop:  stop_check -> features | model | tune | data_analyst (one revisit round)
                  features -> model -> evaluate -> review -> stop_check
                  tune -> evaluate -> review -> stop_check
when the loop stops:  submit (mechanical) -> report (written by the Reviewer) -> END
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import traceback

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from mlagent.agents.base import AgentContext, AgentFailedError
from mlagent.agents.data_analyst import run_data_analyst
from mlagent.agents.data_engineer import run_data_engineer
from mlagent.agents.evaluator import run_evaluator
from mlagent.agents.feature_engineer import run_feature_engineer
from mlagent.agents.ml_lead import run_ml_lead
from mlagent.agents.modeler import run_modeler
from mlagent.agents.reviewer import run_report, run_reviewer
from mlagent.agents.submitter import copy_predictions, run_submitter
from mlagent.agents.tuner import run_tuner
from mlagent.digest import build_progress_text
from mlagent.events import EventType
from mlagent.llm_client import LlmUnavailableError
from mlagent.memory import load_ledger, save_ledger, write_progress_file
from mlagent.state import Ledger, StopReason
from mlagent.stop_rules import decide_stop_reason

STEPS_PER_ROUND_ALLOWANCE = 12
EXTRA_STEPS_ALLOWANCE = 60

NodeFunction = Callable[[Ledger], dict]
Router = Callable[[Ledger], str]


def run_is_over(ledger: Ledger) -> bool:
    return ledger.status == "failed" or ledger.phase == "done"


def stop_when_over(choose_next_node: Router) -> Router:
    """Wrap a router: a failed or finished run ends, and submit/report are respected."""

    def route(ledger: Ledger) -> str:
        if run_is_over(ledger):
            return END
        if ledger.phase in ("submitter", "submit"):
            return "submit"
        if ledger.phase == "report":
            return "report"
        return choose_next_node(ledger)

    return route


def route_from_phase(ledger: Ledger) -> str:
    """Where a run starts or resumes based on saved phase."""
    if ledger.phase == "data_engineer":
        return "data_engineer"
    if ledger.phase == "data_analyst":
        return "data_analyst"
    if ledger.phase == "ml_lead":
        return "ml_lead"
    if ledger.phase == "experiment_loop":
        return "stop_check"
    if ledger.phase in ("submitter", "submit"):
        return "submit"
    if ledger.phase == "report":
        return "report"
    return END


def route_after_stop_check(ledger: Ledger) -> str:
    """Choose next agent based on focus or resume interrupted round."""
    # Resume interrupted round if any runs exist this round
    if ledger.runs_this_round():
        if ledger.unevaluated_runs():
            return "evaluate"
        return "review"

    if ledger.focus == "start":
        return "model" if ledger.current_feature_version >= 1 else "feature_engineer"
    if ledger.focus == "data":
        return "data_analyst"
    if ledger.focus == "features":
        return "feature_engineer"
    if ledger.focus == "model":
        return "model"
    if ledger.focus == "tune":
        return "tune"
    return "feature_engineer"


def route_after_features(ledger: Ledger) -> str:
    return "model" if ledger.last_features_changed else "review"


def route_after_model(ledger: Ledger) -> str:
    runs = ledger.runs_this_round()
    if not runs or runs[-1].crashed:
        return "review"
    return "evaluate"


def route_after_tune(ledger: Ledger) -> str:
    runs = ledger.runs_this_round()
    if any(not r.crashed for r in runs):
        return "evaluate"
    return "review"


def route_after_data_analyst(ledger: Ledger, max_eda_loops: int = 4) -> str:
    """Route after an EDA round: loop back, proceed to ml_lead, or to features on revisit."""
    if not ledger.eda.done and ledger.eda.eda_loops_used < max_eda_loops:
        return "data_analyst"
    if ledger.phase == "data_analyst":
        return "ml_lead"
    return "feature_engineer"


def build_failure_updates(
    ledger: Ledger, node_name: str, stop_reason: StopReason, message: str
) -> dict:
    """What to record when a node cannot continue. Approved work is still handed in."""
    updates: dict = {"stop_reason": stop_reason, "error_message": message}
    if node_name not in ("submit", "report") and ledger.approved_runs():
        updates["phase"] = "submit"
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


def build_fallback_report(ledger: Ledger, agent_context: AgentContext) -> tuple[str, str]:
    """Code-built fallback report when report LLM call fails."""
    best = ledger.best_approved_run()
    score_str = f"{best.score_mean:.4f}" if best and best.score_mean is not None else "none"
    headline = (
        f"The team delivered model {best.model_family if best else 'unknown'} "
        f"achieving validation score {score_str}. Predictions are stored in {ledger.final.predictions_path or 'artifacts'}."
    )
    table_lines = [
        "| Run | Model | Features | Score | Gain vs Best | Status | Summary |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in ledger.runs:
        s = f"{r.score_mean:.4f}" if r.score_mean is not None else "n/a"
        g = f"{r.gain_vs_best:+.4f}" if r.gain_vs_best is not None else "-"
        table_lines.append(
            f"| {r.experiment_id} | {r.model_family} | v{r.feature_version} | {s} | {g} | {r.status} | {r.summary} |"
        )

    sections = [
        f"# Project Final Report: {ledger.user_task or 'ML Solution'}",
        "",
        "## In brief",
        headline,
        "",
        "## Key numbers",
        f"- Total runs: {len(ledger.runs)}",
        f"- Approved runs: {len(ledger.approved_runs())}",
        f"- Best score: {score_str}",
        f"- Primary metric: {ledger.brief.primary_metric or 'unknown'}",
        "",
        "## What the data showed",
        ledger.eda.summary or "EDA completed.",
        "",
        "## What the team did",
        "\n".join(f"- {c}" for c in ledger.round_cards()) or "Baseline and experimental modeling.",
        "",
        "## The final predictions",
        f"Predictions file: {ledger.final.predictions_path or 'not generated'}",
        "",
        "## How far to trust it",
        f"Validation checks: {len(ledger.approved_runs())} runs verified on cross-validation splits.",
        "",
        "## Caveats and next steps",
        "1. Verify performance on real test data distributions.",
        "2. Test additional domain-specific feature interactions.",
        "",
        "## Experiment Runs",
        "\n".join(table_lines),
    ]
    return "\n".join(sections), headline


def build_nodes(agent_context: AgentContext) -> dict[str, NodeFunction]:
    limits = agent_context.config.limits
    event_bus = agent_context.event_bus

    def run_ml_lead_node(ledger: Ledger) -> dict:
        plan = run_ml_lead(ledger, agent_context)
        baseline_short = (
            plan.baseline_description[:50] + "..."
            if len(plan.baseline_description) > 50
            else plan.baseline_description
        )
        target_str = str(plan.target_score) if plan.target_score is not None else "none"
        event_bus.emit(
            EventType.INFO,
            "ml_lead",
            f"Plan: {plan.validation_scheme}, baseline: {baseline_short}, target score: {target_str}",
        )
        return {"plan": plan, "phase": "experiment_loop"}

    def run_stop_check_node(ledger: Ledger) -> dict:
        stop_reason = decide_stop_reason(ledger, limits)
        if stop_reason is not None:
            event_bus.emit(EventType.DECISION, "Controller", f"stopping the loop: {stop_reason}")
            return {"stop_reason": stop_reason, "phase": "submit"}
        event_bus.emit(
            EventType.INFO,
            "Controller",
            f"round {ledger.iteration + 1} of {limits.max_experiment_loops}, focus {ledger.focus}",
        )
        return {}

    def run_evaluate_node(ledger: Ledger) -> dict:
        best_run_before = ledger.best_approved_run()
        working_runs = list(ledger.runs)

        unevaluated_indices = [
            i
            for i, r in enumerate(working_runs)
            if r.iteration == ledger.iteration and not r.crashed and r.verdict is None
        ]

        for idx in unevaluated_indices:
            run_to_eval = working_runs[idx]
            eval_ledger = ledger.model_copy(update={"runs": working_runs})
            evaluated_run = run_evaluator(eval_ledger, agent_context, run_to_eval)
            working_runs[idx] = evaluated_run
            event_bus.emit(EventType.INFO, "evaluator", evaluated_run.headline)

        temp_ledger = ledger.model_copy(update={"runs": working_runs})
        best_run_after = temp_ledger.best_approved_run()
        has_new_best = best_run_after is not None and (
            best_run_before is None or best_run_after.experiment_id != best_run_before.experiment_id
        )
        if has_new_best:
            try:
                saved_path = copy_predictions(
                    Path(best_run_after.test_predictions_path), agent_context
                )
                event_bus.emit(
                    EventType.INFO,
                    "Controller",
                    f"new best {best_run_after.experiment_id}, predictions saved to {saved_path.name}",
                )
            except Exception as e:
                event_bus.emit(EventType.WARNING, "Controller", f"could not copy predictions: {e}")

        return {"runs": working_runs}

    def run_review_node(ledger: Ledger) -> dict:
        review = run_reviewer(ledger, agent_context)
        updates: dict = {
            "reviews": [*ledger.reviews, review],
            "focus": review.focus,
            "iteration": review.iteration,
        }
        temp_ledger = ledger.model_copy(update=updates)
        cards = temp_ledger.round_cards()
        latest_card = cards[-1] if cards else f"Round {review.iteration} [{review.reviewed_focus}]"
        event_bus.emit(EventType.DECISION, "reviewer", f"{latest_card}\nNext: {review.focus}")

        if review.focus == "data":
            eda_copy = ledger.eda.model_copy(deep=True)
            if review.question:
                eda_copy.open_questions.append(review.question)
            eda_copy.done = False
            updates["eda"] = eda_copy
            updates["data_reentries_used"] = ledger.data_reentries_used + 1

        return updates

    def run_report_node(ledger: Ledger) -> dict:
        try:
            report_text, headline = run_report(ledger, agent_context)
        except Exception as err:
            event_bus.emit(
                EventType.WARNING,
                "reviewer",
                f"report generation failed ({err}), writing fallback report",
            )
            report_text, headline = build_fallback_report(ledger, agent_context)

        report_file = agent_context.workspace.report_folder / "report.md"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report_text, encoding="utf-8")

        final = ledger.final.model_copy(
            update={"report_path": str(report_file), "headline": headline}
        )
        event_bus.emit(
            EventType.INFO,
            "reviewer",
            f"Final report ready: {report_file}\n{headline}",
        )
        return {"final": final, "status": "done", "phase": "done"}

    node_actions = {
        "data_engineer": lambda ledger: {"brief": run_data_engineer(ledger, agent_context), "phase": "data_analyst"},
        "data_analyst": lambda ledger: {"eda": run_data_analyst(ledger, agent_context)},
        "ml_lead": run_ml_lead_node,
        "stop_check": run_stop_check_node,
        "feature_engineer": lambda ledger: run_feature_engineer(ledger, agent_context),
        "model": lambda ledger: {"runs": [*ledger.runs, run_modeler(ledger, agent_context)]},
        "tune": lambda ledger: {"runs": [*ledger.runs, *run_tuner(ledger, agent_context)], "last_tuned_iteration": ledger.iteration},
        "evaluate": run_evaluate_node,
        "review": run_review_node,
        "submit": lambda ledger: {"final": run_submitter(ledger, agent_context), "phase": "report"},
        "report": run_report_node,
    }
    return {
        name: make_node(name, fn, agent_context) for name, fn in node_actions.items()
    }


def targets_with_exits(*node_names: str) -> dict[str, str]:
    """A router's possible targets. Every router can also end the run or go to submit/report."""
    return {name: name for name in [*node_names, "submit", "report", END]}


def build_pipeline(agent_context: AgentContext):
    max_eda_loops = agent_context.config.limits.max_eda_loops

    graph = StateGraph(Ledger)
    for node_name, node_function in build_nodes(agent_context).items():
        graph.add_node(node_name, node_function)

    graph.add_conditional_edges(
        START,
        route_from_phase,
        {
            "data_engineer": "data_engineer",
            "data_analyst": "data_analyst",
            "ml_lead": "ml_lead",
            "stop_check": "stop_check",
            "submit": "submit",
            "report": "report",
            END: END,
        },
    )

    # Fixed next nodes
    fixed_next_node = {
        "data_engineer": "data_analyst",
        "ml_lead": "stop_check",
        "evaluate": "review",
        "review": "stop_check",
        "submit": "report",
    }
    for source_name, next_name in fixed_next_node.items():
        router = stop_when_over(lambda ledger, next_name=next_name: next_name)
        graph.add_conditional_edges(source_name, router, targets_with_exits(next_name))

    # Conditional routes
    graph.add_conditional_edges(
        "data_analyst",
        stop_when_over(lambda ledger: route_after_data_analyst(ledger, max_eda_loops)),
        targets_with_exits("data_analyst", "ml_lead", "feature_engineer"),
    )

    chosen_next_node = {
        "stop_check": (
            route_after_stop_check,
            ["feature_engineer", "model", "tune", "data_analyst", "evaluate", "review"],
        ),
        "feature_engineer": (route_after_features, ["model", "review"]),
        "model": (route_after_model, ["evaluate", "review"]),
        "tune": (route_after_tune, ["evaluate", "review"]),
    }
    for source_name, (router, target_names) in chosen_next_node.items():
        graph.add_conditional_edges(
            source_name, stop_when_over(router), targets_with_exits(*target_names)
        )

    graph.add_edge("report", END)
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
