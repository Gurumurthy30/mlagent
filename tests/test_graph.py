import json

import pandas as pd
import pytest
from full_run_answers import (
    REPORT_TEXT,
    answers_before_the_modeler,
    answers_from_the_modeler,
    churn_brief,
)
from langgraph.graph import END
from scripted_answers import coder_answer_printing, two_failed_coder_attempts
from test_feature_engineer import NOISY_SCORES, candidate, screen_answer
from test_modeling_pipeline_on_sample_project import write_sample_project
from test_tuner import tuned_answer

from mlagent.agents.base import AgentFailedError
from mlagent.agents.data_analyst import EdaSummary, NewFinding, RoundReading
from mlagent.agents.evaluator import VerdictAnswer
from mlagent.agents.feature_engineer import CandidateList
from mlagent.agents.modeler import ModelChoice
from mlagent.agents.reviewer import ReviewDraft
from mlagent.config import AppConfig
from mlagent.events import EventType
from mlagent.graph import (
    DATA_ANALYST,
    DATA_ANALYST_REVISIT,
    DATA_ENGINEER,
    EVALUATOR,
    FEATURE_ENGINEER,
    ML_LEAD,
    MODELER,
    REVIEWER,
    STOP_CHECK,
    SUBMITTER,
    TUNER,
    build_failure_updates,
    make_node,
    route_after_features,
    route_after_modeler,
    route_after_stop_check,
    route_after_tuner,
    route_from_phase,
    run_pipeline,
    stop_when_over,
)
from mlagent.llm_client import LlmUnavailableError
from mlagent.memory import create_new_ledger, load_ledger, save_ledger
from mlagent.state import Evaluation, ExperimentRun, Review

CHECKS_OK = {
    "recomputed_score": 0.84,
    "recomputed_holdout_score": None,
    "leakage_flags": [],
    "train_validation_overlap": 0,
    "notes": [],
}


def loop_config(max_experiment_loops: int) -> AppConfig:
    return AppConfig.model_validate(
        {"limits": {"max_experiment_loops": max_experiment_loops, "max_eda_loops": 1}}
    )


def loop_ledger(workspace, focus="model", iteration=0):
    ledger = create_new_ledger("Predict churn.", "macro F1")
    ledger.brief = churn_brief()
    ledger.plan.primary_metric = "macro_f1"
    ledger.plan.extra_metrics = ["auc"]
    ledger.plan.baseline_description = "logistic regression"
    ledger.plan.validation_scheme = "stratified_kfold"
    ledger.plan.splits_path = str(workspace.splits_path)
    ledger.current_feature_version = 1
    ledger.phase = "experiment_loop"
    ledger.focus = focus
    ledger.iteration = iteration
    workspace.features_folder.mkdir(parents=True, exist_ok=True)
    workspace.features_file(1).write_text(
        "def prepare(table):\n    return table\n", encoding="utf-8"
    )
    return ledger


def add_approved_run(ledger, workspace, family, score, iteration=0):
    experiment_id = ledger.next_experiment_id()
    folder = workspace.experiment_folder(experiment_id)
    test_predictions_path = folder / "test_predictions.csv"
    pd.DataFrame({"customer_id": [1, 2], "churn": [0, 1]}).to_csv(
        test_predictions_path, index=False
    )
    validation_path = folder / "validation_predictions.csv"
    validation_path.write_text("fold,probability\n0,0.5\n", encoding="utf-8")
    ledger.runs.append(
        ExperimentRun(
            experiment_id=experiment_id,
            focus="model",
            model_family=family,
            model_params={"depth": 6},
            iteration=iteration,
            feature_version=1,
            validation_scores=[score, score, score],
            score_mean=score,
            score_std=0.0,
            validation_predictions_path=str(validation_path),
            test_predictions_path=str(test_predictions_path),
            script_path=str(folder / "train_model.py"),
        )
    )
    ledger.evaluations.append(Evaluation(experiment_id=experiment_id, verdict="approve"))


def raise_llm_down(prompt):
    raise LlmUnavailableError("Ollama stopped")


def checks_answer() -> str:
    return coder_answer_printing("CHECKS", CHECKS_OK)


def approve() -> VerdictAnswer:
    return VerdictAnswer(verdict="approve", reasons=["fine"])


@pytest.mark.parametrize(
    ("phase", "expected_node"),
    [
        ("data_engineer", DATA_ENGINEER),
        ("data_analyst", DATA_ANALYST),
        ("ml_lead", ML_LEAD),
        ("experiment_loop", STOP_CHECK),
        ("submitter", SUBMITTER),
        ("done", END),
    ],
)
def test_a_run_starts_or_resumes_at_the_step_its_phase_names(phase, expected_node):
    ledger = create_new_ledger("task", "metric")
    ledger.phase = phase

    assert route_from_phase(ledger) == expected_node


def test_a_failed_or_finished_run_ends_and_a_stopped_loop_goes_to_the_submitter():
    route = stop_when_over(lambda ledger: MODELER)
    running = create_new_ledger("task", "metric")
    failed = running.model_copy(update={"status": "failed"})
    finished = running.model_copy(update={"phase": "done"})
    stopped = running.model_copy(update={"phase": "submitter"})

    assert route(running) == MODELER
    assert route(failed) == END
    assert route(finished) == END
    assert route(stopped) == SUBMITTER


@pytest.mark.parametrize(
    ("focus", "expected_node"),
    [
        ("start", FEATURE_ENGINEER),
        ("features", FEATURE_ENGINEER),
        ("model", MODELER),
        ("tune", TUNER),
        ("data", DATA_ANALYST_REVISIT),
    ],
)
def test_the_focus_decides_the_next_agent(focus, expected_node):
    ledger = create_new_ledger("task", "metric")
    ledger.focus = focus

    assert route_after_stop_check(ledger) == expected_node


def test_a_round_interrupted_after_the_first_feature_step_continues_with_the_baseline_model():
    ledger = create_new_ledger("task", "metric")
    ledger.focus = "start"
    ledger.current_feature_version = 1

    assert route_after_stop_check(ledger) == MODELER


def test_a_round_interrupted_after_a_run_continues_with_its_check_or_its_review():
    ledger = create_new_ledger("task", "metric")
    ledger.focus = "model"
    ledger.iteration = 2
    ledger.runs = [ExperimentRun(experiment_id="experiment_001", iteration=2, score_mean=0.8)]
    assert route_after_stop_check(ledger) == EVALUATOR

    ledger.evaluations = [Evaluation(experiment_id="experiment_001", verdict="approve")]
    assert route_after_stop_check(ledger) == REVIEWER

    ledger.evaluations = []
    ledger.runs = [ExperimentRun(experiment_id="experiment_001", iteration=2, error="boom")]
    assert route_after_stop_check(ledger) == REVIEWER


def test_runs_of_earlier_rounds_do_not_change_the_route_of_a_new_round():
    ledger = create_new_ledger("task", "metric")
    ledger.focus = "tune"
    ledger.iteration = 3
    ledger.runs = [ExperimentRun(experiment_id="experiment_001", iteration=2, score_mean=0.8)]

    assert route_after_stop_check(ledger) == TUNER


def test_a_features_round_that_kept_nothing_skips_the_modeler():
    ledger = create_new_ledger("task", "metric")

    ledger.last_features_changed = True
    assert route_after_features(ledger) == MODELER
    ledger.last_features_changed = False
    assert route_after_features(ledger) == REVIEWER


def test_a_crashed_model_goes_to_the_reviewer_and_a_good_one_to_the_evaluator():
    ledger = create_new_ledger("task", "metric")
    assert route_after_modeler(ledger) == REVIEWER

    ledger.runs = [ExperimentRun(experiment_id="experiment_001", error="boom")]
    assert route_after_modeler(ledger) == REVIEWER

    ledger.runs = [ExperimentRun(experiment_id="experiment_001", score_mean=0.8)]
    assert route_after_modeler(ledger) == EVALUATOR


def test_the_tuner_goes_to_the_evaluator_when_any_run_of_this_round_worked():
    ledger = create_new_ledger("task", "metric")
    ledger.iteration = 3
    ledger.runs = [
        ExperimentRun(experiment_id="experiment_001", iteration=1, score_mean=0.9),
        ExperimentRun(experiment_id="experiment_002", iteration=3, error="boom"),
    ]
    assert route_after_tuner(ledger) == REVIEWER

    ledger.runs.append(ExperimentRun(experiment_id="experiment_003", iteration=3, score_mean=0.8))
    assert route_after_tuner(ledger) == EVALUATOR


def test_a_failure_without_approved_work_marks_the_run_failed_and_keeps_the_phase():
    ledger = create_new_ledger("task", "metric")

    updates = build_failure_updates(ledger, MODELER, "agent_failed", "no data")

    assert updates == {
        "stop_reason": "agent_failed",
        "error_message": "no data",
        "status": "failed",
    }


def test_a_failure_with_approved_work_hands_it_to_the_submitter(workspace):
    ledger = loop_ledger(workspace)
    add_approved_run(ledger, workspace, "lightgbm", 0.8)

    updates = build_failure_updates(ledger, MODELER, "llm_unavailable", "Ollama stopped")

    assert updates["phase"] == "submitter"
    assert "status" not in updates


def test_a_failure_in_the_submitter_itself_never_loops_back_to_it(workspace):
    ledger = loop_ledger(workspace)
    add_approved_run(ledger, workspace, "lightgbm", 0.8)

    updates = build_failure_updates(ledger, SUBMITTER, "agent_failed", "cannot write")

    assert updates["status"] == "failed"
    assert "phase" not in updates


def test_a_node_saves_the_ledger_and_the_progress_file_after_it_runs(make_agent_context, workspace):
    agent_context = make_agent_context()
    ledger = create_new_ledger("Predict churn.", "macro F1")
    run_node = make_node("some_node", lambda current: {"iteration": 4}, agent_context)

    updates = run_node(ledger)

    assert updates == {"iteration": 4}
    assert load_ledger(workspace).iteration == 4
    assert "experiment rounds finished: 4" in workspace.progress_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("error", "expected_reason"),
    [
        (LlmUnavailableError("Ollama stopped"), "llm_unavailable"),
        (AgentFailedError("no usable splits"), "agent_failed"),
    ],
)
def test_agent_failures_become_recorded_updates_and_error_events(
    make_agent_context, workspace, error, expected_reason
):
    agent_context = make_agent_context()
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    def failing_node(ledger):
        raise error

    updates = make_node("ml_lead", failing_node, agent_context)(create_new_ledger("t", "m"))

    assert updates["stop_reason"] == expected_reason
    assert updates["status"] == "failed"
    assert load_ledger(workspace).error_message == str(error)
    assert any(e.event_type == EventType.ERROR and e.message == str(error) for e in received_events)


def test_a_complete_small_run_from_the_first_look_to_the_report(make_agent_context, workspace):
    write_sample_project(workspace.project_folder)
    ledger = create_new_ledger("Predict which customers churn.", "macro F1 mainly, also AUC")
    answers = [*answers_before_the_modeler(workspace), *answers_from_the_modeler(workspace)]
    agent_context = make_agent_context(answers, timeout_seconds=180, config=loop_config(2))
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    final_ledger = run_pipeline(ledger, agent_context)

    assert agent_context.llm_client.scripted_answers == []
    assert final_ledger.status == "done"
    assert final_ledger.phase == "done"
    assert final_ledger.stop_reason == "budget"
    assert final_ledger.iteration == 2
    assert final_ledger.brief.primary_metric == "macro_f1"
    assert final_ledger.eda.findings[0].finding_id == "finding_001"
    assert final_ledger.plan.splits_path == str(workspace.splits_path)
    assert final_ledger.current_feature_version == 1
    assert [run.experiment_id for run in final_ledger.runs] == ["experiment_001", "experiment_002"]
    assert [run.iteration for run in final_ledger.runs] == [0, 1]
    assert len(final_ledger.approved_runs()) == 2
    assert [review.counted_gain for review in final_ledger.reviews] == [True, False]
    assert [review.reviewed_focus for review in final_ledger.reviews] == ["start", "model"]

    final_result = final_ledger.final
    assert final_result.best_experiment_id in ("experiment_001", "experiment_002")
    assert final_result.ensemble_description == ""
    assert final_result.check_messages == ["All delivery checks passed."]
    assert pd.read_csv(final_result.predictions_path).shape == (60, 2)
    report_text = workspace.report_path.read_text(encoding="utf-8")
    assert REPORT_TEXT in report_text
    assert "| experiment_002 | model | logistic_regression |" in report_text

    assert load_ledger(workspace) == final_ledger
    assert "stop reason: budget" in workspace.progress_path.read_text(encoding="utf-8")
    messages = [event.message for event in received_events]
    assert any(message.startswith("new best experiment_001") for message in messages)
    assert "stopping the loop: budget" in messages
    llm_log_lines = workspace.llm_calls_log_path.read_text(encoding="utf-8").splitlines()
    assert len(llm_log_lines) == len(answers)


def test_a_crashed_model_is_reviewed_and_the_run_still_ends_cleanly(make_agent_context, workspace):
    ledger = loop_ledger(workspace, focus="model")
    answers = [
        ModelChoice(model_family="lightgbm", model_params={"depth": 6}),
        *two_failed_coder_attempts("out of memory"),
        ReviewDraft(assessment="the script crashed", focus="model", reason="retry smaller"),
        REPORT_TEXT,
    ]
    agent_context = make_agent_context(answers, config=loop_config(1))

    final_ledger = run_pipeline(ledger, agent_context)

    assert final_ledger.runs[0].crashed
    assert final_ledger.evaluations == []
    assert final_ledger.reviews[0].reviewed_focus == "model"
    assert final_ledger.reviews[0].counted_gain is False
    assert final_ledger.stop_reason == "budget"
    assert final_ledger.status == "done"
    assert final_ledger.final.check_messages == [
        "No approved run, so no predictions file was written."
    ]
    assert agent_context.llm_client.scripted_answers == []


def test_a_features_round_that_keeps_nothing_goes_straight_to_the_reviewer(
    make_agent_context, workspace
):
    ledger = loop_ledger(workspace, focus="features")
    answers = [
        CandidateList(candidates=[candidate("noisy_ratio")]),
        screen_answer({"noisy_ratio": NOISY_SCORES}),
        ReviewDraft(assessment="nothing helped", focus="model", reason="change the model"),
        REPORT_TEXT,
    ]
    agent_context = make_agent_context(answers, config=loop_config(1))

    final_ledger = run_pipeline(ledger, agent_context)

    assert final_ledger.runs == []
    assert final_ledger.last_features_changed is False
    assert final_ledger.current_feature_version == 1
    assert [feature.name for feature in final_ledger.features] == ["noisy_ratio"]
    assert final_ledger.reviews[0].reviewed_focus == "features"
    assert final_ledger.reviews[0].counted_gain is False
    assert final_ledger.status == "done"


def test_a_tune_round_checks_every_tuned_run_before_the_review(make_agent_context, workspace):
    ledger = loop_ledger(workspace, focus="tune", iteration=3)
    add_approved_run(ledger, workspace, "lightgbm", 0.83, iteration=0)
    add_approved_run(ledger, workspace, "xgboost", 0.82, iteration=1)
    not_blendable = coder_answer_printing(
        "BLEND", {"validation_scores": None, "test_predictions_path": "", "method": "no"}
    )
    answers = [
        tuned_answer({"depth": 8}),
        tuned_answer({"depth": 5}),
        checks_answer(),
        approve(),
        checks_answer(),
        approve(),
        coder_answer_printing("DIAG", {"weak_spots": [], "comparison": "", "notes": []}),
        ReviewDraft(assessment="tuning helped", focus="model", reason="try another family"),
        not_blendable,
        REPORT_TEXT,
    ]
    agent_context = make_agent_context(answers, config=loop_config(4))

    final_ledger = run_pipeline(ledger, agent_context)

    tuned_runs = [run for run in final_ledger.runs if run.focus == "tune"]
    assert [run.experiment_id for run in tuned_runs] == ["experiment_003", "experiment_004"]
    assert [e.verdict for e in final_ledger.evaluations[2:]] == ["approve", "approve"]
    assert final_ledger.last_tuned_iteration == 3
    assert final_ledger.reviews[0].reviewed_focus == "tune"
    assert final_ledger.reviews[0].counted_gain is True
    assert final_ledger.stop_reason == "budget"
    assert agent_context.llm_client.scripted_answers == []


def test_a_data_focus_reopens_the_analysis_then_continues_with_features(
    make_agent_context, workspace
):
    write_sample_project(workspace.project_folder)
    ledger = loop_ledger(workspace, focus="data")
    ledger.reviews = [
        Review(
            iteration=1,
            focus="data",
            reviewed_focus="start",
            question="Do the same customers appear twice?",
        )
    ]
    ledger.iteration = 1
    ledger.eda.checklist = {"target distribution": "covered"}
    eda_output = {
        "answers": [{"question": "q", "result": "no duplicate customers"}],
        "problems": [],
    }
    answers = [
        coder_answer_printing("FINDINGS", eda_output),
        RoundReading(findings=[NewFinding(statement="Customers are unique.")], coverage=[]),
        EdaSummary(
            issues=[], feature_ideas=[], leakage_risks=[], validation_hints=[], model_hints=[]
        ),
        CandidateList(candidates=[]),
        ReviewDraft(assessment="no new inputs", focus="model", reason="change the model"),
        REPORT_TEXT,
    ]
    agent_context = make_agent_context(answers, config=loop_config(2))

    final_ledger = run_pipeline(ledger, agent_context)

    assert "Do the same customers appear twice?" in agent_context.llm_client.received_prompts[0]
    assert final_ledger.data_reentries_used == 1
    assert [finding.statement for finding in final_ledger.eda.findings] == ["Customers are unique."]
    assert final_ledger.reviews[-1].reviewed_focus == "features"
    assert final_ledger.status == "done"


def test_a_stopped_llm_hands_the_approved_work_to_a_fallback_report(make_agent_context, workspace):
    ledger = loop_ledger(workspace, focus="model", iteration=1)
    add_approved_run(ledger, workspace, "lightgbm", 0.83)
    agent_context = make_agent_context([raise_llm_down, raise_llm_down], config=loop_config(5))

    final_ledger = run_pipeline(ledger, agent_context)

    assert final_ledger.status == "done"
    assert final_ledger.stop_reason == "llm_unavailable"
    assert "Ollama stopped" in final_ledger.error_message
    assert pd.read_csv(final_ledger.final.predictions_path).shape == (2, 2)
    report_text = workspace.report_path.read_text(encoding="utf-8")
    assert "could not be reached, so this report lists the facts only" in report_text
    assert "The run ended early because: Ollama stopped" in report_text


def test_a_failure_before_any_result_ends_the_run_as_failed_and_can_resume(
    make_agent_context, workspace
):
    ledger = create_new_ledger("Predict churn.", "macro F1")
    answers = [
        answer for _ in range(3) for answer in two_failed_coder_attempts("folder unreadable")
    ]
    agent_context = make_agent_context(answers)

    final_ledger = run_pipeline(ledger, agent_context)

    assert final_ledger.status == "failed"
    assert final_ledger.stop_reason == "agent_failed"
    assert "could not find out: files, modality, task_family" in final_ledger.error_message
    assert final_ledger.phase == "data_engineer"
    assert final_ledger.final.report_path == ""


def test_a_run_saved_in_the_submitter_phase_resumes_straight_into_it(make_agent_context, workspace):
    ledger = loop_ledger(workspace, focus="model", iteration=3)
    add_approved_run(ledger, workspace, "lightgbm", 0.8)
    ledger.phase = "submitter"
    ledger.stop_reason = "plateau"
    agent_context = make_agent_context([REPORT_TEXT])

    final_ledger = run_pipeline(ledger, agent_context)

    assert final_ledger.status == "done"
    assert final_ledger.stop_reason == "plateau"
    assert len(agent_context.llm_client.received_prompts) == 1


def test_an_unexpected_error_is_logged_with_its_traceback_and_the_ledger_is_marked_failed(
    make_agent_context, workspace
):
    ledger = create_new_ledger("Predict churn.", "macro F1")
    save_ledger(workspace, ledger)

    def broken_answer(prompt):
        raise ValueError("a bug in an agent")

    agent_context = make_agent_context([broken_answer])

    with pytest.raises(ValueError, match="a bug in an agent"):
        run_pipeline(ledger, agent_context)

    saved_ledger = load_ledger(workspace)
    traceback_text = (workspace.logs_folder / "error_traceback.txt").read_text(encoding="utf-8")
    assert saved_ledger.status == "failed"
    assert "Unexpected error" in saved_ledger.error_message
    assert "a bug in an agent" in traceback_text
    assert json.loads(json.dumps(saved_ledger.model_dump()))["status"] == "failed"
