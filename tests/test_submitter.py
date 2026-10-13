import pandas as pd
from scripted_answers import coder_answer_printing, two_failed_coder_attempts

from mlagent.agents.submitter import (
    build_report_facts,
    build_runs_table,
    pick_ensemble_runs,
    run_submitter,
)
from mlagent.events import EventType
from mlagent.llm_client import LlmUnavailableError
from mlagent.memory import create_new_ledger
from mlagent.state import Evaluation, ExperimentRun, FeatureRecord, FileInfo, FinalResult

REPORT_TEXT = "## Summary\nThe team trained a model."


def submit_ledger(workspace):
    pd.DataFrame({"id": [1, 2, 3], "churn": [0, 0, 0]}).to_csv(
        workspace.project_folder / "sample_submission.csv", index=False
    )
    ledger = create_new_ledger("Predict churn.", "macro F1")
    ledger.brief.summary = "Customer table."
    ledger.brief.deliverable_format = "predictions.csv with id and churn"
    ledger.brief.files = [FileInfo(path="sample_submission.csv", role="sample_submission")]
    ledger.plan.primary_metric = "macro_f1"
    ledger.plan.validation_scheme = "stratified_kfold"
    ledger.plan.splits_path = str(workspace.splits_path)
    ledger.iteration = 5
    ledger.stop_reason = "budget"
    return ledger


def add_run(ledger, workspace, scores, data=None, approved=True, predictions_as_folder=False):
    experiment_id = ledger.next_experiment_id()
    folder = workspace.experiment_folder(experiment_id)
    test_predictions_path = folder / "test_predictions.csv"
    if predictions_as_folder:
        test_predictions_path = folder / "test_predictions"
        test_predictions_path.mkdir()
        (test_predictions_path / "mask_1.png").write_bytes(b"png")
    else:
        pd.DataFrame(data or {"id": [1, 2, 3], "churn": [1, 0, 1]}).to_csv(
            test_predictions_path, index=False
        )
    validation_path = folder / "validation_predictions.csv"
    validation_path.write_text("fold,probability\n0,0.5\n", encoding="utf-8")
    ledger.runs.append(
        ExperimentRun(
            experiment_id=experiment_id,
            focus="model",
            model_family="lightgbm",
            validation_scores=scores,
            score_mean=sum(scores) / len(scores),
            score_std=0.01,
            validation_predictions_path=str(validation_path),
            test_predictions_path=str(test_predictions_path),
        )
    )
    if approved:
        ledger.evaluations.append(Evaluation(experiment_id=experiment_id, verdict="approve"))
    return experiment_id


def blend_answer(workspace, scores, write_file=True, method="average of probabilities") -> str:
    blend_path = workspace.artifacts_folder / "submitter" / "blend_predictions.csv"
    script = f"""
import json
from pathlib import Path

path = Path({str(blend_path)!r})
path.parent.mkdir(parents=True, exist_ok=True)
if {write_file!r}:
    path.write_text("id,churn\\n1,1\\n2,1\\n3,0\\n")
result = {{
    "validation_scores": {scores!r},
    "test_predictions_path": str(path),
    "method": {method!r},
}}
print("<BLEND>" + json.dumps(result) + "</BLEND>")
"""
    return f"```python\n{script}\n```"


def test_without_an_approved_run_only_a_report_is_written(make_agent_context, workspace):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8], approved=False)
    agent_context = make_agent_context([REPORT_TEXT])

    final_result = run_submitter(ledger, agent_context)

    assert final_result.best_experiment_id == ""
    assert final_result.predictions_path == ""
    assert final_result.check_messages == ["No approved run, so no predictions file was written."]
    report_text = workspace.report_path.read_text(encoding="utf-8")
    assert "The team trained a model." in report_text
    assert "## All runs" in report_text
    assert "There is no approved run" in agent_context.llm_client.received_prompts[0]


def test_a_single_approved_run_is_copied_and_checked(make_agent_context, workspace):
    ledger = submit_ledger(workspace)
    best_id = add_run(ledger, workspace, [0.80, 0.82, 0.81])
    agent_context = make_agent_context([REPORT_TEXT])

    final_result = run_submitter(ledger, agent_context)

    assert final_result.best_experiment_id == best_id
    assert final_result.predictions_path == str(workspace.report_folder / "predictions.csv")
    assert pd.read_csv(final_result.predictions_path)["churn"].tolist() == [1, 0, 1]
    assert final_result.check_messages == ["All delivery checks passed."]
    assert final_result.ensemble_description == ""
    assert final_result.report_path == str(workspace.report_path)
    assert len(agent_context.llm_client.received_prompts) == 1


def test_a_clearly_better_blend_replaces_the_best_single_run(make_agent_context, workspace):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.80, 0.80, 0.80])
    add_run(ledger, workspace, [0.78, 0.79, 0.78], data={"id": [1, 2, 3], "churn": [0, 1, 1]})
    answers = [blend_answer(workspace, [0.85, 0.86, 0.85]), REPORT_TEXT]
    agent_context = make_agent_context(answers)

    final_result = run_submitter(ledger, agent_context)

    assert pd.read_csv(final_result.predictions_path)["churn"].tolist() == [1, 1, 0]
    assert "blend of experiment_001, experiment_002: average of probabilities" in (
        final_result.ensemble_description
    )
    blend_prompt = agent_context.llm_client.received_prompts[0]
    assert "validation_predictions.csv" in blend_prompt
    assert "Use a simple blend" in blend_prompt
    assert "<BLEND>JSON</BLEND>" in blend_prompt
    assert "Ensemble: blend of experiment_001" in agent_context.llm_client.received_prompts[1]


def test_a_blend_without_a_counted_gain_is_not_used(make_agent_context, workspace):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.80, 0.80, 0.80])
    add_run(ledger, workspace, [0.78, 0.79, 0.78])
    agent_context = make_agent_context([blend_answer(workspace, [0.80, 0.80, 0.80]), REPORT_TEXT])

    final_result = run_submitter(ledger, agent_context)

    assert final_result.ensemble_description == ""
    assert pd.read_csv(final_result.predictions_path)["churn"].tolist() == [1, 0, 1]


def test_predictions_that_cannot_be_blended_leave_the_best_run_in_place(
    make_agent_context, workspace
):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.80, 0.80, 0.80])
    add_run(ledger, workspace, [0.78, 0.79, 0.78])
    agent_context = make_agent_context([blend_answer(workspace, None), REPORT_TEXT])

    final_result = run_submitter(ledger, agent_context)

    assert final_result.ensemble_description == ""


def test_a_blend_file_that_does_not_exist_is_not_used(make_agent_context, workspace):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.80, 0.80, 0.80])
    add_run(ledger, workspace, [0.78, 0.79, 0.78])
    answers = [blend_answer(workspace, [0.9, 0.9, 0.9], write_file=False), REPORT_TEXT]
    agent_context = make_agent_context(answers)

    assert run_submitter(ledger, agent_context).ensemble_description == ""


def test_a_failed_blend_script_leaves_the_best_run_in_place(make_agent_context, workspace):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.80, 0.80, 0.80])
    add_run(ledger, workspace, [0.78, 0.79, 0.78])
    agent_context = make_agent_context([*two_failed_coder_attempts("shape mismatch"), REPORT_TEXT])

    final_result = run_submitter(ledger, agent_context)

    assert final_result.ensemble_description == ""
    assert final_result.check_messages == ["All delivery checks passed."]


def test_a_blend_result_that_breaks_the_contract_is_ignored(make_agent_context, workspace):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.80, 0.80, 0.80])
    add_run(ledger, workspace, [0.78, 0.79, 0.78])
    broken_answer = coder_answer_printing("BLEND", {"validation_scores": "high"})
    agent_context = make_agent_context([broken_answer, REPORT_TEXT])

    assert run_submitter(ledger, agent_context).ensemble_description == ""


def test_only_runs_with_validation_predictions_can_be_blended(workspace):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8])
    add_run(ledger, workspace, [0.7, 0.7, 0.7])
    ledger.runs[1].validation_predictions_path = ""
    add_run(ledger, workspace, [0.6, 0.6, 0.6])
    add_run(ledger, workspace, [0.9, 0.9, 0.9], approved=False)

    assert [run.experiment_id for run in pick_ensemble_runs(ledger)] == [
        "experiment_001",
        "experiment_003",
    ]


def test_at_most_three_runs_are_blended(workspace):
    ledger = submit_ledger(workspace)
    for score in (0.5, 0.6, 0.7, 0.8, 0.9):
        add_run(ledger, workspace, [score, score, score])

    assert [run.experiment_id for run in pick_ensemble_runs(ledger)] == [
        "experiment_005",
        "experiment_004",
        "experiment_003",
    ]


def test_a_predictions_file_that_fails_the_checks_is_reported(make_agent_context, workspace):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8], data={"id": [1, 2], "label": [0, 1]})
    agent_context = make_agent_context([REPORT_TEXT])
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    final_result = run_submitter(ledger, agent_context)

    assert any("columns differ" in message for message in final_result.check_messages)
    assert any(
        e.event_type == EventType.WARNING and "columns differ" in e.message for e in received_events
    )
    assert "columns differ" in agent_context.llm_client.received_prompts[0]
    assert workspace.report_path.exists()


def test_a_predictions_folder_is_copied_as_a_folder(make_agent_context, workspace):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8], predictions_as_folder=True)
    agent_context = make_agent_context([REPORT_TEXT])

    final_result = run_submitter(ledger, agent_context)

    assert final_result.predictions_path == str(workspace.report_folder / "predictions")
    assert (workspace.report_folder / "predictions" / "mask_1.png").exists()
    assert final_result.check_messages == ["All delivery checks passed."]


def test_the_report_still_lists_the_facts_when_the_llm_cannot_be_reached(
    make_agent_context, workspace
):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8])

    def llm_is_down(prompt):
        raise LlmUnavailableError("Ollama stopped")

    agent_context = make_agent_context([llm_is_down])

    final_result = run_submitter(ledger, agent_context)

    report_text = workspace.report_path.read_text(encoding="utf-8")
    assert "could not be reached, so this report lists the facts only" in report_text
    assert "Best approved run: experiment_001" in report_text
    assert final_result.predictions_path != ""


def test_the_report_prompt_carries_the_facts_and_the_no_invention_rule(
    make_agent_context, workspace
):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.80, 0.82, 0.81])
    agent_context = make_agent_context([REPORT_TEXT])

    run_submitter(ledger, agent_context)

    report_prompt = agent_context.llm_client.received_prompts[0]
    assert "Task (the user's words): Predict churn." in report_prompt
    assert "Best approved run: experiment_001, lightgbm" in report_prompt
    assert "Stop reason: budget" in report_prompt
    assert "Never invent a number" in report_prompt
    assert "Time limit for one script run: 30 minutes" in report_prompt


def test_report_facts_flag_unchecked_runs_kept_features_and_an_early_end():
    ledger = create_new_ledger("Predict churn.", "")
    ledger.runs = [
        ExperimentRun(experiment_id="experiment_001", score_mean=0.8, validation_scores=[0.8])
    ]
    ledger.evaluations = [
        Evaluation(experiment_id="experiment_001", verdict="approve", checks={"unverified": True})
    ]
    ledger.features = [
        FeatureRecord(feature_id="feature_001", name="basic_preparation", kept=True),
        FeatureRecord(feature_id="feature_002", name="noise_idea", kept=False),
    ]
    ledger.error_message = "Ollama stopped answering"
    ledger.stop_reason = "llm_unavailable"

    facts = build_report_facts(ledger, FinalResult(check_messages=["All delivery checks passed."]))

    assert "Approved WITHOUT independent checks: experiment_001." in facts
    assert "Input preparation steps kept: basic_preparation." in facts
    assert "The run ended early because: Ollama stopped answering" in facts
    assert "Metrics (the user's words): not given" in facts
    assert "Stop reason: llm_unavailable" in facts


def test_the_runs_table_shows_each_status(workspace):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8])
    add_run(ledger, workspace, [0.9, 0.9, 0.9], approved=False)
    ledger.runs.append(ExperimentRun(experiment_id="experiment_003", error="boom"))

    table = build_runs_table(ledger)

    lines = table.splitlines()
    assert lines[0].startswith("| Run | Round | Model")
    assert "| experiment_001 | model | lightgbm | 0.8000 +/- 0.0100 | n/a | approved |" in lines
    assert any("experiment_002" in line and "rejected or not checked" in line for line in lines)
    assert any("experiment_003" in line and "crashed" in line for line in lines)


def test_events_name_the_report_file(make_agent_context, workspace):
    ledger = submit_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8])
    agent_context = make_agent_context([REPORT_TEXT])
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    run_submitter(ledger, agent_context)

    submitter_events = [e for e in received_events if e.agent_name == "Submitter"]
    assert submitter_events[0].message == "packaging the result"
    assert submitter_events[-1].event_type == EventType.AGENT_FINISHED
    assert str(workspace.report_path) in submitter_events[-1].message
