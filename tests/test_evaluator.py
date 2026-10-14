import json

import pytest
from scripted_answers import coder_answer_printing, two_failed_coder_attempts

from mlagent.agents.base import AgentFailedError
from mlagent.agents.evaluator import (
    EvaluationChecks,
    LeakageFlag,
    VerdictAnswer,
    hard_check_reasons,
    run_evaluator,
)
from mlagent.events import EventType
from mlagent.memory import create_new_ledger
from mlagent.state import ExperimentRun


def make_run(**changes) -> ExperimentRun:
    values = {
        "experiment_id": "experiment_001",
        "focus": "start",
        "model_family": "lightgbm",
        "model_params": {"depth": 6},
        "feature_version": 1,
        "script_path": "/work/experiment_001/train_model.py",
        "validation_scores": [0.80, 0.82, 0.81],
        "score_mean": 0.81,
        "score_std": 0.01,
        "validation_predictions_path": "/work/experiment_001/validation_predictions.csv",
        "test_predictions_path": "/work/experiment_001/test_predictions.csv",
    }
    values.update(changes)
    return ExperimentRun(**values)


def evaluator_ledger(run=None):
    ledger = create_new_ledger("Predict churn.", "macro F1")
    ledger.brief.summary = "Customer table."
    ledger.plan.primary_metric = "macro_f1"
    ledger.plan.splits_path = "/work/splits.json"
    ledger.runs = [run or make_run()]
    return ledger


def checks_answer(**changes) -> str:
    checks = {
        "recomputed_score": 0.81,
        "recomputed_holdout_score": None,
        "leakage_flags": [],
        "train_validation_overlap": 0,
        "notes": ["looked fine"],
    }
    checks.update(changes)
    return coder_answer_printing("CHECKS", checks)


def approve_answer() -> VerdictAnswer:
    return VerdictAnswer(verdict="approve", reasons=["score matches, no leakage found"])


def test_a_matching_score_without_flags_passes_the_hard_rules():
    checks = EvaluationChecks(recomputed_score=0.815)

    assert hard_check_reasons(make_run(), checks, higher_is_better=True) == []


def test_a_recomputed_score_far_from_the_logged_one_is_rejected():
    reasons = hard_check_reasons(make_run(), EvaluationChecks(recomputed_score=0.85), True)

    assert len(reasons) == 1
    assert "recomputed score 0.8500 differs from the logged score 0.8100" in reasons[0]


def test_the_score_tolerance_is_the_std_or_two_percent_whichever_is_larger():
    just_inside = EvaluationChecks(recomputed_score=0.81 + 0.0160)
    just_outside = EvaluationChecks(recomputed_score=0.81 + 0.0165)
    wide_spread_run = make_run(score_std=0.05)

    assert hard_check_reasons(make_run(), just_inside, True) == []
    assert len(hard_check_reasons(make_run(), just_outside, True)) == 1
    assert hard_check_reasons(wide_spread_run, EvaluationChecks(recomputed_score=0.85), True) == []


def test_a_holdout_much_worse_than_validation_is_rejected():
    run = make_run(holdout_score=0.70)

    reasons = hard_check_reasons(run, EvaluationChecks(), higher_is_better=True)

    assert len(reasons) == 1
    assert "holdout score 0.7000 is much worse" in reasons[0]


def test_a_holdout_slightly_worse_than_validation_is_fine():
    assert hard_check_reasons(make_run(holdout_score=0.78), EvaluationChecks(), True) == []
    assert hard_check_reasons(make_run(holdout_score=0.85), EvaluationChecks(), True) == []


def test_the_holdout_rule_flips_for_metrics_where_lower_is_better():
    rmse_run = make_run(score_mean=10.0, score_std=0.2)

    worse = hard_check_reasons(
        rmse_run.model_copy(update={"holdout_score": 14.0}), EvaluationChecks(), False
    )
    better = hard_check_reasons(
        rmse_run.model_copy(update={"holdout_score": 9.0}), EvaluationChecks(), False
    )

    assert len(worse) == 1
    assert better == []


def test_the_recomputed_holdout_score_is_preferred_over_the_logged_one():
    run = make_run(holdout_score=0.80)
    checks = EvaluationChecks(recomputed_holdout_score=0.60)

    assert len(hard_check_reasons(run, checks, higher_is_better=True)) == 1


def test_only_high_severity_leakage_flags_are_hard_rules():
    flags = [
        LeakageFlag(description="refund_date predicts churn perfectly", severity="high"),
        LeakageFlag(description="id looks ordered", severity="medium"),
        LeakageFlag(description="weak hint", severity="low"),
    ]

    reasons = hard_check_reasons(make_run(), EvaluationChecks(leakage_flags=flags), True)

    assert reasons == ["High-severity leakage flag: refund_date predicts churn perfectly"]


def test_a_hard_rule_rejects_without_asking_the_llm(make_agent_context):
    agent_context = make_agent_context([checks_answer(recomputed_score=0.99)])

    evaluation = run_evaluator(evaluator_ledger(), agent_context, "experiment_001")

    assert evaluation.verdict == "reject"
    assert "differs from the logged score" in evaluation.reasons[0]
    assert evaluation.checks["recomputed_score"] == 0.99
    assert len(agent_context.llm_client.received_prompts) == 1


def test_a_clean_run_goes_to_the_llm_for_the_verdict(make_agent_context):
    agent_context = make_agent_context([checks_answer(), approve_answer()])

    evaluation = run_evaluator(evaluator_ledger(), agent_context, "experiment_001")

    verdict_prompt = agent_context.llm_client.received_prompts[1]
    assert evaluation.verdict == "approve"
    assert evaluation.reasons == ["score matches, no leakage found"]
    assert evaluation.checks["notes"] == ["looked fine"]
    assert "do not trust" in verdict_prompt.lower()
    assert "experiment_001 | start | lightgbm | score 0.8100 +/- 0.0100" in verdict_prompt
    assert '"recomputed_score": 0.81' in verdict_prompt
    assert "already passed the automatic hard rules" in verdict_prompt


def test_the_llm_can_reject_a_run_the_hard_rules_passed(make_agent_context):
    flags = [{"description": "id looks ordered", "severity": "medium"}]
    rejection = VerdictAnswer(verdict="reject", reasons=["the ordered id may leak the label"])
    agent_context = make_agent_context([checks_answer(leakage_flags=flags), rejection])

    evaluation = run_evaluator(evaluator_ledger(), agent_context, "experiment_001")

    assert evaluation.verdict == "reject"
    assert evaluation.reasons == ["the ordered id may leak the label"]


def test_the_checks_task_names_the_files_and_forbids_changing_the_run(
    make_agent_context, workspace
):
    agent_context = make_agent_context([checks_answer(), approve_answer()])

    run_evaluator(evaluator_ledger(), agent_context, "experiment_001")

    coder_prompt = agent_context.llm_client.received_prompts[0]
    evaluation_folder = workspace.artifacts_folder / "evaluation" / "experiment_001"
    assert "Run script: /work/experiment_001/train_model.py" in coder_prompt
    assert "validation_predictions.csv" in coder_prompt
    assert f"Splits file: {workspace.splits_path}" in coder_prompt
    assert "Primary metric: macro_f1 (higher is better)" in coder_prompt
    assert f"write only inside {evaluation_folder}" in coder_prompt
    assert "Never change files in the experiment folder" in coder_prompt
    assert "Do these checks:" in coder_prompt
    assert "<CHECKS>JSON</CHECKS>" in coder_prompt
    assert "fold_of_example" in coder_prompt
    assert (evaluation_folder / "check_run.py").exists()


def test_checks_that_cannot_run_lead_to_a_visible_unverified_approval(make_agent_context):
    agent_context = make_agent_context(two_failed_coder_attempts("predictions unreadable"))
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    evaluation = run_evaluator(evaluator_ledger(), agent_context, "experiment_001")

    assert evaluation.verdict == "approve"
    assert evaluation.checks == {"unverified": True}
    assert "could not run" in evaluation.reasons[0]
    assert "predictions unreadable" in evaluation.reasons[0]
    assert any(
        e.event_type == EventType.WARNING and "WITHOUT checks" in e.message for e in received_events
    )


def test_checks_that_break_the_contract_are_treated_as_unverified(make_agent_context):
    agent_context = make_agent_context([checks_answer(leakage_flags="none")])

    evaluation = run_evaluator(evaluator_ledger(), agent_context, "experiment_001")

    assert evaluation.checks == {"unverified": True}
    assert "did not follow the contract" in evaluation.reasons[0]


def test_a_crashed_or_unknown_run_cannot_be_evaluated(make_agent_context):
    crashed_ledger = evaluator_ledger(make_run(error="out of memory", score_mean=None))
    agent_context = make_agent_context([])

    with pytest.raises(AgentFailedError, match="no result"):
        run_evaluator(crashed_ledger, agent_context, "experiment_001")
    with pytest.raises(AgentFailedError, match="experiment_099"):
        run_evaluator(evaluator_ledger(), agent_context, "experiment_099")


def test_stored_checks_are_plain_json(make_agent_context):
    flags = [{"description": "weak hint", "severity": "low"}]
    agent_context = make_agent_context([checks_answer(leakage_flags=flags), approve_answer()])

    evaluation = run_evaluator(evaluator_ledger(), agent_context, "experiment_001")

    assert json.loads(json.dumps(evaluation.checks))["leakage_flags"][0]["severity"] == "low"


def test_events_report_the_verdict(make_agent_context):
    agent_context = make_agent_context([checks_answer(), approve_answer()])
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    run_evaluator(evaluator_ledger(), agent_context, "experiment_001")

    messages = [e.message for e in received_events if e.agent_name == "Evaluator"]
    assert messages[0] == "checking experiment_001"
    assert "experiment_001 approved" in messages
