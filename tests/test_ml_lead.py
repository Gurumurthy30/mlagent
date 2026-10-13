import json

import pytest
from scripted_answers import two_failed_coder_attempts

from mlagent.agents.base import AgentFailedError
from mlagent.agents.ml_lead import PlanDraft, decide_use_gpu, run_ml_lead
from mlagent.config import AppConfig
from mlagent.environment_probe import EnvironmentInfo
from mlagent.events import EventType
from mlagent.memory import create_new_ledger
from mlagent.state import FileInfo


def ledger_with_eda():
    ledger = create_new_ledger("Predict which customers churn.", "macro F1 mainly")
    ledger.brief.summary = "Customer table with a churn label."
    ledger.brief.modality = "tabular"
    ledger.brief.task_family = "classification"
    ledger.brief.target = "churn"
    ledger.brief.primary_metric = "macro_f1"
    ledger.brief.higher_is_better = True
    ledger.brief.deliverable_format = "predictions.csv with id and churn"
    ledger.brief.files = [FileInfo(path="train.csv", kind="table", role="train")]
    ledger.eda.validation_hints = ["customers repeat across rows, group by customer_id"]
    return ledger


def plan_draft(**changes) -> PlanDraft:
    values = {
        "validation_scheme": "group_kfold",
        "validation_reason": "customers repeat across rows",
        "primary_metric": "macro_f1",
        "higher_is_better": True,
        "extra_metrics": ["auc"],
        "baseline_description": "logistic regression on basic features",
        "target_score": None,
        "use_gpu": False,
        "risks": ["leakage through refund_date"],
        "playbook": "Use class weights. Watch the imbalance.",
    }
    values.update(changes)
    return PlanDraft(**values)


def splits_script(workspace, file_text=None) -> str:
    splits = {
        "scheme": "group_kfold",
        "seed": 42,
        "n_splits": 2,
        "id_description": "row position in train.csv",
        "example_ids": [0, 1, 2, 3],
        "fold_of_example": [0, 1, 0, 1],
    }
    text_to_write = file_text if file_text is not None else json.dumps(splits)
    script = f"""
import json
from pathlib import Path

path = Path({str(workspace.splits_path)!r})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text({text_to_write!r})
result = {{"splits_path": str(path), "n_splits": 2, "n_examples": 4, "n_holdout": 0}}
print("<SPLITS>" + json.dumps(result) + "</SPLITS>")
"""
    return f"```python\n{script}\n```"


def test_the_plan_and_the_splits_file_are_created(make_agent_context, workspace):
    agent_context = make_agent_context([plan_draft(), splits_script(workspace)])

    plan = run_ml_lead(ledger_with_eda(), agent_context)

    assert plan.validation_scheme == "group_kfold"
    assert plan.primary_metric == "macro_f1"
    assert plan.extra_metrics == ["auc"]
    assert plan.baseline_description == "logistic regression on basic features"
    assert plan.risks == ["leakage through refund_date"]
    assert plan.playbook == "Use class weights. Watch the imbalance."
    assert plan.splits_path == str(workspace.splits_path)
    assert workspace.splits_path.exists()


def test_the_plan_prompt_has_the_brief_the_eda_hints_and_the_limits(make_agent_context, workspace):
    agent_context = make_agent_context([plan_draft(), splits_script(workspace)])

    run_ml_lead(ledger_with_eda(), agent_context)

    plan_prompt = agent_context.llm_client.received_prompts[0]
    assert "Customer table with a churn label." in plan_prompt
    assert "macro F1 mainly" in plan_prompt
    assert "stratified_kfold" in plan_prompt and "repeated_seeds" in plan_prompt
    assert "Time limit for one script run: 30 minutes" in plan_prompt
    assert "risks: up to 5 items" in plan_prompt


def test_the_splits_task_names_the_scheme_the_path_the_hints_and_the_format(
    make_agent_context, workspace
):
    agent_context = make_agent_context([plan_draft(), splits_script(workspace)])

    run_ml_lead(ledger_with_eda(), agent_context)

    coder_prompt = agent_context.llm_client.received_prompts[1]
    assert "Scheme: group_kfold. Reason: customers repeat across rows" in coder_prompt
    assert f"Save the file to exactly this path: {workspace.splits_path}" in coder_prompt
    assert "group by customer_id" in coder_prompt
    assert '"fold_of_example"' in coder_prompt
    assert "Use 5 validation splits" in coder_prompt
    assert "at least 1000 examples" in coder_prompt
    assert "primary metric: macro_f1" in coder_prompt


@pytest.mark.parametrize(
    ("plan_wants_gpu", "setting", "machine_can_use_gpu", "expected"),
    [
        (True, "auto", True, True),
        (False, "auto", True, False),
        (True, "auto", False, False),
        (False, "always", True, True),
        (True, "always", False, False),
        (True, "never", True, False),
    ],
)
def test_gpu_decision_follows_the_setting_and_the_machine(
    plan_wants_gpu, setting, machine_can_use_gpu, expected
):
    assert decide_use_gpu(plan_wants_gpu, setting, machine_can_use_gpu) is expected


def test_the_plan_cannot_use_a_gpu_the_machine_does_not_have(make_agent_context, workspace):
    agent_context = make_agent_context([plan_draft(use_gpu=True), splits_script(workspace)])

    plan = run_ml_lead(ledger_with_eda(), agent_context)

    assert plan.use_gpu is False


def test_the_plan_can_use_a_gpu_when_the_machine_can(make_agent_context, workspace):
    agent_context = make_agent_context([plan_draft(use_gpu=True), splits_script(workspace)])
    agent_context.environment = EnvironmentInfo(torch_cuda_available=True, torch_version="2.5.0")

    plan = run_ml_lead(ledger_with_eda(), agent_context)

    assert plan.use_gpu is True


def test_the_config_can_force_the_gpu_off(make_agent_context, workspace):
    config = AppConfig.model_validate({"compute": {"use_gpu": "never"}})
    agent_context = make_agent_context(
        [plan_draft(use_gpu=True), splits_script(workspace)], config=config
    )
    agent_context.environment = EnvironmentInfo(torch_cuda_available=True, torch_version="2.5.0")

    assert run_ml_lead(ledger_with_eda(), agent_context).use_gpu is False


def test_an_empty_metric_in_the_draft_falls_back_to_the_brief(make_agent_context, workspace):
    ledger = ledger_with_eda()
    ledger.brief.primary_metric = "rmse"
    ledger.brief.higher_is_better = False
    draft = plan_draft(primary_metric="  ", higher_is_better=True)
    agent_context = make_agent_context([draft, splits_script(workspace)])

    plan = run_ml_lead(ledger, agent_context)

    assert plan.primary_metric == "rmse"
    assert plan.higher_is_better is False


def test_a_metric_in_the_draft_wins_over_the_brief(make_agent_context, workspace):
    ledger = ledger_with_eda()
    draft = plan_draft(primary_metric="roc_auc", higher_is_better=True)
    agent_context = make_agent_context([draft, splits_script(workspace)])

    assert run_ml_lead(ledger, agent_context).primary_metric == "roc_auc"


def test_risks_and_playbook_are_limited_and_the_target_score_is_kept(make_agent_context, workspace):
    draft = plan_draft(risks=[f"risk {n}" for n in range(9)], playbook="x" * 5000, target_score=0.9)
    agent_context = make_agent_context([draft, splits_script(workspace)])

    plan = run_ml_lead(ledger_with_eda(), agent_context)

    assert len(plan.risks) == 5
    assert len(plan.playbook) == 2000
    assert plan.target_score == 0.9


def test_a_failed_splits_script_stops_the_run_with_a_clear_message(make_agent_context):
    agent_context = make_agent_context(
        [
            plan_draft(),
            *two_failed_coder_attempts("disk full"),
            *two_failed_coder_attempts("disk full"),
        ]
    )

    with pytest.raises(AgentFailedError, match="could not create the validation splits"):
        run_ml_lead(ledger_with_eda(), agent_context)


def test_an_unusable_splits_file_stops_the_run(make_agent_context, workspace):
    broken_splits = json.dumps({"scheme": "kfold", "n_splits": 2, "example_ids": [0, 1]})
    agent_context = make_agent_context(
        [
            plan_draft(),
            splits_script(workspace, broken_splits),
            splits_script(workspace, broken_splits),
        ]
    )

    with pytest.raises(AgentFailedError, match="splits file is not usable"):
        run_ml_lead(ledger_with_eda(), agent_context)


def test_splits_script_fails_on_first_try_and_succeeds_on_second_try(make_agent_context, workspace):
    agent_context = make_agent_context(
        [plan_draft(), *two_failed_coder_attempts("first try error"), splits_script(workspace)]
    )

    plan = run_ml_lead(ledger_with_eda(), agent_context)

    assert plan.splits_path == str(workspace.splits_path)
    assert workspace.splits_path.exists()
    coder_prompts = [
        p for p in agent_context.llm_client.received_prompts if "TASK FROM THE ML Lead" in p
    ]
    assert len(coder_prompts) == 3
    assert "A first attempt failed with: RuntimeError: first try error" in coder_prompts[-1]


def test_the_input_ledger_is_not_changed(make_agent_context, workspace):
    ledger = ledger_with_eda()
    agent_context = make_agent_context([plan_draft(), splits_script(workspace)])

    run_ml_lead(ledger, agent_context)

    assert ledger.plan.primary_metric == ""


def test_events_report_the_decision(make_agent_context, workspace):
    agent_context = make_agent_context([plan_draft(), splits_script(workspace)])
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    run_ml_lead(ledger_with_eda(), agent_context)

    decisions = [e for e in received_events if e.event_type == EventType.DECISION]
    assert "validation group_kfold, metric macro_f1 (higher is better), GPU no" in (
        decisions[0].message
    )
