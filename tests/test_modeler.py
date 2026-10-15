import pytest
from scripted_answers import two_failed_coder_attempts

from mlagent.agents.modeler import ModelChoice, run_modeler
from mlagent.events import EventType
from mlagent.memory import create_new_ledger
from mlagent.state import Evaluation, ExperimentRun, Review

ALL_FILES = ("validation_predictions.csv", "test_predictions.csv", "model.txt")


def result_answer(result_changes=None, files=ALL_FILES, remove_keys=()) -> str:
    """A Coder answer: a script that creates files and prints the result line."""
    result = {
        "validation_scores": [0.80, 0.82, 0.81],
        "holdout_score": None,
        "extra_scores": {"auc": 0.9},
        "validation_predictions_path": "validation_predictions.csv",
        "test_predictions_path": "test_predictions.csv",
        "model_path": "model.txt",
        "notes": "ok",
    }
    result.update(result_changes or {})
    for key in remove_keys:
        del result[key]
    script = f"""
import json
from pathlib import Path

for name in {list(files)!r}:
    Path(name).write_text("data")
result = {result!r}
result["validation_scores"] = [
    float("nan") if value == "nan" else value for value in result["validation_scores"]
]
print("<RESULT>" + json.dumps(result) + "</RESULT>")
"""
    return f"```python\n{script}\n```"


def modeler_ledger(workspace, focus="start", feature_version=1):
    ledger = create_new_ledger("Predict churn.", "macro F1")
    ledger.brief.summary = "Customer table."
    ledger.brief.deliverable_format = "predictions.csv with columns id, churn"
    ledger.plan.primary_metric = "macro_f1"
    ledger.plan.extra_metrics = ["auc"]
    ledger.plan.baseline_description = "logistic regression on basic features"
    ledger.plan.validation_scheme = "stratified_kfold"
    ledger.plan.splits_path = str(workspace.splits_path)
    ledger.current_feature_version = feature_version
    ledger.focus = focus
    workspace.features_folder.mkdir(parents=True, exist_ok=True)
    for version in range(1, feature_version + 1):
        workspace.features_file(version).write_text(
            f"def prepare_inputs_v{version}(table):\n    return table\n", encoding="utf-8"
        )
    return ledger


def add_run(ledger, family, params, feature_version=1, error="", approved=True, score=0.8):
    experiment_id = ledger.next_experiment_id()
    ledger.runs.append(
        ExperimentRun(
            experiment_id=experiment_id,
            model_family=family,
            model_params=params,
            feature_version=feature_version,
            score_mean=None if error else score,
            error=error,
        )
    )
    if approved and not error:
        ledger.evaluations.append(Evaluation(experiment_id=experiment_id, verdict="approve"))


def lightgbm_choice(**params) -> ModelChoice:
    return ModelChoice(
        model_family="lightgbm", model_params=params or {"depth": 6}, reason="strong on tables"
    )


def test_the_baseline_run_is_filled_from_the_script_result(make_agent_context, workspace):
    ledger = modeler_ledger(workspace)
    agent_context = make_agent_context([lightgbm_choice(), result_answer()])

    run = run_modeler(ledger, agent_context)

    experiment_folder = workspace.artifacts_folder / "experiment_001"
    assert not run.crashed
    assert run.experiment_id == "experiment_001"
    assert run.focus == "start"
    assert run.model_family == "lightgbm"
    assert run.model_params == {"depth": 6}
    assert run.description == 'lightgbm {"depth": 6}'
    assert run.feature_version == 1
    assert run.validation_scores == [0.80, 0.82, 0.81]
    assert run.score_mean == pytest.approx(0.81)
    assert run.score_std == pytest.approx(0.01)
    assert run.holdout_score is None
    assert run.extra_scores == {"auc": 0.9}
    assert run.test_predictions_path == str(experiment_folder / "test_predictions.csv")
    assert run.validation_predictions_path == str(experiment_folder / "validation_predictions.csv")
    assert run.model_path == str(experiment_folder / "model.txt")
    assert run.script_path == str(experiment_folder / "train_model.py")
    assert run.artifact_dir == str(experiment_folder)


def test_the_baseline_choice_prompt_and_the_training_task(make_agent_context, workspace):
    ledger = modeler_ledger(workspace)
    agent_context = make_agent_context([lightgbm_choice(), result_answer()])

    run_modeler(ledger, agent_context)

    choose_prompt, training_prompt = agent_context.llm_client.received_prompts
    assert (
        "Train the baseline from the plan: logistic regression on basic features" in choose_prompt
    )
    assert "Never repeat a setup from SETUPS TRIED" in choose_prompt
    assert "Time limit for one script run: 30 minutes" in choose_prompt
    assert "Model: lightgbm" in training_prompt
    assert 'Parameters: {"depth": 6}' in training_prompt
    assert "Primary metric: macro_f1 (higher is better)" in training_prompt
    assert "Extra metrics: auc" in training_prompt
    assert "Use the GPU: no" in training_prompt
    assert "Seed: 42" in training_prompt
    assert str(workspace.splits_path) in training_prompt
    assert f"import {workspace.features_file(1).stem}" in training_prompt
    assert "predictions.csv with columns id, churn" in training_prompt
    assert "SCRIPT RULES:" in training_prompt
    assert "inside each split" in training_prompt
    assert '"fold_of_example"' in training_prompt
    assert "def prepare_inputs_v1" in training_prompt
    assert "Instructions from the Reviewer:\nnone" in training_prompt


def test_a_model_round_asks_for_a_new_setup_and_passes_the_reviewers_instructions(
    make_agent_context, workspace
):
    ledger = modeler_ledger(workspace, focus="model")
    add_run(ledger, "logistic_regression", {"c": 1.0})
    ledger.reviews = [
        Review(
            iteration=1,
            focus="model",
            reason="try boosting",
            instructions=["Try gradient boosting with depth 6."],
        )
    ]
    agent_context = make_agent_context([lightgbm_choice(), result_answer()])

    run_modeler(ledger, agent_context)

    choose_prompt, training_prompt = agent_context.llm_client.received_prompts
    assert "Choose a NEW model family or setup" in choose_prompt
    assert "logistic_regression" in choose_prompt
    assert "- Try gradient boosting with depth 6." in training_prompt


def test_a_features_round_retrains_the_best_setup_without_asking_the_llm(
    make_agent_context, workspace
):
    ledger = modeler_ledger(workspace, focus="features", feature_version=2)
    add_run(ledger, "xgboost", {"depth": 3}, score=0.7)
    add_run(ledger, "lightgbm", {"depth": 6}, score=0.9)
    agent_context = make_agent_context([result_answer()])

    run = run_modeler(ledger, agent_context)

    assert len(agent_context.llm_client.received_prompts) == 1
    assert run.model_family == "lightgbm"
    assert run.model_params == {"depth": 6}
    assert run.feature_version == 2
    assert run.focus == "features"
    assert "Model: lightgbm" in agent_context.llm_client.received_prompts[0]


def test_a_features_round_without_an_approved_run_asks_the_llm(make_agent_context, workspace):
    ledger = modeler_ledger(workspace, focus="features", feature_version=2)
    agent_context = make_agent_context([lightgbm_choice(), result_answer()])

    run = run_modeler(ledger, agent_context)

    assert len(agent_context.llm_client.received_prompts) == 2
    assert run.model_family == "lightgbm"


def test_a_repeated_setup_is_refused_and_the_llm_is_asked_again(make_agent_context, workspace):
    ledger = modeler_ledger(workspace, focus="model")
    add_run(ledger, "lightgbm", {"depth": 6})
    xgboost_choice = ModelChoice(model_family="xgboost", model_params={"depth": 4})
    agent_context = make_agent_context([lightgbm_choice(depth=6), xgboost_choice, result_answer()])

    run = run_modeler(ledger, agent_context)

    second_choose_prompt = agent_context.llm_client.received_prompts[1]
    assert "Your last choice repeated an earlier run: lightgbm" in second_choose_prompt
    assert run.model_family == "xgboost"
    assert not run.crashed


def test_three_repeated_setups_make_a_crash_run_without_running_code(make_agent_context, workspace):
    ledger = modeler_ledger(workspace, focus="model")
    add_run(ledger, "lightgbm", {"depth": 6})
    agent_context = make_agent_context([lightgbm_choice(depth=6) for _ in range(3)])

    run = run_modeler(ledger, agent_context)

    assert run.crashed
    assert run.error == "Every setup the Modeler chose had already been tried."
    assert run.model_family == ""
    assert len(agent_context.llm_client.received_prompts) == 3


def test_a_crashed_run_does_not_block_trying_the_same_setup_again(make_agent_context, workspace):
    ledger = modeler_ledger(workspace, focus="model")
    add_run(ledger, "lightgbm", {"depth": 6}, error="script bug")
    agent_context = make_agent_context([lightgbm_choice(depth=6), result_answer()])

    run = run_modeler(ledger, agent_context)

    assert not run.crashed
    assert len(agent_context.llm_client.received_prompts) == 2


def test_the_same_setup_on_new_features_is_a_new_experiment(make_agent_context, workspace):
    ledger = modeler_ledger(workspace, focus="model", feature_version=2)
    add_run(ledger, "lightgbm", {"depth": 6}, feature_version=1)
    agent_context = make_agent_context([lightgbm_choice(depth=6), result_answer()])

    run = run_modeler(ledger, agent_context)

    assert not run.crashed
    assert run.feature_version == 2
    assert len(agent_context.llm_client.received_prompts) == 2


def test_a_failed_script_becomes_a_crash_run_with_the_error(make_agent_context, workspace):
    ledger = modeler_ledger(workspace)
    agent_context = make_agent_context(
        [lightgbm_choice(), *two_failed_coder_attempts("out of memory")]
    )

    run = run_modeler(ledger, agent_context)

    assert run.crashed
    assert "out of memory" in run.error
    assert run.score_mean is None
    assert run.script_path.endswith("train_model.py")
    assert run.model_family == "lightgbm"


@pytest.mark.parametrize(
    ("answer", "expected_problem"),
    [
        (result_answer(files=("validation_predictions.csv", "model.txt")), "test predictions file"),
        (result_answer(files=("test_predictions.csv", "model.txt")), "validation predictions file"),
        (
            result_answer(files=("validation_predictions.csv", "test_predictions.csv")),
            "model file",
        ),
        (result_answer({"validation_scores": [0.8, "nan", 0.7]}), "finite numbers"),
        (result_answer({"validation_scores": []}), "finite numbers"),
        (result_answer(remove_keys=("test_predictions_path",)), "did not follow the contract"),
        (result_answer({"extra_scores": "none"}), "did not follow the contract"),
    ],
)
def test_a_result_that_breaks_the_contract_becomes_a_crash_run(
    make_agent_context, workspace, answer, expected_problem
):
    ledger = modeler_ledger(workspace)
    agent_context = make_agent_context([lightgbm_choice(), answer])

    run = run_modeler(ledger, agent_context)

    assert run.crashed
    assert expected_problem in run.error
    assert run.score_mean is None


def test_a_script_without_validation_predictions_is_fine(make_agent_context, workspace):
    ledger = modeler_ledger(workspace)
    answer = result_answer(
        {"validation_predictions_path": "", "model_path": ""}, files=("test_predictions.csv",)
    )
    agent_context = make_agent_context([lightgbm_choice(), answer])

    run = run_modeler(ledger, agent_context)

    assert not run.crashed
    assert run.validation_predictions_path == ""
    assert run.model_path == ""


def test_the_holdout_score_is_kept(make_agent_context, workspace):
    ledger = modeler_ledger(workspace)
    agent_context = make_agent_context([lightgbm_choice(), result_answer({"holdout_score": 0.79})])

    assert run_modeler(ledger, agent_context).holdout_score == 0.79


def test_the_second_run_gets_the_next_experiment_id_and_folder(make_agent_context, workspace):
    ledger = modeler_ledger(workspace, focus="model")
    add_run(ledger, "logistic_regression", {"c": 1.0})
    agent_context = make_agent_context([lightgbm_choice(), result_answer()])

    run = run_modeler(ledger, agent_context)

    assert run.experiment_id == "experiment_002"
    assert (workspace.artifacts_folder / "experiment_002" / "train_model.py").exists()


def test_events_show_the_decision_and_the_score(make_agent_context, workspace):
    ledger = modeler_ledger(workspace)
    agent_context = make_agent_context([lightgbm_choice(), result_answer()])
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    run_modeler(ledger, agent_context)

    modeler_events = [e for e in received_events if e.agent_name == "Modeler"]
    assert modeler_events[0].message == "experiment_001, focus start"
    assert any(e.event_type == EventType.DECISION for e in modeler_events)
    assert modeler_events[-1].message == "experiment_001 score 0.8100"
