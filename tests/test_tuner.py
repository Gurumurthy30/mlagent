from pathlib import Path

import pytest
from scripted_answers import two_failed_coder_attempts

from mlagent.agents.tuner import pick_tune_targets, run_tuner
from mlagent.events import EventType
from mlagent.memory import create_new_ledger
from mlagent.state import Evaluation, ExperimentRun

FILES = ("validation_predictions.csv", "test_predictions.csv", "model.txt")


def tuned_answer(best_params=None, changes=None, files=FILES) -> str:
    result = {
        "validation_scores": [0.84, 0.85, 0.83],
        "holdout_score": None,
        "extra_scores": {"auc": 0.93},
        "validation_predictions_path": "validation_predictions.csv",
        "test_predictions_path": "test_predictions.csv",
        "model_path": "model.txt",
        "model_params": best_params if best_params is not None else {"depth": 8},
        "trials_run": 20,
    }
    result.update(changes or {})
    if best_params == "omit":
        del result["model_params"]
    script = f"""
import json
from pathlib import Path

for name in {list(files)!r}:
    Path(name).write_text("data")
print("<RESULT>" + json.dumps({result!r}) + "</RESULT>")
"""
    return f"```python\n{script}\n```"


def tuning_ledger(workspace, higher_is_better=True):
    ledger = create_new_ledger("Predict churn.", "macro F1")
    ledger.brief.summary = "Customer table."
    ledger.brief.deliverable_format = "predictions.csv"
    ledger.plan.primary_metric = "macro_f1"
    ledger.plan.higher_is_better = higher_is_better
    ledger.plan.extra_metrics = ["auc"]
    ledger.iteration = 4
    ledger.focus = "tune"
    return ledger


def add_run(ledger, family, params, score, feature_version=1, approved=True):
    experiment_id = ledger.next_experiment_id()
    ledger.runs.append(
        ExperimentRun(
            experiment_id=experiment_id,
            model_family=family,
            model_params=params,
            feature_version=feature_version,
            validation_scores=[score, score, score],
            score_mean=score,
            script_path=f"/work/{experiment_id}/train_model.py",
        )
    )
    if approved:
        ledger.evaluations.append(Evaluation(experiment_id=experiment_id, verdict="approve"))
    return experiment_id


def test_the_best_approved_run_of_each_family_is_picked_best_first(workspace):
    ledger = tuning_ledger(workspace)
    add_run(ledger, "lightgbm", {"depth": 4}, 0.80)
    add_run(ledger, "lightgbm", {"depth": 6}, 0.83)
    add_run(ledger, "XGBoost", {"depth": 3}, 0.82)
    add_run(ledger, "logistic_regression", {"c": 1}, 0.75)
    add_run(ledger, "catboost", {"depth": 6}, 0.99, approved=False)

    targets = pick_tune_targets(ledger)

    assert [run.experiment_id for run in targets] == ["experiment_002", "experiment_003"]


def test_lower_is_better_picks_the_lowest_scores(workspace):
    ledger = tuning_ledger(workspace, higher_is_better=False)
    add_run(ledger, "lightgbm", {}, 12.0)
    add_run(ledger, "lightgbm", {}, 9.0)
    add_run(ledger, "xgboost", {}, 10.0)
    add_run(ledger, "ridge", {}, 15.0)

    targets = pick_tune_targets(ledger)

    assert [run.experiment_id for run in targets] == ["experiment_002", "experiment_003"]


def test_tuned_runs_get_new_ids_the_best_parameters_and_the_round(make_agent_context, workspace):
    ledger = tuning_ledger(workspace)
    add_run(ledger, "lightgbm", {"depth": 6}, 0.83, feature_version=2)
    add_run(ledger, "xgboost", {"depth": 3}, 0.82, feature_version=1)
    agent_context = make_agent_context(
        [tuned_answer({"depth": 8}), tuned_answer({"depth": 5, "eta": 0.1})]
    )

    runs = run_tuner(ledger, agent_context)

    assert [run.experiment_id for run in runs] == ["experiment_003", "experiment_004"]
    assert [run.focus for run in runs] == ["tune", "tune"]
    assert [run.model_params for run in runs] == [{"depth": 8}, {"depth": 5, "eta": 0.1}]
    assert [run.feature_version for run in runs] == [2, 1]
    assert all(run.iteration == 4 for run in runs)
    assert runs[0].score_mean == pytest.approx(0.84)
    assert runs[0].description == 'tuned lightgbm {"depth": 8}'
    assert not any(run.crashed for run in runs)
    assert Path(runs[0].script_path).parts[-2:] == ("experiment_003", "tune_model.py")
    assert (workspace.artifacts_folder / "experiment_004" / "tune_model.py").exists()


def test_the_tuning_task_names_the_source_the_search_limits_and_the_features(
    make_agent_context, workspace
):
    ledger = tuning_ledger(workspace)
    add_run(ledger, "lightgbm", {"depth": 6}, 0.83, feature_version=2)
    agent_context = make_agent_context([tuned_answer()])

    run_tuner(ledger, agent_context)

    prompt = agent_context.llm_client.received_prompts[0]
    assert (
        "Source script (read it and follow its pattern): /work/experiment_001/train_model.py"
        in prompt
    )
    assert 'Current parameters: {"depth": 6}' in prompt
    assert "at most 20 trials with Optuna" in prompt
    assert "60 percent of the time limit (18 minutes)" in prompt
    assert str(workspace.features_file(2)) in prompt
    assert "import features_v2" in prompt
    assert "Primary metric: macro_f1 (higher is better)" in prompt
    assert '"model_params"' in prompt and '"trials_run"' in prompt
    assert "SCRIPT RULES:" in prompt


def test_a_crashed_search_is_a_crash_run_and_the_next_model_is_still_tuned(
    make_agent_context, workspace
):
    ledger = tuning_ledger(workspace)
    add_run(ledger, "lightgbm", {"depth": 6}, 0.83)
    add_run(ledger, "xgboost", {"depth": 3}, 0.82)
    agent_context = make_agent_context(
        [*two_failed_coder_attempts("optuna missing"), tuned_answer()]
    )

    runs = run_tuner(ledger, agent_context)

    assert runs[0].crashed and "optuna missing" in runs[0].error
    assert not runs[1].crashed
    assert runs[1].experiment_id == "experiment_004"


def test_a_result_without_parameters_keeps_the_source_parameters(make_agent_context, workspace):
    ledger = tuning_ledger(workspace)
    add_run(ledger, "lightgbm", {"depth": 6}, 0.83)
    agent_context = make_agent_context([tuned_answer("omit")])

    runs = run_tuner(ledger, agent_context)

    assert runs[0].model_params == {"depth": 6}
    assert not runs[0].crashed


def test_a_result_that_breaks_the_contract_is_a_crash_run(make_agent_context, workspace):
    ledger = tuning_ledger(workspace)
    add_run(ledger, "lightgbm", {"depth": 6}, 0.83)
    agent_context = make_agent_context([tuned_answer(files=("model.txt",))])

    runs = run_tuner(ledger, agent_context)

    assert runs[0].crashed
    assert "does not exist" in runs[0].error


def test_without_approved_runs_nothing_is_tuned(make_agent_context, workspace):
    ledger = tuning_ledger(workspace)
    add_run(ledger, "lightgbm", {}, 0.8, approved=False)
    agent_context = make_agent_context([])

    assert run_tuner(ledger, agent_context) == []


def test_the_input_ledger_is_not_changed_and_events_are_emitted(make_agent_context, workspace):
    ledger = tuning_ledger(workspace)
    add_run(ledger, "lightgbm", {"depth": 6}, 0.83)
    agent_context = make_agent_context([tuned_answer()])
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    run_tuner(ledger, agent_context)

    assert len(ledger.runs) == 1
    tuner_events = [e for e in received_events if e.agent_name == "Tuner"]
    assert tuner_events[0].message == "tuning 1 model(s)"
    assert any(e.event_type == EventType.DECISION for e in tuner_events)
    assert tuner_events[-1].message == "1 tuned run(s)"
