"""Unit tests for Evaluator deterministic approval ratchet and metric direction handling."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from mlagent.agents.base import AgentContext
from mlagent.agents.evaluator import run_evaluator
from mlagent.configs.config import AppConfig
from mlagent.graph.state import EvaluationVerdict, ExperimentRun, Ledger
from mlagent.helper.events import EventBus
from mlagent.helper.workspace import Workspace


def _create_mock_context(tmp_path: Path, llm_verdict: EvaluationVerdict) -> AgentContext:
    workspace = Workspace(tmp_path)
    workspace.create_folders()
    event_bus = EventBus()
    config = AppConfig()

    llm_mock = MagicMock()
    llm_mock.ask_structured.return_value = llm_verdict

    return AgentContext(
        workspace=workspace,
        event_bus=event_bus,
        config=config,
        llm_client=llm_mock,
        code_runner=MagicMock(),
        environment=MagicMock(),
    )


def _write_dummy_predictions(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    df_val = pd.DataFrame({"Id": [1, 2, 3], "prediction": [10.0, 20.0, 30.0]})
    df_val.to_csv(artifact_dir / "val_predictions.csv", index=False)
    df_test = pd.DataFrame({"Id": [101, 102], "prediction": [15.0, 25.0]})
    df_test.to_csv(artifact_dir / "test_predictions.csv", index=False)


def test_evaluator_approves_baseline_even_if_llm_reports_zero_gain(tmp_path: Path) -> None:
    """When no Champion exists yet, Evaluator approves valid Baseline run and establishes Champion."""
    exp_dir = tmp_path / "artifacts" / "experiments" / "experiment_001"
    _write_dummy_predictions(exp_dir)

    ledger = Ledger(user_task="House Prices")
    ledger.brief.primary_metric = "rmse"
    ledger.brief.higher_is_better = False

    run = ExperimentRun(
        experiment_id="experiment_001",
        iteration=0,
        model_family="ridge",
        cv_mean=35000.0,
        train_score_mean=31000.0,
        artifact_dir=str(exp_dir),
    )
    ledger.runs.append(run)

    # Simulate LLM mistakenly emitting verdict="reject" because gain vs champion was 0.0
    llm_verdict = EvaluationVerdict(
        thinking="Baseline run with 0.0 gain.",
        cli_summary="Rejecting baseline due to zero gain.",
        experiment_id="experiment_001",
        verdict="reject",
        rejection_reasons=["No gain vs champion (0.0000). Missing holdout metrics."],
        overfitting_status="mild_overfit",
        train_vs_cv_gap=4000.0,
        beats_current_champion=False,
        primary_metric_gain_vs_champion=0.0,
        actionable_feedback_for_ml_lead="Proceed to tree models.",
    )

    ctx = _create_mock_context(tmp_path, llm_verdict)
    final_verdict = run_evaluator(ledger, ctx)

    # Python must have overridden the hallucinated rejection and promoted baseline to Champion!
    assert final_verdict.verdict == "approve"
    assert run.promoted_to_champion is True
    assert ledger.champion_experiment_id == "experiment_001"


def test_evaluator_lower_is_better_metric_gain_and_promotion(tmp_path: Path) -> None:
    """For lower-is-better metrics (e.g. RMSE, loss), lower CV achieves positive gain and promotes."""
    exp1_dir = tmp_path / "artifacts" / "experiments" / "experiment_001"
    _write_dummy_predictions(exp1_dir)
    exp2_dir = tmp_path / "artifacts" / "experiments" / "experiment_002"
    _write_dummy_predictions(exp2_dir)

    ledger = Ledger(user_task="House Prices")
    ledger.brief.primary_metric = "rmse"
    ledger.brief.higher_is_better = False

    # Established Champion
    champ = ExperimentRun(
        experiment_id="experiment_001",
        iteration=0,
        model_family="ridge",
        cv_mean=35000.0,
        train_score_mean=31000.0,
        promoted_to_champion=True,
        verdict="approve",
        artifact_dir=str(exp1_dir),
    )
    ledger.runs.append(champ)
    ledger.champion_experiment_id = "experiment_001"

    # Candidate with BETTER (lower) CV RMSE
    candidate = ExperimentRun(
        experiment_id="experiment_002",
        iteration=1,
        model_family="elastic_net",
        cv_mean=32000.0,
        train_score_mean=29000.0,
        artifact_dir=str(exp2_dir),
    )
    ledger.runs.append(candidate)

    llm_verdict = EvaluationVerdict(
        thinking="Candidate beats champion CV.",
        cli_summary="Approved candidate.",
        experiment_id="experiment_002",
        verdict="approve",
        rejection_reasons=[],
        overfitting_status="healthy",
        train_vs_cv_gap=3000.0,
        beats_current_champion=True,
        primary_metric_gain_vs_champion=3000.0,
        actionable_feedback_for_ml_lead="Great progress.",
    )

    ctx = _create_mock_context(tmp_path, llm_verdict)
    final_verdict = run_evaluator(ledger, ctx)

    assert final_verdict.verdict == "approve"
    assert final_verdict.primary_metric_gain_vs_champion == 3000.0
    assert candidate.promoted_to_champion is True
    assert ledger.champion_experiment_id == "experiment_002"


def test_evaluator_lower_is_better_metric_rejects_worse_score(tmp_path: Path) -> None:
    """For lower-is-better metrics (e.g. RMSE), higher CV is rejected and Champion remains unchanged."""
    exp1_dir = tmp_path / "artifacts" / "experiments" / "experiment_001"
    _write_dummy_predictions(exp1_dir)
    exp2_dir = tmp_path / "artifacts" / "experiments" / "experiment_002"
    _write_dummy_predictions(exp2_dir)

    ledger = Ledger(user_task="House Prices")
    ledger.brief.primary_metric = "rmse"
    ledger.brief.higher_is_better = False

    champ = ExperimentRun(
        experiment_id="experiment_001",
        iteration=0,
        model_family="ridge",
        cv_mean=35000.0,
        promoted_to_champion=True,
        verdict="approve",
        artifact_dir=str(exp1_dir),
    )
    ledger.runs.append(champ)
    ledger.champion_experiment_id = "experiment_001"

    # Candidate with WORSE (higher) CV RMSE
    candidate = ExperimentRun(
        experiment_id="experiment_002",
        iteration=1,
        model_family="random_forest",
        cv_mean=39000.0,
        train_score_mean=20000.0,
        artifact_dir=str(exp2_dir),
    )
    ledger.runs.append(candidate)

    llm_verdict = EvaluationVerdict(
        thinking="Candidate worse than champion.",
        cli_summary="Rejected candidate.",
        experiment_id="experiment_002",
        verdict="reject",
        rejection_reasons=["Did not beat champion CV."],
        overfitting_status="mild_overfit",
        train_vs_cv_gap=19000.0,
        beats_current_champion=False,
        primary_metric_gain_vs_champion=-4000.0,
        actionable_feedback_for_ml_lead="Try GBDT.",
    )

    ctx = _create_mock_context(tmp_path, llm_verdict)
    final_verdict = run_evaluator(ledger, ctx)

    assert final_verdict.verdict == "reject"
    assert candidate.promoted_to_champion is False
    assert ledger.champion_experiment_id == "experiment_001"


def test_evaluator_higher_is_better_metric_gain_and_promotion(tmp_path: Path) -> None:
    """For higher-is-better metrics (e.g. Accuracy, AUC), higher CV promotes and lower CV rejects."""
    exp1_dir = tmp_path / "artifacts" / "experiments" / "exp_01"
    _write_dummy_predictions(exp1_dir)
    exp2_dir = tmp_path / "artifacts" / "experiments" / "exp_02"
    _write_dummy_predictions(exp2_dir)

    ledger = Ledger(user_task="Titanic")
    ledger.brief.primary_metric = "accuracy"
    ledger.brief.higher_is_better = True

    champ = ExperimentRun(
        experiment_id="exp_01",
        iteration=0,
        model_family="logistic_regression",
        cv_mean=0.8200,
        promoted_to_champion=True,
        verdict="approve",
        artifact_dir=str(exp1_dir),
    )
    ledger.runs.append(champ)
    ledger.champion_experiment_id = "exp_01"

    # Candidate with higher accuracy (0.8500)
    candidate = ExperimentRun(
        experiment_id="exp_02",
        iteration=1,
        model_family="lightgbm",
        cv_mean=0.8500,
        train_score_mean=0.8600,
        artifact_dir=str(exp2_dir),
    )
    ledger.runs.append(candidate)

    llm_verdict = EvaluationVerdict(
        thinking="Accuracy improved.",
        cli_summary="Approved candidate.",
        experiment_id="exp_02",
        verdict="approve",
        rejection_reasons=[],
        overfitting_status="healthy",
        train_vs_cv_gap=0.0100,
        beats_current_champion=True,
        primary_metric_gain_vs_champion=0.0300,
        actionable_feedback_for_ml_lead="Tune tree depth.",
    )

    ctx = _create_mock_context(tmp_path, llm_verdict)
    final_verdict = run_evaluator(ledger, ctx)

    assert final_verdict.verdict == "approve"
    assert abs(final_verdict.primary_metric_gain_vs_champion - 0.0300) < 1e-5
    assert candidate.promoted_to_champion is True
    assert ledger.champion_experiment_id == "exp_02"


def test_evaluator_hard_failure_rejects(tmp_path: Path) -> None:
    """If predictions file is missing, Evaluator forces hard rejection even if LLM hallucinated approve."""
    exp_dir = tmp_path / "artifacts" / "experiments" / "exp_01"
    exp_dir.mkdir(parents=True, exist_ok=True)
    # val_predictions.csv NOT created

    ledger = Ledger(user_task="Titanic")
    run = ExperimentRun(
        experiment_id="exp_01",
        iteration=0,
        model_family="logistic_regression",
        cv_mean=0.8200,
        artifact_dir=str(exp_dir),
    )
    ledger.runs.append(run)

    llm_verdict = EvaluationVerdict(
        thinking="Looks good.",
        cli_summary="Approved.",
        experiment_id="exp_01",
        verdict="approve",
        rejection_reasons=[],
        overfitting_status="healthy",
        train_vs_cv_gap=0.01,
        beats_current_champion=True,
        primary_metric_gain_vs_champion=0.0,
        actionable_feedback_for_ml_lead="",
    )

    ctx = _create_mock_context(tmp_path, llm_verdict)
    final_verdict = run_evaluator(ledger, ctx)

    assert final_verdict.verdict == "reject"
    assert any("not found" in r.lower() for r in final_verdict.rejection_reasons)
