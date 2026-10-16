"""Tests for model approval, best model selection, and metric direction (higher vs lower is better)."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from mlagent.agents.evaluator import EvaluationChecks, LeakageFlag, hard_check_reasons
from mlagent.agents.submitter import pick_ensemble_runs
from mlagent.scoring import is_counted_gain
from mlagent.state import ExperimentRun, Ledger, ProjectBrief


@pytest.mark.unit
def test_best_approved_run_higher_is_better() -> None:
    """When higher_is_better=True (e.g. accuracy/AUC), highest score among approved runs is selected."""
    ledger = Ledger(
        brief=ProjectBrief(primary_metric="accuracy", higher_is_better=True),
        runs=[
            ExperimentRun(
                experiment_id="exp_001",
                model_family="lightgbm",
                score_mean=0.7874,
                verdict="reject",
                verdict_reasons=["False leakage flag: val_predictions.csv is missing"],
            ),
            ExperimentRun(
                experiment_id="exp_006",
                model_family="lightgbm",
                score_mean=0.7984,
                verdict="reject",
                verdict_reasons=["False leakage flag: test_predictions.csv is missing"],
            ),
            ExperimentRun(
                experiment_id="exp_007",
                model_family="catboost",
                score_mean=None,
                crashed=True,
            ),
            ExperimentRun(
                experiment_id="exp_008",
                model_family="xgboost",
                score_mean=0.7340,
                verdict="approve",
            ),
        ],
    )

    # Only exp_008 is approved, so it is returned despite lower score
    assert ledger.best_approved_run().experiment_id == "exp_008"
    assert ledger.best_approved_run().score_mean == 0.7340

    # If exp_006 is approved, it beats exp_008 because 0.7984 > 0.7340
    ledger.runs[1].verdict = "approve"
    assert ledger.best_approved_run().experiment_id == "exp_006"
    assert ledger.best_approved_run().score_mean == 0.7984


@pytest.mark.unit
def test_best_approved_run_lower_is_better() -> None:
    """When higher_is_better=False (e.g. RMSE/MAE/log_loss), lowest score among approved runs is selected."""
    ledger = Ledger(
        brief=ProjectBrief(primary_metric="rmse", higher_is_better=False),
        runs=[
            ExperimentRun(
                experiment_id="exp_001",
                model_family="lightgbm",
                score_mean=0.4500,
                verdict="approve",
            ),
            ExperimentRun(
                experiment_id="exp_002",
                model_family="xgboost",
                score_mean=0.3800,
                verdict="approve",
            ),
            ExperimentRun(
                experiment_id="exp_003",
                model_family="catboost",
                score_mean=0.3100,
                verdict="reject",
                verdict_reasons=["High leakage"],
            ),
            ExperimentRun(
                experiment_id="exp_004",
                model_family="ridge",
                score_mean=0.1500,
                crashed=True,
            ),
        ],
    )

    # exp_002 (0.3800) beats exp_001 (0.4500) because lower is better. exp_003 is lower but rejected, exp_004 crashed.
    best = ledger.best_approved_run()
    assert best is not None
    assert best.experiment_id == "exp_002"
    assert best.score_mean == 0.3800

    # If exp_003 is approved, it becomes the new best
    ledger.runs[2].verdict = "approve"
    assert ledger.best_approved_run().experiment_id == "exp_003"
    assert ledger.best_approved_run().score_mean == 0.3100


@pytest.mark.unit
def test_pick_ensemble_runs_direction() -> None:
    """pick_ensemble_runs ranks models correctly for both higher_is_better and lower_is_better."""
    # 1. Higher is better
    ledger_higher = Ledger(
        brief=ProjectBrief(primary_metric="accuracy", higher_is_better=True),
        runs=[
            ExperimentRun(experiment_id="exp_lgb_1", model_family="lightgbm", score_mean=0.78, verdict="approve", artifact_dir="exp_lgb_1"),
            ExperimentRun(experiment_id="exp_lgb_2", model_family="lightgbm", score_mean=0.82, verdict="approve", artifact_dir="exp_lgb_2"),
            ExperimentRun(experiment_id="exp_xgb_1", model_family="xgboost", score_mean=0.79, verdict="approve", artifact_dir="exp_xgb_1"),
            ExperimentRun(experiment_id="exp_cat_1", model_family="catboost", score_mean=0.80, verdict="reject", artifact_dir="exp_cat_1"),
        ],
    )
    ensemble_higher = pick_ensemble_runs(ledger_higher)
    # Picks best per family: lightgbm (exp_lgb_2: 0.82) and xgboost (exp_xgb_1: 0.79), sorted descending
    assert [r.experiment_id for r in ensemble_higher] == ["exp_lgb_2", "exp_xgb_1"]

    # 2. Lower is better
    ledger_lower = Ledger(
        brief=ProjectBrief(primary_metric="rmse", higher_is_better=False),
        runs=[
            ExperimentRun(experiment_id="exp_lgb_1", model_family="lightgbm", score_mean=0.45, verdict="approve", artifact_dir="exp_lgb_1"),
            ExperimentRun(experiment_id="exp_lgb_2", model_family="lightgbm", score_mean=0.35, verdict="approve", artifact_dir="exp_lgb_2"),
            ExperimentRun(experiment_id="exp_xgb_1", model_family="xgboost", score_mean=0.40, verdict="approve", artifact_dir="exp_xgb_1"),
            ExperimentRun(experiment_id="exp_cat_1", model_family="catboost", score_mean=0.30, verdict="reject", artifact_dir="exp_cat_1"),
        ],
    )
    ensemble_lower = pick_ensemble_runs(ledger_lower)
    # Picks best per family: lightgbm (exp_lgb_2: 0.35) and xgboost (exp_xgb_1: 0.40), sorted ascending
    assert [r.experiment_id for r in ensemble_lower] == ["exp_lgb_2", "exp_xgb_1"]


@pytest.mark.unit
def test_evaluator_hard_checks_direction() -> None:
    """Evaluator holdout shortfall check respects higher_is_better direction."""
    # Higher is better: CV 0.85, holdout 0.65 -> severe shortfall (0.20)
    run_higher = ExperimentRun(experiment_id="r1", score_mean=0.85, score_std=0.01, holdout_score=0.65)
    checks_higher = EvaluationChecks(recomputed_score=0.85, recomputed_holdout_score=0.65)
    reasons_higher = hard_check_reasons(run_higher, checks_higher, higher_is_better=True)
    assert any("holdout" in r.lower() for r in reasons_higher)

    # Lower is better: CV 0.30, holdout 0.50 -> severe shortfall (0.20 worse on holdout)
    run_lower = ExperimentRun(experiment_id="r2", score_mean=0.30, score_std=0.01, holdout_score=0.50)
    checks_lower = EvaluationChecks(recomputed_score=0.30, recomputed_holdout_score=0.50)
    reasons_lower = hard_check_reasons(run_lower, checks_lower, higher_is_better=False)
    assert any("holdout" in r.lower() for r in reasons_lower)

    # Lower is better: CV 0.30, holdout 0.28 -> holdout is better, no shortfall
    checks_lower_ok = EvaluationChecks(recomputed_score=0.30, recomputed_holdout_score=0.28)
    run_lower_ok = ExperimentRun(experiment_id="r3", score_mean=0.30, score_std=0.01, holdout_score=0.28)
    reasons_lower_ok = hard_check_reasons(run_lower_ok, checks_lower_ok, higher_is_better=False)
    assert reasons_lower_ok == []


@pytest.mark.unit
def test_evaluator_ignores_false_missing_prediction_leakage_flags() -> None:
    """When prediction files exist on disk, false 'missing' leakage flags are ignored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        val_path = Path(tmpdir) / "val_predictions.csv"
        test_path = Path(tmpdir) / "test_predictions.csv"
        val_path.write_text("example_id,prediction\n0,0.5\n", encoding="utf-8")
        test_path.write_text("PassengerId,Transported\n0001_01,True\n", encoding="utf-8")

        run = ExperimentRun(
            experiment_id="exp_001",
            score_mean=0.85,
            score_std=0.01,
            holdout_score=0.85,
            artifact_dir=tmpdir,
        )

        # Check script emitted false missing flags because it looked in evaluation/ cwd instead of exp dir
        checks = EvaluationChecks(
            recomputed_score=0.85,
            recomputed_holdout_score=0.85,
            leakage_flags=[
                LeakageFlag(description="val_predictions.csv is missing", severity="high"),
                LeakageFlag(description="test_predictions.csv is missing", severity="high"),
            ],
        )

        # With the guard, the false missing flags are filtered out since the files exist on disk
        reasons = hard_check_reasons(run, checks, higher_is_better=True)
        assert reasons == []


@pytest.mark.unit
def test_is_counted_gain_both_directions() -> None:
    """is_counted_gain correctly calculates improvement for both higher and lower metrics."""
    # Higher is better: 0.85 beats 0.80
    assert is_counted_gain([0.85, 0.85, 0.85], [0.80, 0.80, 0.80], higher_is_better=True)
    assert not is_counted_gain([0.75, 0.75, 0.75], [0.80, 0.80, 0.80], higher_is_better=True)

    # Lower is better: 0.20 beats 0.25
    assert is_counted_gain([0.20, 0.20, 0.20], [0.25, 0.25, 0.25], higher_is_better=False)
    assert not is_counted_gain([0.30, 0.30, 0.30], [0.25, 0.25, 0.25], higher_is_better=False)
