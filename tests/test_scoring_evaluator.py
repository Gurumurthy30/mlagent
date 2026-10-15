"""Tests for scoring.py and evaluator hard checks: monotonicity, noise guard, and fraud/leakage rejections."""

from __future__ import annotations

import statistics

import pytest

from mlagent.agents.evaluator import (
    EvaluationChecks,
    LeakageFlag,
    hard_check_reasons,
)
from mlagent.scoring import is_counted_gain
from mlagent.state import ExperimentRun


@pytest.mark.unit
def test_is_counted_gain_noise_guard() -> None:
    """When variance/spread across splits is large relative to improvement, gain is not counted."""
    # Baseline on 5 folds: [0.80, 0.80, 0.80, 0.80, 0.80] (mean 0.80)
    baseline = [0.80, 0.80, 0.80, 0.80, 0.80]

    # Noisy new scores with positive mean gain (+0.01) but huge spread across folds:
    # Improvements: [+0.10, -0.09, +0.10, -0.09, +0.03] -> mean +0.01, but std err is ~0.04
    noisy_new = [0.90, 0.71, 0.90, 0.71, 0.83]
    assert statistics.fmean(noisy_new) > statistics.fmean(baseline)
    assert not is_counted_gain(noisy_new, baseline, higher_is_better=True)


@pytest.mark.unit
def test_is_counted_gain_monotonicity() -> None:
    """Uniformly increasing scores increases gain monotonically without increasing noise."""
    baseline = [0.80, 0.80, 0.80, 0.80, 0.80]
    # Solid uniform gain across all folds (+0.02)
    consistent_new = [0.82, 0.82, 0.82, 0.82, 0.82]
    assert is_counted_gain(consistent_new, baseline, higher_is_better=True)

    # Even larger uniform gain (+0.05) must also be counted
    higher_gain_new = [0.85, 0.85, 0.85, 0.85, 0.85]
    assert is_counted_gain(higher_gain_new, baseline, higher_is_better=True)


@pytest.mark.unit
def test_evaluator_rejects_train_val_overlap() -> None:
    """Evaluator hard checks strictly reject any train/validation overlap."""
    run = ExperimentRun(experiment_id="run_1", score_mean=0.85, score_std=0.01)
    checks = EvaluationChecks(
        recomputed_score=0.85,
        train_validation_overlap=1,  # 1 example leaked
    )
    reasons = hard_check_reasons(run, checks, higher_is_better=True)
    assert any("overlap" in r.lower() for r in reasons)


@pytest.mark.unit
def test_evaluator_rejects_score_discrepancy() -> None:
    """Evaluator hard checks reject runs where recomputed score doesn't match reported score."""
    run = ExperimentRun(experiment_id="run_1", score_mean=0.95, score_std=0.01)
    checks = EvaluationChecks(
        recomputed_score=0.80,  # 0.15 lower than reported
        train_validation_overlap=0,
    )
    reasons = hard_check_reasons(run, checks, higher_is_better=True)
    assert any("differs" in r.lower() for r in reasons)


@pytest.mark.unit
def test_evaluator_rejects_severe_holdout_drop() -> None:
    """Evaluator hard checks reject runs whose holdout score drops severely below CV score."""
    run = ExperimentRun(
        experiment_id="run_1",
        score_mean=0.90,
        score_std=0.01,
        holdout_score=0.70,  # 0.20 drop on holdout
    )
    checks = EvaluationChecks(
        recomputed_score=0.90,
        recomputed_holdout_score=0.70,
        train_validation_overlap=0,
    )
    reasons = hard_check_reasons(run, checks, higher_is_better=True)
    assert any("holdout" in r.lower() for r in reasons)


@pytest.mark.unit
def test_evaluator_rejects_high_leakage_flags() -> None:
    """Evaluator hard checks reject runs with high severity leakage flags."""
    run = ExperimentRun(experiment_id="run_1", score_mean=0.85, score_std=0.01)
    checks = EvaluationChecks(
        recomputed_score=0.85,
        train_validation_overlap=0,
        leakage_flags=[LeakageFlag(description="Target column in feature matrix", severity="high")],
    )
    reasons = hard_check_reasons(run, checks, higher_is_better=True)
    assert any("leakage" in r.lower() for r in reasons)


@pytest.mark.unit
def test_evaluator_approves_clean_run() -> None:
    """Evaluator hard checks approve an honest run with matching scores and no leakage."""
    run = ExperimentRun(
        experiment_id="run_1",
        score_mean=0.85,
        score_std=0.01,
        holdout_score=0.845,
    )
    checks = EvaluationChecks(
        recomputed_score=0.85,
        recomputed_holdout_score=0.845,
        train_validation_overlap=0,
        leakage_flags=[],
    )
    reasons = hard_check_reasons(run, checks, higher_is_better=True)
    assert reasons == []
