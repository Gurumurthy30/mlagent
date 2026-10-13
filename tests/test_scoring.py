import math

import pytest

from mlagent.scoring import (
    all_finite,
    is_counted_gain,
    make_setup_signature,
    mean_and_std,
    score_improvement,
)


def test_mean_and_std_of_several_scores():
    mean_score, std_score = mean_and_std([1.0, 2.0, 3.0])

    assert mean_score == 2.0
    assert std_score == 1.0


def test_a_single_score_has_zero_std():
    assert mean_and_std([0.7]) == (0.7, 0.0)


def test_all_finite_rejects_empty_nan_and_infinity():
    assert all_finite([0.1, 0.2])
    assert not all_finite([])
    assert not all_finite([0.1, math.nan])
    assert not all_finite([math.inf])


def test_score_improvement_is_positive_when_better_in_either_direction():
    assert score_improvement([0.9], [0.8], higher_is_better=True) == pytest.approx(0.1)
    assert score_improvement([8.0], [10.0], higher_is_better=False) == pytest.approx(2.0)
    assert score_improvement([0.7], [0.8], higher_is_better=True) == pytest.approx(-0.1)


def test_consistent_per_split_improvement_is_a_counted_gain():
    best_scores = [0.80, 0.80, 0.80, 0.80, 0.80]
    new_scores = [0.82, 0.83, 0.82, 0.84, 0.83]

    assert is_counted_gain(new_scores, best_scores, higher_is_better=True)


def test_noisy_improvement_is_not_a_counted_gain():
    best_scores = [0.80, 0.80, 0.80, 0.80, 0.80]
    new_scores = [0.95, 0.70, 0.85, 0.72, 0.84]

    assert not is_counted_gain(new_scores, best_scores, higher_is_better=True)


def test_a_higher_mean_with_one_good_split_only_is_not_a_counted_gain():
    best_scores = [0.80, 0.80, 0.80, 0.80]
    new_scores = [0.80, 0.80, 0.80, 0.90]

    assert not is_counted_gain(new_scores, best_scores, higher_is_better=True)


def test_lower_is_better_flips_the_direction():
    best_rmse = [10.0, 10.0, 10.0, 10.0]
    better_rmse = [9.0, 9.2, 8.9, 9.1]
    worse_rmse = [11.0, 11.2, 10.9, 11.1]

    assert is_counted_gain(better_rmse, best_rmse, higher_is_better=False)
    assert not is_counted_gain(worse_rmse, best_rmse, higher_is_better=False)


def test_fewer_than_three_scores_must_beat_the_relative_margin():
    assert is_counted_gain([0.8021], [0.80], higher_is_better=True)
    assert not is_counted_gain([0.8005], [0.80], higher_is_better=True)
    assert not is_counted_gain([0.79], [0.80], higher_is_better=True)


def test_scores_of_different_lengths_use_the_relative_margin():
    assert is_counted_gain([0.85, 0.85, 0.85, 0.85], [0.80], higher_is_better=True)


def test_nan_scores_are_never_a_gain():
    assert not is_counted_gain([0.9, math.nan, 0.9], [0.8, 0.8, 0.8], higher_is_better=True)
    assert not is_counted_gain([0.9, 0.9, 0.9], [], higher_is_better=True)


def test_setup_signature_ignores_case_and_parameter_order():
    first_signature = make_setup_signature("LightGBM", {"depth": 6, "lr": 0.1}, 1)
    second_signature = make_setup_signature(" lightgbm ", {"lr": 0.1, "depth": 6}, 1)

    assert first_signature == second_signature


def test_setup_signature_differs_for_other_parameters_or_feature_versions():
    base_signature = make_setup_signature("lightgbm", {"depth": 6}, 1)

    assert base_signature != make_setup_signature("lightgbm", {"depth": 7}, 1)
    assert base_signature != make_setup_signature("lightgbm", {"depth": 6}, 2)
    assert base_signature != make_setup_signature("xgboost", {"depth": 6}, 1)
