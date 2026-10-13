import pytest

from mlagent.memory import create_new_ledger
from mlagent.state import Evaluation, ExperimentRun, Review
from mlagent.stop_rules import (
    apply_reviewer_guards,
    count_trailing_crashes,
    decide_stop_reason,
    find_focus_problem,
    has_reached_target,
    is_on_plateau,
    round_had_counted_gain,
)

GOOD_SCORES = [0.82, 0.83, 0.82, 0.84, 0.83]
BASE_SCORES = [0.80, 0.80, 0.80, 0.80, 0.80]


def make_ledger(higher_is_better=True):
    ledger = create_new_ledger("task", "metric")
    ledger.plan.primary_metric = "metric"
    ledger.plan.higher_is_better = higher_is_better
    return ledger


def add_run(ledger, scores=None, iteration=0, error="", approved=True):
    experiment_id = ledger.next_experiment_id()
    mean_score = sum(scores) / len(scores) if scores else None
    ledger.runs.append(
        ExperimentRun(
            experiment_id=experiment_id,
            iteration=iteration,
            validation_scores=scores or [],
            score_mean=mean_score,
            error=error,
        )
    )
    if approved and not error:
        ledger.evaluations.append(Evaluation(experiment_id=experiment_id, verdict="approve"))
    return experiment_id


def add_review(ledger, reviewed_focus, focus="model", counted_gain=False):
    ledger.reviews.append(
        Review(
            iteration=len(ledger.reviews) + 1,
            focus=focus,
            reviewed_focus=reviewed_focus,
            counted_gain=counted_gain,
        )
    )


def review_for(focus, ledger, reviewed_focus="model", counted_gain=False, question="") -> Review:
    return Review(
        iteration=len(ledger.reviews) + 1,
        focus=focus,
        reviewed_focus=reviewed_focus,
        counted_gain=counted_gain,
        question=question,
    )


def ledger_ready_to_finish():
    """Three finished rounds (start, features, model) and an approved run."""
    ledger = make_ledger()
    add_run(ledger, BASE_SCORES, iteration=0)
    add_review(ledger, "start", focus="features", counted_gain=True)
    add_review(ledger, "features", focus="model")
    return ledger


def test_the_first_approved_run_counts_as_progress():
    ledger = make_ledger()
    add_run(ledger, BASE_SCORES, iteration=0)

    assert round_had_counted_gain(ledger, 0) is True


def test_a_clearly_better_run_counts_and_a_noisy_one_does_not():
    ledger = make_ledger()
    add_run(ledger, BASE_SCORES, iteration=0)
    add_run(ledger, GOOD_SCORES, iteration=1)
    add_run(ledger, [0.95, 0.70, 0.85, 0.72, 0.84], iteration=2)

    assert round_had_counted_gain(ledger, 1) is True
    assert round_had_counted_gain(ledger, 2) is False


def test_rejected_crashed_or_missing_runs_give_no_gain():
    ledger = make_ledger()
    add_run(ledger, BASE_SCORES, iteration=0)
    add_run(ledger, GOOD_SCORES, iteration=1, approved=False)
    add_run(ledger, None, iteration=2, error="boom")

    assert round_had_counted_gain(ledger, 1) is False
    assert round_had_counted_gain(ledger, 2) is False
    assert round_had_counted_gain(ledger, 3) is False


def test_the_gain_rule_flips_when_lower_is_better():
    ledger = make_ledger(higher_is_better=False)
    add_run(ledger, [10.0, 10.0, 10.0, 10.0], iteration=0)
    add_run(ledger, [9.0, 9.2, 8.9, 9.1], iteration=1)
    add_run(ledger, [11.0, 11.2, 10.9, 11.1], iteration=2)

    assert round_had_counted_gain(ledger, 1) is True
    assert round_had_counted_gain(ledger, 2) is False


def test_any_tuned_run_of_the_round_can_give_the_gain():
    ledger = make_ledger()
    add_run(ledger, BASE_SCORES, iteration=0)
    add_run(ledger, [0.79, 0.80, 0.80, 0.81, 0.80], iteration=1)
    add_run(ledger, GOOD_SCORES, iteration=1)

    assert round_had_counted_gain(ledger, 1) is True


def test_trailing_crashes_are_counted_from_the_end():
    ledger = make_ledger()
    add_run(ledger, None, error="old crash")
    add_run(ledger, BASE_SCORES)
    add_run(ledger, None, error="one")
    add_run(ledger, None, error="two")

    assert count_trailing_crashes(ledger) == 2


def test_target_reached_depends_on_the_direction():
    ledger = make_ledger()
    ledger.plan.target_score = 0.9
    add_run(ledger, [0.91, 0.92])
    assert has_reached_target(ledger) is True

    ledger.plan.target_score = 0.95
    assert has_reached_target(ledger) is False

    lower_ledger = make_ledger(higher_is_better=False)
    lower_ledger.plan.target_score = 10.0
    add_run(lower_ledger, [9.0, 9.5])
    assert has_reached_target(lower_ledger) is True


def test_no_target_or_no_approved_run_never_reaches_the_target():
    ledger = make_ledger()
    assert has_reached_target(ledger) is False
    add_run(ledger, [0.99, 0.99])
    assert has_reached_target(ledger) is False


def test_plateau_needs_three_reviews_in_a_row_without_gain():
    ledger = make_ledger()
    add_review(ledger, "start", counted_gain=True)
    add_review(ledger, "features")
    add_review(ledger, "model")
    assert is_on_plateau(ledger) is False

    add_review(ledger, "tune")
    assert is_on_plateau(ledger) is True


def test_no_stop_reason_in_a_healthy_loop():
    ledger = ledger_ready_to_finish()
    ledger.iteration = 2
    ledger.focus = "model"

    assert decide_stop_reason(ledger, max_experiment_loops=15) is None


def test_each_stop_reason_is_found():
    finish_ledger = ledger_ready_to_finish()
    finish_ledger.focus = "finish"
    assert decide_stop_reason(finish_ledger, 15) == "reviewer_finish"

    target_ledger = make_ledger()
    target_ledger.plan.target_score = 0.5
    add_run(target_ledger, BASE_SCORES)
    assert decide_stop_reason(target_ledger, 15) == "target"

    crash_ledger = make_ledger()
    for _ in range(3):
        add_run(crash_ledger, None, error="boom")
    assert decide_stop_reason(crash_ledger, 15) == "crashes"

    budget_ledger = make_ledger()
    budget_ledger.iteration = 15
    assert decide_stop_reason(budget_ledger, 15) == "budget"
    budget_ledger.iteration = 4
    assert decide_stop_reason(budget_ledger, 4) == "budget"

    plateau_ledger = make_ledger()
    for focus in ("features", "model", "tune"):
        add_review(plateau_ledger, focus)
    plateau_ledger.iteration = 3
    assert decide_stop_reason(plateau_ledger, 15) == "plateau"


def test_the_reviewer_finishing_wins_over_other_reasons():
    ledger = make_ledger()
    ledger.plan.target_score = 0.5
    ledger.iteration = 20
    add_run(ledger, BASE_SCORES)
    ledger.focus = "finish"

    assert decide_stop_reason(ledger, 15) == "reviewer_finish"


def test_a_legal_choice_is_kept_without_a_note():
    ledger = ledger_ready_to_finish()
    review = review_for("finish", ledger, reviewed_focus="model")

    guarded_review = apply_reviewer_guards(ledger, review)

    assert guarded_review.focus == "finish"
    assert guarded_review.guard_note == ""


def test_finishing_too_early_is_replaced_and_the_reason_is_recorded():
    ledger = make_ledger()
    add_run(ledger, BASE_SCORES)
    review = review_for("finish", ledger, reviewed_focus="start", counted_gain=True)

    guarded_review = apply_reviewer_guards(ledger, review)

    assert guarded_review.focus == "features"
    assert "finish was replaced by features: too early" in guarded_review.guard_note


def test_finishing_without_a_model_round_goes_to_a_model_round():
    ledger = make_ledger()
    add_run(ledger, BASE_SCORES)
    add_review(ledger, "start", focus="features", counted_gain=True)
    add_review(ledger, "features", focus="features")
    review = review_for("finish", ledger, reviewed_focus="features")

    guarded_review = apply_reviewer_guards(ledger, review)

    assert guarded_review.focus == "model"
    assert "no model round has run yet" in guarded_review.guard_note


def test_finishing_without_any_approved_run_is_replaced():
    ledger = make_ledger()
    add_run(ledger, None, error="boom")
    add_review(ledger, "start", focus="features")
    add_review(ledger, "features", focus="model")
    review = review_for("finish", ledger, reviewed_focus="model")

    guarded_review = apply_reviewer_guards(ledger, review)

    assert guarded_review.focus != "finish"
    assert "no approved run" in guarded_review.guard_note


def test_a_data_round_counts_as_the_features_round_needed_to_finish():
    ledger = make_ledger()
    add_run(ledger, BASE_SCORES)
    add_review(ledger, "start", focus="data", counted_gain=True)
    add_review(ledger, "data", focus="model")
    review = review_for("finish", ledger, reviewed_focus="model")

    assert find_focus_problem(ledger, review) == ""


def test_tuning_needs_an_approved_run():
    ledger = make_ledger()
    review = review_for("tune", ledger, reviewed_focus="start")

    assert "no approved run" in find_focus_problem(ledger, review)


def test_tuning_is_allowed_when_it_never_ran_and_blocked_just_after_a_tune_round():
    ledger = make_ledger()
    add_run(ledger, BASE_SCORES)
    add_review(ledger, "start", focus="tune", counted_gain=True)
    review = review_for("tune", ledger, reviewed_focus="start")
    assert find_focus_problem(ledger, review) == ""

    ledger.last_tuned_iteration = 2
    add_review(ledger, "tune", focus="model")
    add_review(ledger, "model", focus="features")
    too_recent = review_for("tune", ledger, reviewed_focus="features")
    assert "too recent" in find_focus_problem(ledger, too_recent)

    add_review(ledger, "features", focus="model")
    long_enough = review_for("tune", ledger, reviewed_focus="model")
    assert find_focus_problem(ledger, long_enough) == ""


def test_a_data_round_needs_a_question_and_is_limited_to_two_reopenings():
    ledger = make_ledger()
    without_question = review_for("data", ledger, question="  ")
    assert "needs a question" in find_focus_problem(ledger, without_question)

    with_question = review_for("data", ledger, question="Do train and test overlap?")
    assert find_focus_problem(ledger, with_question) == ""

    ledger.data_reentries_used = 2
    assert "already reopened 2 times" in find_focus_problem(ledger, with_question)


def test_an_unusable_data_choice_is_replaced_by_the_next_focus_in_order():
    ledger = make_ledger()
    review = review_for("data", ledger, question="")

    guarded_review = apply_reviewer_guards(ledger, review)

    assert guarded_review.focus == "features"
    assert "data was replaced by features" in guarded_review.guard_note


def test_the_same_focus_three_times_without_gain_is_replaced():
    ledger = make_ledger()
    for _ in range(3):
        add_review(ledger, "features", focus="features")
    review = review_for("features", ledger, reviewed_focus="features")
    assert review.focus == "features"

    guarded_review = apply_reviewer_guards(ledger, review)

    assert guarded_review.focus == "model"
    assert "features gave no counted gain in the last 3 rounds" in guarded_review.guard_note


@pytest.mark.parametrize("gain_in_window", [True, False])
def test_a_counted_gain_in_the_window_removes_the_stuck_rule(gain_in_window):
    ledger = make_ledger()
    add_review(ledger, "features", focus="features")
    add_review(ledger, "features", focus="features", counted_gain=gain_in_window)
    review = review_for("features", ledger, reviewed_focus="features")

    problem = find_focus_problem(ledger, review)

    assert (problem == "") is gain_in_window


def test_a_different_focus_in_the_window_removes_the_stuck_rule():
    ledger = make_ledger()
    add_review(ledger, "features", focus="model")
    add_review(ledger, "model", focus="features")
    review = review_for("features", ledger, reviewed_focus="features")

    assert find_focus_problem(ledger, review) == ""


def test_the_replacement_for_a_stuck_model_focus_skips_unavailable_choices():
    ledger = make_ledger()
    for _ in range(3):
        add_review(ledger, "model", focus="model")
    review = review_for("model", ledger, reviewed_focus="model")

    guarded_review = apply_reviewer_guards(ledger, review)

    assert guarded_review.focus == "features"
