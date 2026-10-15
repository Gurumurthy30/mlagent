"""Rules that the code applies and the LLM cannot override.

When the experiment loop stops, and which focus the Reviewer is allowed to choose next.
"""

from mlagent.settings import (
    MAX_CONSECUTIVE_CRASHES,
    MAX_DATA_REENTRIES,
    MIN_EXPERIMENT_LOOPS,
    MIN_ROUNDS_BETWEEN_TUNING,
    PLATEAU_ROUNDS,
    STUCK_FOCUS_LIMIT,
)
from mlagent.state import Ledger, Review, ReviewFocus, StopReason

FOCUS_ORDER: list[ReviewFocus] = ["features", "model", "tune", "data"]


def round_had_counted_gain(ledger: Ledger, round_index: int) -> bool:
    """Did any run in this round produce a counted gain?"""
    return ledger.round_had_gain(round_index)


def has_reached_target(ledger: Ledger) -> bool:
    best_run = ledger.best_approved_run()
    target_score = ledger.plan.target_score
    if best_run is None or target_score is None:
        return False
    if ledger.brief.higher_is_better:
        return best_run.score_mean >= target_score
    return best_run.score_mean <= target_score


def is_on_plateau(ledger: Ledger) -> bool:
    latest_reviews = ledger.reviews[-PLATEAU_ROUNDS:]
    return len(latest_reviews) == PLATEAU_ROUNDS and not any(
        ledger.round_had_gain(review.iteration) for review in latest_reviews
    )


def decide_stop_reason(ledger: Ledger, max_experiment_loops: int) -> StopReason | None:
    """The reason the experiment loop must end now, or None to go on."""
    if ledger.focus == "finish":
        return "reviewer_finish"
    if has_reached_target(ledger):
        return "target"
    if ledger.trailing_crashes() >= MAX_CONSECUTIVE_CRASHES:
        return "crashes"
    if ledger.iteration >= max_experiment_loops:
        return "budget"
    if is_on_plateau(ledger):
        return "plateau"
    return None


def describe_finish_problem(ledger: Ledger, history: list[Review]) -> str:
    completed_focuses = [review.reviewed_focus for review in history]
    if len(history) < MIN_EXPERIMENT_LOOPS:
        return f"too early: at least {MIN_EXPERIMENT_LOOPS} rounds are needed, {len(history)} done"
    if "features" not in completed_focuses and "data" not in completed_focuses:
        return "no features round has run yet"
    if "model" not in completed_focuses:
        return "no model round has run yet"
    if not ledger.approved_runs():
        return "there is no approved run yet"
    return ""


def describe_tune_problem(ledger: Ledger, history: list[Review]) -> str:
    if not ledger.approved_runs():
        return "there is no approved run to tune yet"
    next_round_index = len(history)
    rounds_since_tuning = next_round_index - ledger.last_tuned_iteration
    if ledger.last_tuned_iteration >= 0 and rounds_since_tuning <= MIN_ROUNDS_BETWEEN_TUNING:
        return f"the last tune round was too recent (wait {MIN_ROUNDS_BETWEEN_TUNING} rounds)"
    return ""


def describe_data_problem(ledger: Ledger, review: Review) -> str:
    if ledger.data_reentries_used >= MAX_DATA_REENTRIES:
        return f"the data analysis was already reopened {MAX_DATA_REENTRIES} times"
    if not review.question.strip():
        return "a data round needs a question"
    return ""


def describe_stuck_problem(ledger: Ledger, history: list[Review], focus: ReviewFocus) -> str:
    latest_reviews = history[-STUCK_FOCUS_LIMIT:]
    if len(latest_reviews) < STUCK_FOCUS_LIMIT:
        return ""
    ran_same_focus = all(review.reviewed_focus == focus for review in latest_reviews)
    if ran_same_focus and not any(ledger.round_had_gain(r.iteration) for r in latest_reviews):
        return f"{focus} gave no counted gain in the last {STUCK_FOCUS_LIMIT} rounds"
    return ""


def find_focus_problem(ledger: Ledger, review: Review) -> str:
    """Why the focus in this review is not allowed. Empty text means it is allowed."""
    history = [*ledger.reviews, review]
    if review.focus == "finish":
        return describe_finish_problem(ledger, history)
    if review.focus == "tune":
        return describe_tune_problem(ledger, history)
    if review.focus == "data":
        return describe_data_problem(ledger, review)
    return describe_stuck_problem(ledger, history, review.focus)


def pick_replacement_focus(ledger: Ledger, review: Review) -> ReviewFocus:
    completed_focuses = [r.reviewed_focus for r in [*ledger.reviews, review]]
    if review.focus == "finish":
        if "features" not in completed_focuses and "data" not in completed_focuses:
            return "features"
        if "model" not in completed_focuses:
            return "model"

    start_index = FOCUS_ORDER.index(review.focus) + 1 if review.focus in FOCUS_ORDER else 0
    rotated_order = FOCUS_ORDER[start_index:] + FOCUS_ORDER[:start_index]
    for focus in rotated_order:
        candidate = review.model_copy(update={"focus": focus})
        if not find_focus_problem(ledger, candidate):
            return focus
    return "model"


def apply_reviewer_guards(ledger: Ledger, review: Review) -> Review:
    """Replace a focus the rules do not allow and note why. An allowed focus is kept."""
    problem = find_focus_problem(ledger, review)
    if not problem:
        return review
    replacement_focus = pick_replacement_focus(ledger, review)
    return review.model_copy(
        update={
            "focus": replacement_focus,
            "guard_note": f"{review.focus} was replaced by {replacement_focus}: {problem}.",
        }
    )
