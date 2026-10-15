"""Rules that the code applies and the LLM cannot override.

When the experiment loop stops, and which focus the Reviewer is allowed to choose next.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlagent.config import LimitsConfig
from mlagent.state import Ledger, Review, ReviewFocus, StopReason

if TYPE_CHECKING:
    pass

FOCUS_ORDER: list[ReviewFocus] = ["features", "model", "tune", "data"]
MIN_EXPERIMENT_LOOPS = 3
MIN_ROUNDS_BETWEEN_TUNING = 2
STUCK_FOCUS_LIMIT = 3


def decide_stop_reason(
    ledger: Ledger, limits_or_max_loops: LimitsConfig | int = 15
) -> StopReason | None:
    """The reason the experiment loop must end now, or None to go on."""
    if isinstance(limits_or_max_loops, LimitsConfig):
        max_loops = limits_or_max_loops.max_experiment_loops
        max_crashes = limits_or_max_loops.max_consecutive_crashes
        plateau_patience = limits_or_max_loops.plateau_rounds
    else:
        max_loops = int(limits_or_max_loops)
        max_crashes = 3
        plateau_patience = 3

    if ledger.focus == "finish":
        return "reviewer_finish"

    # Check target score reached
    best_run = ledger.best_approved_run()
    target_score = ledger.plan.target_score
    if best_run is not None and target_score is not None:
        reached = (
            best_run.score_mean >= target_score
            if ledger.brief.higher_is_better
            else best_run.score_mean <= target_score
        )
        if reached:
            return "target"

    if ledger.trailing_crashes() >= max_crashes:
        return "crashes"
    if ledger.iteration >= max_loops:
        return "budget"

    # Check plateau
    latest_reviews = ledger.reviews[-plateau_patience:]
    if len(latest_reviews) == plateau_patience and not any(
        ledger.round_had_gain(r.iteration) for r in latest_reviews
    ):
        return "plateau"

    return None


def find_focus_problem(
    ledger: Ledger, review: Review, limits: LimitsConfig | None = None
) -> str:
    """Why the focus in this review is not allowed. Empty text means it is allowed."""
    max_reentries = limits.max_data_reentries if limits is not None else 2
    history = [*ledger.reviews, review]
    completed_focuses = [r.reviewed_focus for r in history]

    if review.focus == "finish":
        if len(history) < MIN_EXPERIMENT_LOOPS:
            return f"too early: at least {MIN_EXPERIMENT_LOOPS} rounds are needed, {len(history)} done"
        if "features" not in completed_focuses and "data" not in completed_focuses:
            return "no features round has run yet"
        if "model" not in completed_focuses:
            return "no model round has run yet"
        if not ledger.approved_runs():
            return "there is no approved run yet"
        return ""

    if review.focus == "tune":
        if not ledger.approved_runs():
            return "there is no approved run to tune yet"
        next_round_index = len(history)
        rounds_since_tuning = next_round_index - ledger.last_tuned_iteration
        if ledger.last_tuned_iteration >= 0 and rounds_since_tuning <= MIN_ROUNDS_BETWEEN_TUNING:
            return f"the last tune round was too recent (wait {MIN_ROUNDS_BETWEEN_TUNING} rounds)"
        return ""

    if review.focus == "data":
        if ledger.data_reentries_used >= max_reentries:
            return f"the data analysis was already reopened {max_reentries} times"
        if not review.question.strip():
            return "a data round needs a question"
        return ""

    # Check stuck focus
    latest_reviews = history[-STUCK_FOCUS_LIMIT:]
    if len(latest_reviews) >= STUCK_FOCUS_LIMIT:
        ran_same = all(r.reviewed_focus == review.focus for r in latest_reviews)
        if ran_same and not any(ledger.round_had_gain(r.iteration) for r in latest_reviews):
            return f"{review.focus} gave no counted gain in the last {STUCK_FOCUS_LIMIT} rounds"

    return ""


def pick_replacement_focus(
    ledger: Ledger, review: Review, limits: LimitsConfig | None = None
) -> ReviewFocus:
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
        if not find_focus_problem(ledger, candidate, limits=limits):
            return focus
    return "model"


def apply_reviewer_guards(
    ledger: Ledger, review: Review, limits: LimitsConfig | None = None
) -> Review:
    """Replace a focus the rules do not allow and note why. An allowed focus is kept."""
    problem = find_focus_problem(ledger, review, limits=limits)
    if not problem:
        return review
    replacement_focus = pick_replacement_focus(ledger, review, limits=limits)
    return review.model_copy(
        update={
            "focus": replacement_focus,
            "guard_note": f"{review.focus} was replaced by {replacement_focus}: {problem}.",
        }
    )
