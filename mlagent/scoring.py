"""Score comparison rules used by the Feature Engineer, the Reviewer guards and the stop rules."""

import json
import math
import statistics

from mlagent.settings import MIN_RELATIVE_GAIN

MIN_SCORES_FOR_PAIRED_COMPARISON = 3


def all_finite(scores: list[float]) -> bool:
    return bool(scores) and all(math.isfinite(score) for score in scores)


def mean_and_std(scores: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation. The deviation is 0 for a single score."""
    mean_score = statistics.fmean(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return mean_score, std_score


def score_improvement(
    new_scores: list[float], best_scores: list[float], higher_is_better: bool
) -> float:
    """How much better the new mean is. Positive means better, whatever the metric direction."""
    difference = statistics.fmean(new_scores) - statistics.fmean(best_scores)
    return difference if higher_is_better else -difference


def is_counted_gain(
    new_scores: list[float], best_scores: list[float], higher_is_better: bool
) -> bool:
    """Is the new result a real gain over the best one?

    With 3 or more scores on the same splits, the mean of the per-split improvements must be
    larger than its standard error. Otherwise the improvement must beat a small relative margin.
    """
    if not all_finite(new_scores) or not all_finite(best_scores):
        return False

    can_compare_per_split = (
        len(new_scores) == len(best_scores) and len(new_scores) >= MIN_SCORES_FOR_PAIRED_COMPARISON
    )
    if can_compare_per_split:
        sign = 1 if higher_is_better else -1
        differences = [sign * (new - best) for new, best in zip(new_scores, best_scores)]
        mean_difference = statistics.fmean(differences)
        standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
        return mean_difference > 0 and mean_difference > standard_error

    improvement = score_improvement(new_scores, best_scores, higher_is_better)
    margin = MIN_RELATIVE_GAIN * abs(statistics.fmean(best_scores))
    return improvement > max(margin, 0.0) and improvement > 0


def make_setup_signature(model_family: str, model_params: dict, feature_version: int) -> str:
    """A text key for 'this model with these parameters on these features'.

    Two runs with the same key are duplicates. The feature version is part of the key, because
    the same model on new features is a real experiment.
    """
    params_text = json.dumps(model_params, sort_keys=True, default=str)
    return f"{model_family.strip().lower()}|{params_text}|features_v{feature_version}"
