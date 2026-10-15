"""Score comparison rules used by the Feature Engineer, the Reviewer guards and the stop rules."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError


if TYPE_CHECKING:
    from mlagent.state import ExperimentRun

MIN_SCORES_FOR_PAIRED_COMPARISON = 3


class ModelRunResult(BaseModel):
    validation_scores: list[float]
    holdout_score: float | None = None
    extra_scores: dict[str, Any] = {}
    validation_predictions_path: str = ""
    test_predictions_path: str = ""
    model_path: str = ""
    notes: str = ""


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
    return improvement > 0


def make_setup_signature(model_family: str, model_params: dict, feature_version: int) -> str:
    """A text key for 'this model with these parameters on these features'.

    Two runs with the same key are duplicates. The feature version is part of the key, because
    the same model on new features is a real experiment.
    """
    params_text = json.dumps(model_params, sort_keys=True, default=str)
    return f"{model_family.strip().lower()}|{params_text}|features_v{feature_version}"


def fill_run_from_result(run: ExperimentRun, result_data: dict, experiment_folder: Path) -> str:
    """Copy the script's result into the run. Returns the problem found, or '' when it is fine."""
    try:
        result = ModelRunResult.model_validate(result_data)
    except ValidationError as error:
        return f"The result line did not follow the contract: {error}"
    if not all_finite(result.validation_scores):
        return "validation_scores must be a non-empty list of finite numbers."

    test_predictions_file = experiment_folder / "test_predictions.csv"
    if not test_predictions_file.is_file():
        if result.test_predictions_path:
            p = Path(result.test_predictions_path)
            alt = p if p.is_absolute() else experiment_folder / p
            if alt.is_file():
                test_predictions_file = alt
        if not test_predictions_file.is_file():
            return f"The test predictions file does not exist: {test_predictions_file}"

    model_path_text = ""
    if result.model_path:
        p = Path(result.model_path)
        model_path = p if p.is_absolute() else experiment_folder / p
        if not model_path.exists():
            return f"The model file does not exist: {model_path}"
        model_path_text = str(model_path)

    run.validation_scores = result.validation_scores
    run.score_mean, run.score_std = mean_and_std(result.validation_scores)
    run.holdout_score = result.holdout_score
    run.extra_scores = {
        str(k): float(v)
        for k, v in result.extra_scores.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    run.model_path = model_path_text
    return ""
