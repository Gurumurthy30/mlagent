"""Checks on the final predictions file, before it is handed to the user.

For a CSV file with a sample submission in the project the file must match it. For other files
(images, audio, JSON, folders) the check is that something was written.
"""

from pathlib import Path
from typing import Any

import pandas as pd

SAMPLE_SUBMISSION_ROLE = "sample_submission"


# Search project files for sample submission template
def find_sample_submission(brief: Any, project_folder: Path) -> Path | None:
    # Retrieve files from dataset profile or direct files attribute
    files = (
        brief.profile.files
        if hasattr(brief, "profile") and brief.profile
        else getattr(brief, "files", [])
    )
    for file_info in files:
        role = getattr(file_info, "inferred_role", getattr(file_info, "role", ""))
        file_path_str = getattr(file_info, "path", "")
        if role == SAMPLE_SUBMISSION_ROLE and file_path_str.lower().endswith(".csv"):
            sample_path = project_folder / file_path_str
            if sample_path.is_file():
                return sample_path
    return None


def compare_with_sample(predictions: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    problems = []
    if list(predictions.columns) != list(sample.columns):
        problems.append(
            f"The columns differ from the sample submission: expected {list(sample.columns)}, "
            f"got {list(predictions.columns)}."
        )
        return problems
    if len(predictions) != len(sample):
        problems.append(
            f"The row count differs from the sample submission: expected {len(sample)}, "
            f"got {len(predictions)}."
        )

    id_column = sample.columns[0]
    missing_ids = set(sample[id_column]) - set(predictions[id_column])
    extra_ids = set(predictions[id_column]) - set(sample[id_column])
    if missing_ids or extra_ids:
        problems.append(
            f"The ids do not match the sample submission ({len(missing_ids)} missing, "
            f"{len(extra_ids)} extra)."
        )

    for column in sample.columns[1:]:
        sample_is_number = pd.api.types.is_numeric_dtype(sample[column])
        predictions_are_numbers = pd.api.types.is_numeric_dtype(predictions[column])
        if sample_is_number and not predictions_are_numbers:
            problems.append(f"The column {column} should be numeric like the sample submission.")
    return problems


def check_csv_file(predictions_path: Path, brief: Any, project_folder: Path) -> list[str]:
    try:
        predictions = pd.read_csv(predictions_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        return [f"The predictions file cannot be read as CSV: {error}"]
    if predictions.empty:
        return ["The predictions file has no rows."]

    problems = []
    sample_path = find_sample_submission(brief, project_folder)
    if sample_path is not None:
        problems.extend(compare_with_sample(predictions, pd.read_csv(sample_path)))

    columns_with_missing_values = [
        column for column in predictions.columns if predictions[column].isna().any()
    ]
    if columns_with_missing_values:
        problems.append(f"Missing values in the columns: {columns_with_missing_values}.")
    return problems


def check_predictions_file(predictions_path: Path, brief: Any, project_folder: Path) -> list[str]:
    """Return the problems found. An empty list means the file passed every check."""
    if not predictions_path.exists():
        return [f"The predictions file {predictions_path} does not exist."]
    if predictions_path.is_dir():
        if not any(predictions_path.iterdir()):
            return [f"The predictions folder {predictions_path} is empty."]
        return []
    if predictions_path.stat().st_size == 0:
        return ["The predictions file is empty."]
    if predictions_path.suffix.lower() == ".csv":
        return check_csv_file(predictions_path, brief, project_folder)
    return []
