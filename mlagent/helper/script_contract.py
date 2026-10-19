"""The contract between the agents and the scripts the Coder writes.

Texts here go into Coder tasks. check_splits_file is the code check for the splits file.
"""

from __future__ import annotations

import json
from pathlib import Path

SPLITS_FORMAT_TEXT = """\
The splits file is one JSON object with these keys:
- "scheme": the scheme name
- "seed": integer
- "n_splits": number of validation splits (1 or more)
- "id_description": what an example id means. For a table: the row position (0-indexed) in train. For files: path relative to project folder
- "example_ids": list with one id per training example that has a label
- "fold_of_example": list of integers (0 to n_splits-1), or -1 for holdout"""

VALIDATION_PREDICTIONS_FORMAT_TEXT = """\
Validation predictions file format:
- Saved as val_predictions.csv in the experiment folder.
- Columns:
  * "example_id": integer row index in training data matching splits.json
  * "fold": split index (0 to n_splits-1), or -1 for holdout
  * "prediction": the predicted value or class label
  * "proba_<class>": (optional for classification) predicted probability for each class
IMPORTANT: Validation rows are grouped by fold. Use `from mlagent_tools import load_val_preds_with_labels` to align."""

MODEL_RESULT_KEYS_TEXT = """\
- "validation_scores": list of numbers, primary metric on each validation split in split order
- "holdout_score": number or null from model fitted on all non-holdout examples
- "extra_scores": object mapping each extra metric name to its mean over splits
- "validation_predictions_path": "val_predictions.csv"
- "test_predictions_path": "test_predictions.csv"
- "model_path": path of saved model file inside the experiment folder
- "notes": short text with seconds per split and observations"""

MODEL_SCRIPT_RULES_TEXT = """\
1. Read data from the project folder and splits from splits.json.
2. Reuse input preparation from the features file (fit_preparation on train fold, apply_preparation on train and val).
3. Use given random seed everywhere. Learn parameters strictly inside each split from train fold.
4. Save inside the experiment folder: test_predictions.csv, val_predictions.csv, and model file.
5. Predict test data in deliverable format (same row order and ID column as test.csv).
6. Time management: keep training bounded and print short progress lines after each split.
7. NaN/Inf Safety: NEVER call np.isinf(df) or np.isnan(df) directly on DataFrames with string/object columns (raises TypeError). Use df.fillna(0.0) or select numeric columns first.
8. GBDT Early Stopping: When using early stopping in LightGBM/XGBoost during CV, you MUST pass eval_set=[(X_va, y_va)] to .fit(). When training the final model on full train data, do NOT include early stopping callbacks without an eval set."""


def check_splits_file(splits_path: Path) -> list[str]:
    """Return the problems found in a splits file. An empty list means it is usable."""
    if not splits_path.exists():
        return [f"The splits file {splits_path} does not exist."]
    try:
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"The splits file is not valid JSON: {error}"]
    if not isinstance(splits, dict):
        return ["The splits file must hold one JSON object."]

    split_count = splits.get("n_splits")
    if not isinstance(split_count, int) or split_count < 1:
        return ["n_splits must be an integer of 1 or more."]

    if "seeds" in splits:
        seeds = splits["seeds"]
        if not isinstance(seeds, list) or len(seeds) != split_count:
            return ["seeds must be a list with n_splits entries."]
        return []

    example_ids = splits.get("example_ids")
    fold_of_example = splits.get("fold_of_example")
    if not isinstance(example_ids, list) or not isinstance(fold_of_example, list):
        return ["Both example_ids and fold_of_example must be lists (or give seeds)."]
    if not example_ids or len(example_ids) != len(fold_of_example):
        return ["example_ids and fold_of_example must be non-empty and the same length."]
    if not all(isinstance(fold, int) for fold in fold_of_example):
        return ["fold_of_example must hold integers only."]
    valid_folds = set(range(-1, split_count))
    if not set(fold_of_example) <= valid_folds:
        return [f"fold_of_example may only hold -1 or 0 to {split_count - 1}."]
    missing_folds = set(range(split_count)) - set(fold_of_example)
    if missing_folds:
        return [f"These validation splits have no examples: {sorted(missing_folds)}."]
    n_holdout = sum(1 for f in fold_of_example if f == -1)
    if len(fold_of_example) >= 100 and n_holdout > len(fold_of_example) * 0.35:
        return [
            f"Holdout has {n_holdout} examples ({n_holdout / len(fold_of_example):.1%}), which is too large. "
            "Holdout should be ~10-15% of total examples with ~85%+ reserved for cross-validation."
        ]
    return []
