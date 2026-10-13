"""The contract between the agents and the scripts the Coder writes.

Texts here go into Coder tasks. check_splits_file is the code check for the splits file.
"""

import json
from pathlib import Path

SPLITS_FORMAT_TEXT = """\
The splits file is one JSON object with these keys:
- "scheme": the scheme name
- "seed": integer
- "n_splits": number of validation splits (1 or more)
- "id_description": what an example id means (row position in train.csv, file path relative \
to the project folder, ...)
- "example_ids": list with one id per training example, in a fixed order
- "fold_of_example": list of integers with the same length: the validation split the example \
belongs to (0 to n_splits-1), or -1 when it is held out for the final holdout check
For tasks without labels (scheme repeated_seeds) write "seeds" instead of "example_ids" and \
"fold_of_example": a list of n_splits different integer seeds."""

MODEL_RESULT_KEYS_TEXT = """\
- "validation_scores": list of numbers, the primary metric on each validation split (or for \
each seed), in split order
- "holdout_score": number, or null when the splits hold out nothing
- "extra_scores": object mapping each extra metric name to its mean over the splits
- "validation_predictions_path": path of the saved validation predictions, or "" if none
- "test_predictions_path": path of the predictions file for the data to predict, in the \
deliverable format
- "model_path": path of the saved final model
- "notes": one short text"""

MODEL_SCRIPT_RULES_TEXT = """\
1. Read the data from the project folder and the splits from the splits file (format below).
2. Reuse the input preparation from the features file (source below). Import it, do not copy it.
3. Use the given seed. Anything learned from data (means, category maps, vocabularies, \
scalers, target encodings) must be learned inside each split from its training examples only.
4. Train and score on every validation split with the primary metric. Score every extra \
metric on the same splits and report its mean.
5. When the splits hold out examples (fold -1), fit on all non-holdout examples and score the \
holdout once.
6. Fit the final model on all labeled examples and predict the data to predict, in the \
deliverable format.
7. Save the model, the validation predictions and the test predictions inside the experiment \
folder.
8. Bound the work (epochs, trees, rows) so the script finishes well inside the time limit."""


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
    return []
