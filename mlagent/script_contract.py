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
- "id_description": what an example id means. For a table: the row position, counted from 0, \
in the training file. For a folder of files: the file path relative to the project folder
- "example_ids": list with one id per training example that has a label, in the order of the \
training data
- "fold_of_example": list of integers with the same length: the validation split the example \
belongs to (0 to n_splits-1), or -1 when it is held out for the final holdout check
How a script uses it. For validation split k, the validation part is every example whose fold \
is k, and the training part is every example whose fold is neither k nor -1. The holdout is \
every example with fold -1. It is never used for training or tuning inside a split, and it is \
scored once. For time_split the validation examples are later than the training examples.
For tasks without labels (scheme repeated_seeds) write "seeds" instead of "example_ids" and \
"fold_of_example": a list of n_splits different integer seeds."""

VALIDATION_PREDICTIONS_FORMAT_TEXT = """\
Validation predictions file format:
- One row per validation example in each split, plus one row per holdout example (fold -1).
- Columns:
  * "example_id": integer row index (0-indexed) in the original training data matching splits.json
  * "fold": integer split index (0 to n_splits-1), or -1 for holdout
  * "prediction": the predicted value or class label
  * "proba_<class>": (optional for classification) predicted probability for each class
IMPORTANT: Validation rows are grouped by fold, NOT in 0..N original row order! To align validation \
predictions with training features and true labels, join by example_id with the training dataset \
(or use `from mlagent_tools import load_val_preds_with_labels`). Never zip or assume row-by-row equality."""

MODEL_RESULT_KEYS_TEXT = """\
- "validation_scores": list of numbers, the primary metric on each validation split (or for \
each seed), in split order
- "holdout_score": number from the model fitted on all non-holdout examples, or null when the \
splits hold out nothing
- "extra_scores": object mapping each extra metric name to its mean over the splits
- "validation_predictions_path": path of the saved validation predictions file (format in the \
script rules), or "" if none
- "test_predictions_path": path of the predictions file for the data to predict, in the \
deliverable format
- "model_path": path of the saved final model
- "notes": one short text: the training seconds per split, and any change from the task (for \
example fewer trees or a subsample) with the reason"""

MODEL_SCRIPT_RULES_TEXT = """\
1. Read the data from the project folder and the splits from the splits file (format below). \
Use only the examples that the splits file lists.
2. Reuse the input preparation from the features file (interface below). Import it, do not \
copy it. In every split call fit_preparation on the training part only, then apply_preparation \
on the training part and on the validation part. Do the same for the final fit. If the file \
has no such functions, use its own pattern the same way.
3. Use the given seed everywhere (Python, numpy and the framework). Anything learned from data \
(means, category maps, vocabularies, scalers, target encodings) must be learned inside each \
split from its training part only.
4. Never tune anything on the validation part or the holdout: not early stopping, not the \
number of trees or epochs, not thresholds, not class weights. Use fixed numbers from the \
setup, or an inner split taken from the training part.
5. Train and score every validation split with the primary metric, with the same function \
each time. Give the metric what it needs (labels, probabilities or numbers). When a label is \
needed from probabilities, fix the rule before you look at validation, for example a threshold \
chosen on the training part. Score every extra metric on the same splits and report its mean.
6. When the splits hold out examples (fold -1), fit on all non-holdout examples and score the \
holdout once.
7. After the holdout is scored, fit the final model on all labeled examples. Predict the data \
to predict in the deliverable format: the same row order as the test file, with the id column \
that the deliverable asks for.
8. Save inside the experiment folder: the model, the test predictions, and the validation \
predictions as one CSV with one row per validation example of every split plus one row per \
holdout example. Its columns are example_id (the same ids as in the splits file), fold (the \
split number, or -1 for the holdout), prediction (the label or number, as the metric needs it) \
and, for classification with probabilities, one column proba_<class> per class. Do not put \
the true labels in it.
9. Bound the work. One training, including the final fit, must take well under the time limit \
divided by the number of splits plus 2. Reduce rows, epochs or trees for every split the same \
way when needed, and say so in notes. After each split print one short progress line with its \
score and seconds, so a timeout still shows how far it got.
10. Use the GPU only when the plan says yes, and free its memory between splits."""


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
