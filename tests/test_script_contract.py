import json

import pytest

from mlagent.script_contract import (
    MODEL_RESULT_KEYS_TEXT,
    MODEL_SCRIPT_RULES_TEXT,
    SPLITS_FORMAT_TEXT,
    check_splits_file,
)


def write_splits(tmp_path, content) -> object:
    splits_path = tmp_path / "splits.json"
    text = content if isinstance(content, str) else json.dumps(content)
    splits_path.write_text(text, encoding="utf-8")
    return splits_path


def valid_splits(**changes) -> dict:
    splits = {
        "scheme": "kfold",
        "seed": 42,
        "n_splits": 2,
        "id_description": "row position in train.csv",
        "example_ids": [0, 1, 2, 3],
        "fold_of_example": [0, 1, 0, 1],
    }
    splits.update(changes)
    return splits


def test_a_valid_splits_file_has_no_problems(tmp_path):
    assert check_splits_file(write_splits(tmp_path, valid_splits())) == []


def test_holdout_examples_marked_minus_one_are_allowed(tmp_path):
    splits = valid_splits(example_ids=[0, 1, 2, 3, 4], fold_of_example=[0, 1, 0, 1, -1])

    assert check_splits_file(write_splits(tmp_path, splits)) == []


def test_a_seeds_file_for_tasks_without_labels_is_allowed(tmp_path):
    splits = {"scheme": "repeated_seeds", "n_splits": 3, "seeds": [1, 2, 3]}

    assert check_splits_file(write_splits(tmp_path, splits)) == []


@pytest.mark.parametrize(
    ("content", "expected_problem"),
    [
        ("{ not json", "not valid JSON"),
        ([1, 2, 3], "one JSON object"),
        (valid_splits(n_splits=0), "n_splits must be an integer"),
        (valid_splits(n_splits="two"), "n_splits must be an integer"),
        ({"n_splits": 2, "seeds": [1]}, "seeds must be a list with n_splits entries"),
        (valid_splits(example_ids=None), "must be lists"),
        (valid_splits(fold_of_example=[0, 1]), "same length"),
        (valid_splits(example_ids=[], fold_of_example=[]), "non-empty"),
        (valid_splits(fold_of_example=[0, 1, 0, "1"]), "integers only"),
        (valid_splits(fold_of_example=[0, 1, 0, 5]), "may only hold -1 or 0 to 1"),
        (valid_splits(fold_of_example=[0, 0, 0, 0]), "no examples: [1]"),
    ],
)
def test_each_kind_of_broken_splits_file_is_named(tmp_path, content, expected_problem):
    problems = check_splits_file(write_splits(tmp_path, content))

    assert len(problems) == 1
    assert expected_problem in problems[0]


def test_a_missing_file_is_reported(tmp_path):
    problems = check_splits_file(tmp_path / "missing.json")

    assert "does not exist" in problems[0]


def test_contract_texts_name_the_keys_the_code_relies_on():
    assert '"fold_of_example"' in SPLITS_FORMAT_TEXT
    assert '"seeds"' in SPLITS_FORMAT_TEXT
    for key in ("validation_scores", "holdout_score", "test_predictions_path", "model_path"):
        assert f'"{key}"' in MODEL_RESULT_KEYS_TEXT
    assert "inside each split" in MODEL_SCRIPT_RULES_TEXT
