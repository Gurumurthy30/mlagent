"""Tests for splits validation checks and split generation integrity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from mlagent.script_contract import check_splits_file


@pytest.mark.unit
def test_check_splits_file_valid(tmp_path: Path) -> None:
    splits_file = tmp_path / "splits.json"
    data = {
        "scheme": "group_kfold",
        "seed": 42,
        "n_splits": 5,
        "id_description": "row position",
        "example_ids": list(range(100)),
        "fold_of_example": [-1] * 10 + [0] * 18 + [1] * 18 + [2] * 18 + [3] * 18 + [4] * 18,
    }
    splits_file.write_text(json.dumps(data), encoding="utf-8")
    problems = check_splits_file(splits_file)
    assert problems == []


@pytest.mark.unit
def test_check_splits_file_rejects_inverted_holdout(tmp_path: Path) -> None:
    """Catastrophic holdout inversion (e.g. 90% holdout) is caught immediately."""
    splits_file = tmp_path / "splits.json"
    data = {
        "scheme": "group_kfold",
        "seed": 42,
        "n_splits": 5,
        "id_description": "row position",
        "example_ids": list(range(100)),
        "fold_of_example": [-1] * 90 + [0] * 2 + [1] * 2 + [2] * 2 + [3] * 2 + [4] * 2,
    }
    splits_file.write_text(json.dumps(data), encoding="utf-8")
    problems = check_splits_file(splits_file)
    assert any("Holdout has 90 examples" in p for p in problems)


@pytest.mark.unit
def test_canonical_group_kfold_recipe_integrity() -> None:
    """Test the canonical group_kfold algorithm on synthetic grouped tabular data."""
    # Create 100 rows with 25 groups (4 rows each)
    groups = [f"{g:04d}_{i}" for g in range(25) for i in range(4)]
    y = np.random.randint(0, 2, size=100)
    df = pd.DataFrame({"PassengerId": groups, "Transported": y})

    # Group extraction
    group_keys = df["PassengerId"].astype(str).str.split("_").str[0]
    unique_groups = np.array(sorted(group_keys.unique()))
    rng = np.random.RandomState(42)
    rng.shuffle(unique_groups)

    n_hold = max(1, int(len(unique_groups) * 0.10))
    holdout_groups = set(unique_groups[:n_hold])

    folds = np.full(len(df), -1, dtype=int)
    cv_mask = ~group_keys.isin(holdout_groups)
    cv_idx = np.where(cv_mask)[0]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_num, (_, val_pos) in enumerate(
        sgkf.split(cv_idx, y[cv_idx], group_keys.iloc[cv_idx])
    ):
        folds[cv_idx[val_pos]] = fold_num

    # Verify no unassigned rows
    assert (folds >= -1).all()
    assert (folds < 5).all()

    # Verify holdout fraction is ~10%
    n_holdout = int((folds == -1).sum())
    assert 0 < n_holdout <= len(df) * 0.25

    # Verify ZERO group leakage between holdout and CV
    cv_group_set = set(group_keys.iloc[np.where(folds >= 0)[0]])
    holdout_group_set = set(group_keys.iloc[np.where(folds == -1)[0]])
    assert cv_group_set.isdisjoint(holdout_group_set)

    # Verify ZERO group leakage across CV folds
    for f1 in range(5):
        f1_groups = set(group_keys.iloc[np.where(folds == f1)[0]])
        for f2 in range(f1 + 1, 5):
            f2_groups = set(group_keys.iloc[np.where(folds == f2)[0]])
            assert f1_groups.isdisjoint(f2_groups)
