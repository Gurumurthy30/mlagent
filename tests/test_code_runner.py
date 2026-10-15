"""Tests for code_runner.py and contract_tools.py: execution, timeouts, crash handling, protocol, and tools."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlagent.code_runner import CodeRunner, extract_tagged_json, find_blocked_text
from mlagent.contract_tools import load_splits, load_val_preds_with_labels
from mlagent.memory import Workspace


@pytest.mark.unit
def test_extract_tagged_json_handles_noise_and_markdown() -> None:
    """extract_tagged_json extracts valid JSON amidst log noise, markdown fences, and text."""
    raw_output = """
    [INFO] Starting model training...
    Training epoch 1/10 - loss: 0.42
    Training epoch 2/10 - loss: 0.35

    <RESULT>
    ```json
    {
        "status": "ok",
        "scores": [0.84, 0.86, 0.85],
        "mean_score": 0.85
    }
    ```
    </RESULT>

    [INFO] Finished training run in 1.2s.
    """
    parsed = extract_tagged_json(raw_output, "RESULT")
    assert parsed["status"] == "ok"
    assert parsed["mean_score"] == 0.85
    assert len(parsed["scores"]) == 3


@pytest.mark.unit
def test_extract_tagged_json_rejects_missing_or_malformed_block() -> None:
    """extract_tagged_json returns None on missing tag or malformed JSON."""
    assert extract_tagged_json("No tags here", "RESULT") is None
    assert extract_tagged_json("<RESULT>{unquoted_key: invalid}</RESULT>", "RESULT") is None
    assert extract_tagged_json("<RESULT></RESULT>", "RESULT") is None


@pytest.mark.unit
def test_blocked_shell_commands() -> None:
    """Dangerous destructive commands are detected by deny-list."""
    assert find_blocked_text("rm -rf /") is not None
    assert find_blocked_text("rmdir /s /q C:\\") is not None
    assert find_blocked_text("shutdown /s /t 0") is not None
    assert find_blocked_text("python script.py") is None


@pytest.mark.unit
def test_successful_script_execution(tmp_path: Path) -> None:
    """CodeRunner executes Python script and captures output."""
    ws = Workspace(tmp_path / "ws")
    ws.create_folders()
    runner = CodeRunner(ws, timeout_seconds=10)

    script_path = tmp_path / "test_success.py"
    script_path.write_text(
        'print("<RESULT>{\\"success\\": true, \\"val\\": 123}</RESULT>")',
        encoding="utf-8",
    )

    result = runner.run_python_script(script_path, working_folder=tmp_path)
    assert result.succeeded
    assert result.exit_code == 0
    assert not result.timed_out
    parsed = extract_tagged_json(result.stdout, "RESULT")
    assert parsed["success"] is True
    assert parsed["val"] == 123


@pytest.mark.unit
def test_crashing_script_captures_error(tmp_path: Path) -> None:
    """CodeRunner captures traceback and exit code from crashing script."""
    ws = Workspace(tmp_path / "ws")
    ws.create_folders()
    runner = CodeRunner(ws, timeout_seconds=10)

    script_path = tmp_path / "test_crash.py"
    script_path.write_text("raise RuntimeError('Intentional script failure')", encoding="utf-8")

    result = runner.run_python_script(script_path, working_folder=tmp_path)
    assert not result.succeeded
    assert result.exit_code != 0
    assert "Intentional script failure" in result.error_text


@pytest.mark.unit
def test_timeout_script_is_killed(tmp_path: Path) -> None:
    """CodeRunner enforces timeout and kills hanging processes."""
    ws = Workspace(tmp_path / "ws")
    ws.create_folders()
    # 1 second timeout
    runner = CodeRunner(ws, timeout_seconds=1)

    script_path = tmp_path / "test_hang.py"
    script_path.write_text("import time\ntime.sleep(10)", encoding="utf-8")

    result = runner.run_python_script(script_path, working_folder=tmp_path)
    assert not result.succeeded
    assert result.timed_out
    assert "timed out" in result.error_text.lower()


@pytest.mark.unit
def test_contract_tools_load_splits_and_preds(tmp_path: Path) -> None:
    """Generated mlagent_tools works on tiny DataFrames and splits."""
    ws = Workspace(tmp_path / "ws")
    ws.create_folders()

    # 1. Test splits.json loading
    splits_data = {
        "train_indices": [[0, 1], [1, 2]],
        "val_indices": [[2], [0]],
    }
    splits_file = tmp_path / "splits.json"
    splits_file.write_text(json.dumps(splits_data), encoding="utf-8")

    loaded_splits = load_splits(splits_file)
    assert loaded_splits == splits_data

    # 2. Test val preds alignment
    train_df = pd.DataFrame({
        "id": [101, 102, 103],
        "feature": [1.0, 2.0, 3.0],
        "target": [0, 1, 0],
    })
    train_csv = tmp_path / "train.csv"
    train_df.to_csv(train_csv, index=False)

    val_preds_df = pd.DataFrame({
        "example_id": [0, 2],  # row 0 and row 2 in train
        "prediction": [0, 0],
    })
    val_preds_csv = tmp_path / "val_predictions.csv"
    val_preds_df.to_csv(val_preds_csv, index=False)

    aligned = load_val_preds_with_labels(val_preds_csv, train_csv, target_column="target")
    assert "true_label" in aligned.columns
    assert list(aligned["true_label"]) == [0, 0]  # target at row 0 and row 2
