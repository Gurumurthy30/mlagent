import json

import pytest

from mlagent.events import EventBus, EventType
from mlagent.memory import (
    LedgerLoadError,
    RunLogWriter,
    append_json_line,
    create_new_ledger,
    is_ignored_folder_name,
    load_ledger,
    save_ledger,
)
from mlagent.state import ExperimentRun


def test_create_folders_makes_the_four_folders(workspace):
    root_content = sorted(path.name for path in workspace.root.iterdir())

    assert root_content == ["artifacts", "logs", "memory", "report"]


def test_save_and_load_ledger_round_trip(workspace):
    ledger = create_new_ledger("predict price", "RMSE")
    ledger.runs.append(ExperimentRun(experiment_id="experiment_001", score_mean=1.5))

    save_ledger(workspace, ledger)

    assert load_ledger(workspace) == ledger
    assert ledger.created_at != ""


def test_saving_twice_keeps_the_previous_version(workspace):
    ledger = create_new_ledger("task", "metric")
    save_ledger(workspace, ledger)
    ledger.iteration = 1
    save_ledger(workspace, ledger)

    previous_text = workspace.previous_ledger_path.read_text(encoding="utf-8")

    assert json.loads(previous_text)["iteration"] == 0
    assert load_ledger(workspace).iteration == 1


def test_no_temporary_file_is_left_after_saving(workspace):
    save_ledger(workspace, create_new_ledger("task", "metric"))

    file_names = [path.name for path in workspace.memory_folder.iterdir()]

    assert not any(name.endswith(".tmp") for name in file_names)


def test_damaged_latest_ledger_falls_back_to_the_previous_one(workspace):
    ledger = create_new_ledger("task", "metric")
    save_ledger(workspace, ledger)
    ledger.iteration = 5
    save_ledger(workspace, ledger)
    workspace.ledger_path.write_text("{ this is not valid json", encoding="utf-8")

    assert load_ledger(workspace).iteration == 0


def test_loading_fails_clearly_when_nothing_is_saved(workspace):
    with pytest.raises(LedgerLoadError):
        load_ledger(workspace)


def test_loading_fails_clearly_when_every_file_is_damaged(workspace):
    workspace.ledger_path.write_text("damaged", encoding="utf-8")
    workspace.previous_ledger_path.write_text("also damaged", encoding="utf-8")

    with pytest.raises(LedgerLoadError):
        load_ledger(workspace)


def test_ledger_from_another_schema_version_is_refused(workspace):
    ledger = create_new_ledger("task", "metric")
    ledger.schema_version = 999
    save_ledger(workspace, ledger)

    with pytest.raises(LedgerLoadError, match="different mlagent version"):
        load_ledger(workspace)


def test_has_saved_ledger(workspace):
    assert not workspace.has_saved_ledger()

    save_ledger(workspace, create_new_ledger("task", "metric"))

    assert workspace.has_saved_ledger()


def test_archive_existing_renames_the_workspace_and_keeps_the_files(workspace, tmp_path):
    save_ledger(workspace, create_new_ledger("task", "metric"))

    archive_path = workspace.archive_existing()

    assert not workspace.exists()
    assert archive_path.name.startswith("mlagent_workspace_old_")
    assert (archive_path / "memory" / "ledger.json").exists()


def test_old_archives_and_own_workspace_are_ignored_folders():
    assert is_ignored_folder_name("mlagent_workspace")
    assert is_ignored_folder_name("mlagent_workspace_old_20261011_120000")
    assert is_ignored_folder_name(".git")
    assert not is_ignored_folder_name("data")


def test_experiment_folder_is_created_inside_artifacts(workspace):
    folder = workspace.experiment_folder("experiment_001")

    assert folder.exists()
    assert folder.parent == workspace.artifacts_folder


def test_append_json_line_writes_one_json_object_per_line(workspace):
    append_json_line(workspace.commands_log_path, {"command": "pip list", "exit_code": 0})
    append_json_line(workspace.commands_log_path, {"command": "dir", "note": "caf\u00e9"})

    lines = workspace.commands_log_path.read_text(encoding="utf-8").splitlines()

    assert json.loads(lines[0])["exit_code"] == 0
    assert json.loads(lines[1])["note"] == "caf\u00e9"


def test_run_log_writer_writes_one_line_per_event(workspace):
    event_bus = EventBus()
    event_bus.subscribe(RunLogWriter(workspace.run_log_path).handle_event)

    event_bus.emit(EventType.AGENT_STARTED, "Data Analyst", "EDA round 1")
    event_bus.emit(EventType.WARNING, "Coder", "line one\nline two")

    lines = workspace.run_log_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert "agent_started | Data Analyst | EDA round 1" in lines[0]
    assert "line one line two" in lines[1]


def test_workspace_knows_the_paths_of_splits_features_and_evaluation_folders(workspace):
    assert workspace.splits_path == workspace.artifacts_folder / "splits.json"
    assert workspace.features_file(3) == workspace.artifacts_folder / "features" / "features_v3.py"

    evaluation_folder = workspace.evaluation_folder("experiment_004")

    assert evaluation_folder.exists()
    assert evaluation_folder == workspace.artifacts_folder / "evaluation" / "experiment_004"
