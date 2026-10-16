import gc
import json
import os
import shutil
import stat
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mlagent.events import AgentEvent
from mlagent.settings import (
    ARCHIVED_WORKSPACE_PREFIX,
    ARTIFACTS_FOLDER_NAME,
    COMMANDS_LOG_FILE_NAME,
    CONFIG_FILE_NAME,
    ENVIRONMENT_FILE_NAME,
    EVALUATION_FOLDER_NAME,
    FEATURES_FOLDER_NAME,
    IGNORED_FOLDER_NAMES,
    LEDGER_FILE_NAME,
    LLM_CALLS_LOG_FILE_NAME,
    LOGS_FOLDER_NAME,
    MEMORY_FOLDER_NAME,
    PREVIOUS_LEDGER_FILE_NAME,
    PROGRESS_FILE_NAME,
    REPORT_FILE_NAME,
    REPORT_FOLDER_NAME,
    RUN_LOG_FILE_NAME,
    SPLITS_FILE_NAME,
    WORKSPACE_FOLDER_NAME,
)
from mlagent.state import SCHEMA_VERSION, Ledger


class LedgerLoadError(Exception):
    """Raised when there is no usable saved ledger."""


class Workspace:
    """Every folder and file that mlagent creates inside the project folder."""

    def __init__(self, project_folder: Path) -> None:
        self.project_folder = project_folder
        self.root = project_folder / WORKSPACE_FOLDER_NAME
        self.config_path = self.root / CONFIG_FILE_NAME

        self.memory_folder = self.root / MEMORY_FOLDER_NAME
        self.logs_folder = self.root / LOGS_FOLDER_NAME
        self.artifacts_folder = self.root / ARTIFACTS_FOLDER_NAME
        self.report_folder = self.root / REPORT_FOLDER_NAME

        self.features_folder = self.artifacts_folder / FEATURES_FOLDER_NAME
        self.splits_path = self.artifacts_folder / SPLITS_FILE_NAME
        self.report_path = self.report_folder / REPORT_FILE_NAME

        self.ledger_path = self.memory_folder / LEDGER_FILE_NAME
        self.previous_ledger_path = self.memory_folder / PREVIOUS_LEDGER_FILE_NAME
        self.environment_path = self.memory_folder / ENVIRONMENT_FILE_NAME
        self.progress_path = self.memory_folder / PROGRESS_FILE_NAME

        self.run_log_path = self.logs_folder / RUN_LOG_FILE_NAME
        self.llm_calls_log_path = self.logs_folder / LLM_CALLS_LOG_FILE_NAME
        self.commands_log_path = self.logs_folder / COMMANDS_LOG_FILE_NAME

    def exists(self) -> bool:
        return self.root.exists()

    def has_saved_ledger(self) -> bool:
        return self.ledger_path.exists() or self.previous_ledger_path.exists()

    def create_folders(self) -> None:
        from mlagent.contract_tools import CONTRACT_TOOLS_SOURCE

        for folder in (
            self.memory_folder,
            self.logs_folder,
            self.artifacts_folder,
            self.report_folder,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        for tools_path in (
            self.artifacts_folder / "mlagent_tools.py",
            self.root / "mlagent_tools.py",
        ):
            try:
                tools_path.write_text(CONTRACT_TOOLS_SOURCE, encoding="utf-8")
            except OSError:
                pass

    def archive_existing(self) -> Path:
        """Rename or copy the current workspace so a fresh start never deletes old results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.project_folder / f"{ARCHIVED_WORKSPACE_PREFIX}{timestamp}"
        counter = 1
        while archive_path.exists():
            archive_path = self.project_folder / f"{ARCHIVED_WORKSPACE_PREFIX}{timestamp}_{counter}"
            counter += 1

        # Attempt atomic directory rename with brief retries (allows locks to release on Windows)
        for attempt in range(3):
            gc.collect()
            try:
                self.root.rename(archive_path)
                return archive_path
            except (PermissionError, OSError):
                if attempt < 2:
                    time.sleep(0.3)

        # On Windows, if open file handles, Explorer, or virus scanners prevent renaming the directory itself,
        # copy the directory tree to archive_path, then clear the contents of self.root so old results
        # are safely preserved and a clean workspace is ready for the fresh run.
        try:
            shutil.copytree(self.root, archive_path, dirs_exist_ok=True)
        except Exception:
            archive_path.mkdir(parents=True, exist_ok=True)
            for item in self.root.iterdir():
                try:
                    if item.is_dir():
                        shutil.copytree(item, archive_path / item.name, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, archive_path / item.name)
                except Exception:
                    pass

        # Clean out self.root so fresh run starts with an empty workspace
        _clear_directory_contents(self.root)
        return archive_path

    def experiment_folder(self, experiment_id: str) -> Path:
        folder = self.artifacts_folder / experiment_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def evaluation_folder(self, experiment_id: str) -> Path:
        folder = self.artifacts_folder / EVALUATION_FOLDER_NAME / experiment_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def features_file(self, feature_version: int) -> Path:
        return self.features_folder / f"features_v{feature_version}.py"


    def is_ignored_folder(self, folder_name: str) -> bool:
        """True for folders the agents must not explore."""
        return folder_name in IGNORED_FOLDER_NAMES or folder_name.startswith(ARCHIVED_WORKSPACE_PREFIX)


def _handle_remove_readonly(func: Any, path: Any, exc_info: Any) -> None:
    """Handle read-only files on Windows during deletion."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def _clear_directory_contents(directory: Path) -> None:
    """Clear all files and subdirectories inside a directory without deleting the directory itself."""
    if not directory.exists() or not directory.is_dir():
        return
    for item in list(directory.iterdir()):
        try:
            if item.is_dir():
                try:
                    shutil.rmtree(item, on_exc=_handle_remove_readonly)
                except TypeError:
                    shutil.rmtree(item, onerror=_handle_remove_readonly)
            else:
                try:
                    os.chmod(item, stat.S_IWRITE)
                except OSError:
                    pass
                item.unlink(missing_ok=True)
        except OSError:
            pass


def is_ignored_folder_name(folder_name: str) -> bool:
    return folder_name in IGNORED_FOLDER_NAMES or folder_name.startswith(ARCHIVED_WORKSPACE_PREFIX)


def create_new_ledger(user_task: str, user_metrics_text: str) -> Ledger:
    return Ledger(
        user_task=user_task,
        user_metrics_text=user_metrics_text,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def write_text_atomically(file_path: Path, text: str) -> None:
    """Write to a temporary file first, so a crash never leaves a half-written file."""
    temporary_path = file_path.with_name(file_path.name + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, file_path)


def save_ledger(workspace: Workspace, ledger: Ledger) -> None:
    workspace.memory_folder.mkdir(parents=True, exist_ok=True)
    if workspace.ledger_path.exists():
        shutil.copyfile(workspace.ledger_path, workspace.previous_ledger_path)
    write_text_atomically(workspace.ledger_path, ledger.model_dump_json(indent=2))


def load_ledger(workspace: Workspace) -> Ledger:
    """Load the saved ledger. If the latest file is damaged, use the one saved before it."""
    for ledger_path in (workspace.ledger_path, workspace.previous_ledger_path):
        if not ledger_path.exists():
            continue
        try:
            ledger = Ledger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
        except ValidationError:
            continue
        if ledger.schema_version != SCHEMA_VERSION:
            raise LedgerLoadError(
                f"{ledger_path} was saved by a different mlagent version. Choose a fresh start."
            )
        return ledger
    raise LedgerLoadError(f"No usable saved ledger in {workspace.memory_folder}.")


def write_progress_file(workspace: Workspace, progress_text: str) -> None:
    write_text_atomically(workspace.progress_path, progress_text)


def append_json_line(file_path: Path, record: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class RunLogWriter:
    """An event listener that writes every event as one line in logs/run.log."""

    def __init__(self, run_log_path: Path) -> None:
        self.run_log_path = run_log_path

    def handle_event(self, event: AgentEvent) -> None:
        if getattr(event, "event_type", None) == "status" or str(getattr(event, "event_type", "")) == "status":
            return
        self.run_log_path.parent.mkdir(parents=True, exist_ok=True)
        time_text = event.created_at.isoformat(timespec="seconds")
        message = event.message.replace("\n", " ")
        line = f"{time_text} | {event.event_type.value} | {event.agent_name} | {message}\n"
        with self.run_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line)
