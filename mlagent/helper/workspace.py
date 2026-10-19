"""Workspace and file management for ML project runs."""

from __future__ import annotations

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

from mlagent.configs.settings import (
    ARCHIVED_WORKSPACE_PREFIX,
    ARTIFACTS_FOLDER_NAME,
    COMMANDS_LOG_FILE_NAME,
    CONFIG_FILE_NAME,
    ENVIRONMENT_FILE_NAME,
    FEATURES_FOLDER_NAME,
    LEDGER_FILE_NAME,
    LLM_CALLS_LOG_FILE_NAME,
    LOGS_FOLDER_NAME,
    PREDICTIONS_FILE_STEM,
    PROGRESS_FILE_NAME,
    REPORT_FILE_NAME,
    REPORT_FOLDER_NAME,
    RUN_LOG_FILE_NAME,
    SPLITS_FILE_NAME,
    WORKSPACE_FOLDER_NAME,
)
from mlagent.helper.events import AgentEvent


# Error raised when saved run cannot be read
class LedgerLoadError(Exception):
    """Raised when there is no usable saved ledger."""


# Tracks all filesystem paths inside the workspace
class Workspace:
    """Manages files and directories created inside the user project folder."""

    def __init__(self, project_folder: Path) -> None:
        # Resolve project root folder
        self.project_folder = Path(project_folder).resolve()
        # Primary workspace root directory
        self.root = self.project_folder / WORKSPACE_FOLDER_NAME
        # User or default config file path
        self.config_path = self.root / CONFIG_FILE_NAME
        # Logs directory
        self.logs_folder = self.root / LOGS_FOLDER_NAME
        # Artifacts directory for intermediate scripts and data
        self.artifacts_folder = self.root / ARTIFACTS_FOLDER_NAME
        # Report and deliverables folder
        self.report_folder = self.root / REPORT_FOLDER_NAME
        # Features scripts subfolder
        self.features_folder = self.artifacts_folder / FEATURES_FOLDER_NAME
        # Cross-validation splits file
        self.splits_path = self.artifacts_folder / SPLITS_FILE_NAME
        # Final markdown report file
        self.report_path = self.report_folder / REPORT_FILE_NAME
        # Final deliverable predictions file
        self.predictions_path = self.report_folder / f"{PREDICTIONS_FILE_STEM}.csv"
        # Master ledger JSON path
        self.ledger_path = self.root / LEDGER_FILE_NAME
        # Memory and card folders
        self.memory_folder = self.root / "memory"
        self.cards_folder = self.memory_folder / "cards"
        # Previous backup ledger path
        self.previous_ledger_path = self.root / "ledger.previous.json"
        # Environment probe cache path
        self.environment_path = self.root / ENVIRONMENT_FILE_NAME
        # Readable progress report markdown
        self.progress_path = self.root / PROGRESS_FILE_NAME
        # Execution event log
        self.run_log_path = self.logs_folder / RUN_LOG_FILE_NAME
        # Raw LLM calls trace log
        self.llm_calls_log_path = self.logs_folder / LLM_CALLS_LOG_FILE_NAME
        # Shell command execution log
        self.commands_log_path = self.logs_folder / COMMANDS_LOG_FILE_NAME

    # Check if workspace directory already exists
    def exists(self) -> bool:
        return self.root.exists()

    # Check if a valid ledger exists to resume from
    def has_saved_ledger(self) -> bool:
        return self.ledger_path.exists() or self.previous_ledger_path.exists()

    # Create directory tree and write contract tools
    def create_folders(self) -> None:
        from mlagent.helper.contract_tools import CONTRACT_TOOLS_SOURCE

        # Ensure required directories exist
        for folder in (
            self.root,
            self.logs_folder,
            self.artifacts_folder,
            self.report_folder,
            self.features_folder,
            self.memory_folder,
            self.cards_folder,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        # Write mlagent_tools.py helper for code runner scripts
        for tools_path in (
            self.artifacts_folder / "mlagent_tools.py",
            self.root / "mlagent_tools.py",
        ):
            try:
                tools_path.write_text(CONTRACT_TOOLS_SOURCE, encoding="utf-8")
            except OSError:
                pass

    # Archive previous run folder if starting fresh
    def archive_existing(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.project_folder / f"{ARCHIVED_WORKSPACE_PREFIX}{timestamp}"
        counter = 1
        while archive_path.exists():
            archive_path = self.project_folder / f"{ARCHIVED_WORKSPACE_PREFIX}{timestamp}_{counter}"
            counter += 1

        # Garbage collect and attempt atomic rename
        gc.collect()
        try:
            self.root.rename(archive_path)
            return archive_path
        except (PermissionError, OSError):
            pass

        # Fallback to copy and remove on Windows file lock
        shutil.copytree(self.root, archive_path)
        self._remove_directory(self.root)
        return archive_path

    # Robust directory cleanup helper for Windows
    def _remove_directory(self, target_dir: Path) -> None:
        def handle_readonly(func: Any, path_str: str, exc_info: Any) -> None:
            try:
                os.chmod(path_str, stat.S_IWRITE)
                func(path_str)
            except Exception:
                pass

        for _ in range(3):
            try:
                shutil.rmtree(target_dir, onerror=handle_readonly)
                return
            except Exception:
                gc.collect()
                time.sleep(0.1)


# Saves master ledger atomically and syncs node memory cards
def save_ledger(workspace: Workspace, ledger: Any, node_name: str | None = None) -> None:
    # Ensure root workspace folder exists
    workspace.create_folders()
    # Serialize ledger state to JSON string
    serialized_json = ledger.model_dump_json(indent=2)
    # Temporary file path for atomic write
    temp_target = workspace.ledger_path.with_suffix(".tmp")
    # Write to temporary file
    temp_target.write_text(serialized_json, encoding="utf-8")
    # Back up existing ledger if present
    if workspace.ledger_path.exists():
        try:
            shutil.copyfile(workspace.ledger_path, workspace.previous_ledger_path)
        except OSError:
            pass
    # Atomic replace
    temp_target.replace(workspace.ledger_path)

    # Sync node output to dynamic memory store
    try:
        from mlagent.memory.sync import sync_ledger_node_memory

        sync_ledger_node_memory(workspace, ledger, node_name=node_name)
    except Exception:
        pass


# Loads saved ledger state
def load_ledger(workspace: Workspace) -> Any:
    from mlagent.graph.state import Ledger

    # Candidate file locations
    candidates = [workspace.ledger_path, workspace.previous_ledger_path]
    last_err: Exception | None = None

    # Try loading primary or previous ledger
    for path in candidates:
        if path.exists():
            try:
                raw_text = path.read_text(encoding="utf-8")
                return Ledger.model_validate_json(raw_text)
            except (OSError, ValidationError, json.JSONDecodeError) as err:
                last_err = err
                continue

    # Raise error if no ledger could be loaded
    raise LedgerLoadError(f"Could not load saved ledger: {last_err}")


# Logs events cleanly to run.log file
class RunLogWriter:
    """Appends agent events to the workspace run.log file."""

    def __init__(self, log_path: Path) -> None:
        # Destination log file
        self.log_path = log_path

    # Event handler callback
    def handle_event(self, event: AgentEvent) -> None:
        try:
            # Ensure parent folder exists
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Format event timestamp and agent message
            timestamp = event.created_at.strftime("%Y-%m-%d %H:%M:%S")
            line = (
                f"[{timestamp}] [{event.agent_name:<16}] [{event.event_type:<10}] {event.message}\n"
            )
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass


# Append JSON line to log file
def append_json_line(file_path: Path, data: dict[str, Any]) -> None:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    except OSError:
        pass
