"""Runs Python scripts and shell commands for the Coder, with a time limit and a command log.

Scripts run with the same Python as the agent, so they see the libraries installed for it.
There is no sandbox: everything runs as the current user.
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psutil

from mlagent.environment_probe import get_shell_name
from mlagent.memory import Workspace, append_json_line
from mlagent.settings import (
    BLOCKED_SHELL_TEXT,
    LOGGED_OUTPUT_CHARS,
    MAX_COMMAND_OUTPUT_CHARS,
    MAX_ERROR_TEXT_CHARS,
)

POWERSHELL_UTF8_PREFIX = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
SECONDS_TO_COLLECT_OUTPUT_AFTER_KILL = 10


@dataclass
class CommandResult:
    command_text: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float
    timeout_seconds: float

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def error_text(self) -> str:
        """A short description of what went wrong. Empty when the command succeeded."""
        if self.timed_out:
            return f"Timed out after {self.timeout_seconds:.0f} seconds (the limit for one run)."
        if self.succeeded:
            return ""
        return keep_end_of_text(self.stderr or self.stdout, MAX_ERROR_TEXT_CHARS)


def keep_end_of_text(text: str, character_limit: int) -> str:
    """Keep the last characters, because errors and results are printed at the end."""
    return text[-character_limit:] if len(text) > character_limit else text


def extract_tagged_json(output_text: str, tag_name: str) -> dict | None:
    """Find the last <TAG>{...}</TAG> block in a script's output and return it as a dict."""
    escaped_tag = re.escape(tag_name)
    matches = re.findall(rf"<{escaped_tag}>(.*?)</{escaped_tag}>", output_text, re.DOTALL)
    if not matches:
        return None
    try:
        parsed_value = json.loads(matches[-1])
    except json.JSONDecodeError:
        return None
    return parsed_value if isinstance(parsed_value, dict) else None


def find_blocked_text(shell_text: str) -> str | None:
    lowered_text = shell_text.lower()
    for blocked_text in BLOCKED_SHELL_TEXT:
        if blocked_text in lowered_text:
            return blocked_text
    return None


def build_shell_command(shell_text: str) -> list[str]:
    if get_shell_name() == "powershell":
        full_text = POWERSHELL_UTF8_PREFIX + shell_text
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", full_text]
    return ["bash", "-c", shell_text]


def build_child_environment(extra_paths: list[str | Path] | None = None) -> dict[str, str]:
    """Force UTF-8 so printed text reads the same on Windows and Linux. No .pyc clutter."""
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if extra_paths:
        existing_pp = os.environ.get("PYTHONPATH", "")
        paths_str = [str(p) for p in extra_paths if p]
        if existing_pp:
            paths_str.append(existing_pp)
        env["PYTHONPATH"] = os.pathsep.join(paths_str)
    return env


def kill_process_tree(process: subprocess.Popen) -> None:
    """Kill the process and everything it started, so no training keeps running."""
    try:
        parent_process = psutil.Process(process.pid)
        child_processes = parent_process.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child_process in child_processes:
        try:
            child_process.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        parent_process.kill()
    except psutil.NoSuchProcess:
        pass


class CodeRunner:
    def __init__(self, workspace: Workspace, timeout_seconds: float) -> None:
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds

    def run_python_script(self, script_path: Path, working_folder: Path) -> CommandResult:
        command = [sys.executable, "-u", str(script_path)]
        return self._run(command, f"python {script_path}", "python", working_folder)

    def run_python_module(
        self, module_name: str, arguments: list[str], working_folder: Path
    ) -> CommandResult:
        command = [sys.executable, "-u", "-m", module_name, *arguments]
        command_text = f"python -m {module_name} {' '.join(arguments)}"
        return self._run(command, command_text, "python", working_folder)

    def run_shell_command(self, shell_text: str, working_folder: Path) -> CommandResult:
        blocked_text = find_blocked_text(shell_text)
        if blocked_text:
            result = CommandResult(
                command_text=shell_text,
                exit_code=-1,
                stdout="",
                stderr=f"Blocked: the command contains '{blocked_text}'. Use a safer command.",
                timed_out=False,
                seconds=0.0,
                timeout_seconds=self.timeout_seconds,
            )
            self._write_log("shell", working_folder, result)
            return result
        return self._run(build_shell_command(shell_text), shell_text, "shell", working_folder)

    def _run(
        self, command: list[str], command_text: str, kind: str, working_folder: Path
    ) -> CommandResult:
        working_folder.mkdir(parents=True, exist_ok=True)
        started_at = time.monotonic()
        timed_out = False
        try:
            process = subprocess.Popen(
                command,
                cwd=working_folder,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=build_child_environment(
                    [
                        self.workspace.artifacts_folder,
                        self.workspace.root,
                        self.workspace.project_folder,
                    ]
                ),
            )
        except OSError as error:
            result = self._build_result(command_text, -1, "", str(error), False, started_at)
            self._write_log(kind, working_folder, result)
            return result

        try:
            stdout_text, stderr_text = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_tree(process)
            try:
                stdout_text, stderr_text = process.communicate(
                    timeout=SECONDS_TO_COLLECT_OUTPUT_AFTER_KILL
                )
            except subprocess.TimeoutExpired:
                stdout_text, stderr_text = "", ""

        result = self._build_result(
            command_text, process.returncode, stdout_text, stderr_text, timed_out, started_at
        )
        self._write_log(kind, working_folder, result)
        return result

    def _build_result(
        self,
        command_text: str,
        exit_code: int,
        stdout_text: str,
        stderr_text: str,
        timed_out: bool,
        started_at: float,
    ) -> CommandResult:
        return CommandResult(
            command_text=command_text,
            exit_code=exit_code,
            stdout=keep_end_of_text(stdout_text, MAX_COMMAND_OUTPUT_CHARS),
            stderr=keep_end_of_text(stderr_text, MAX_COMMAND_OUTPUT_CHARS),
            timed_out=timed_out,
            seconds=round(time.monotonic() - started_at, 1),
            timeout_seconds=self.timeout_seconds,
        )

    def _write_log(self, kind: str, working_folder: Path, result: CommandResult) -> None:
        append_json_line(
            self.workspace.commands_log_path,
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "kind": kind,
                "command": result.command_text,
                "working_folder": str(working_folder),
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "seconds": result.seconds,
                "stdout_end": keep_end_of_text(result.stdout, LOGGED_OUTPUT_CHARS),
                "stderr_end": keep_end_of_text(result.stderr, LOGGED_OUTPUT_CHARS),
            },
        )
