"""The Coder: the sub-agent that executes coding tasks.

Writes scripts, runs them, reads errors, and fixes the script until it succeeds or attempts expire.
Supports shell commands (e.g. pip install) and library lookups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlagent.graph.state import Ledger

# Base agent context and data structures
from mlagent.agents.base import AgentContext

# Core execution constants
from mlagent.configs.settings import (
    MAX_ERROR_TEXT_CHARS,
    MAX_LIBRARY_LOOKUPS,
    RANDOM_SEED,
)

# Subprocess and output extraction helpers
from mlagent.helper.code_runner import extract_tagged_json, keep_end_of_text

# Event notification types
from mlagent.helper.events import EventType

# Prompt template
from mlagent.prompts.coder import CODER_PROMPT

AGENT_KEY = "coder"
AGENT_NAME = "Coder"

PYTHON_LANGUAGES = {"python", "py", ""}
SHELL_LANGUAGES = {"bash", "sh", "shell", "powershell", "ps1", "cmd"}
LOOKUP_LANGUAGE = "lookup"

CODE_BLOCK_PATTERN = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
LIBRARY_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
LIBRARY_LOOKUP_MODULE = "mlagent.helper.library_lookup"
OUTPUT_END_CHARS_FOR_MODEL = 1500


def build_coder_digest(ledger: Ledger) -> str:
    # Build clean execution summary for Coder context
    champ = ledger.champion_run()
    task_family = (
        ledger.brief.task_family
        if ledger.brief and ledger.brief.task_family
        else (ledger.scope.task_family if ledger.scope else "unknown")
    )
    primary_metric = (
        ledger.brief.primary_metric
        if ledger.brief and ledger.brief.primary_metric
        else (ledger.scope.primary_metric if ledger.scope else "unknown")
    )
    champ_score = f"{champ.cv_mean:.4f}" if champ and champ.cv_mean is not None else "None"
    champ_model = champ.model_family if champ and champ.model_family else "None"

    lines = [
        f"Task Family: {task_family}",
        f"Primary Metric: {primary_metric}",
        f"Champion Score: {champ_score}",
        f"Champion Model: {champ_model}",
        f"Active Feature Version: {ledger.active_base_feature_version()}",
    ]
    return "\n".join(lines)


def extract_dataset_columns(ledger: Ledger, project_folder: Path) -> str:
    """Extract list of actual dataset columns for Coder prompt context."""
    # 1. Check known tabular files in project data folder first for authentic column names
    if project_folder.exists():
        for candidate in ["train.csv", "train.parquet", "data.csv"]:
            p = project_folder / candidate
            if p.exists():
                try:
                    if p.suffix == ".csv":
                        import pandas as pd

                        df = pd.read_csv(p, nrows=2)
                        return ", ".join(list(df.columns)[:80])
                    elif p.suffix == ".parquet":
                        import pandas as pd

                        df = pd.read_parquet(p)
                        return ", ".join(list(df.columns)[:80])
                except Exception:
                    pass
        for p in project_folder.glob("*.csv"):
            try:
                import pandas as pd

                df = pd.read_csv(p, nrows=2)
                return ", ".join(list(df.columns)[:80])
            except Exception:
                pass

    # 2. Fallback to eda.column_preprocessing_plan if files could not be read
    if ledger.eda and ledger.eda.column_preprocessing_plan:
        cols = [spec.column_name for spec in ledger.eda.column_preprocessing_plan]
        if cols:
            return ", ".join(cols[:80])

    return "See dataset files in project data directory."


@dataclass
class CodeBlock:
    language: str
    code: str


@dataclass
class AttemptOutcome:
    succeeded: bool
    result_data: dict
    stdout: str
    error_text: str
    message_for_model: str


@dataclass
class CoderResult:
    succeeded: bool
    result_data: dict
    script_path: Path
    stdout: str
    error_text: str
    attempts_used: int


def extract_code_blocks(answer_text: str) -> list[CodeBlock]:
    return [
        CodeBlock(language.lower(), code)
        for language, code in CODE_BLOCK_PATTERN.findall(answer_text)
    ]


def collect_lookup_names(code_blocks: list[CodeBlock]) -> list[str]:
    lookup_names = []
    for code_block in code_blocks:
        if code_block.language == LOOKUP_LANGUAGE:
            lookup_names.extend(
                line.strip() for line in code_block.code.splitlines() if line.strip()
            )
    return lookup_names


def last_error_line(error_text: str) -> str:
    non_empty = [line.strip() for line in error_text.splitlines() if line.strip()]
    return non_empty[-1] if non_empty else ""


def first_line(text: str, character_limit: int = 90) -> str:
    lines = text.strip().splitlines()
    return lines[0][:character_limit] if lines else ""


def build_failure(message_for_model: str) -> AttemptOutcome:
    return AttemptOutcome(False, {}, "", message_for_model, message_for_model)


def run_library_lookups(
    lookup_names: list[str], working_folder: Path, agent_context: AgentContext
) -> str:
    lookup_texts = []
    for lookup_name in lookup_names:
        if not LIBRARY_NAME_PATTERN.match(lookup_name):
            lookup_texts.append(f"Lookup '{lookup_name}': not a valid dotted name.")
            continue
        agent_context.event_bus.emit(EventType.CODE_RUN, AGENT_NAME, f"looking up {lookup_name}")
        lookup_result = agent_context.code_runner.run_python_module(
            LIBRARY_LOOKUP_MODULE, [lookup_name], working_folder
        )
        if lookup_result.succeeded:
            lookup_texts.append(f"Lookup '{lookup_name}':\n{lookup_result.stdout.strip()}")
        else:
            lookup_texts.append(
                f"Lookup '{lookup_name}' failed: {last_error_line(lookup_result.error_text)}"
            )
    return "\n\n".join(lookup_texts)


def execute_code_blocks(
    code_blocks: list[CodeBlock],
    script_path: Path,
    result_tag: str,
    working_folder: Path,
    agent_context: AgentContext,
) -> AttemptOutcome:
    python_blocks = [b for b in code_blocks if b.language in PYTHON_LANGUAGES]
    shell_blocks = [b for b in code_blocks if b.language in SHELL_LANGUAGES]

    if len(python_blocks) != 1:
        return build_failure(f"Expected exactly one ```python block, found {len(python_blocks)}.")

    for shell_block in shell_blocks:
        agent_context.event_bus.emit(
            EventType.CODE_RUN, AGENT_NAME, f"shell: {first_line(shell_block.code)}"
        )
        shell_result = agent_context.code_runner.run_shell_command(shell_block.code, working_folder)
        if not shell_result.succeeded:
            return build_failure(
                f"Shell command failed:\nCommand: {shell_block.code.strip()}\nError: {shell_result.error_text}"
            )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(python_blocks[0].code, encoding="utf-8")
    agent_context.event_bus.emit(EventType.CODE_RUN, AGENT_NAME, f"running {script_path.name}")
    script_result = agent_context.code_runner.run_python_script(script_path, working_folder)

    if not script_result.succeeded:
        message = f"The script failed:\n{script_result.error_text}"
        if script_result.timed_out:
            message += "\nExecution timed out. Subsample data or reduce complexity to run faster."
        return AttemptOutcome(False, {}, script_result.stdout, script_result.error_text, message)

    result_data = extract_tagged_json(script_result.stdout, result_tag)
    if result_data is None:
        end_of_output = keep_end_of_text(script_result.stdout, OUTPUT_END_CHARS_FOR_MODEL)
        message = (
            f"Script executed but did not output <{result_tag}>JSON</{result_tag}>.\n"
            f"End of stdout:\n{end_of_output}"
        )
        return AttemptOutcome(False, {}, script_result.stdout, message, message)

    return AttemptOutcome(True, result_data, script_result.stdout, "", "")


def run_coder(
    task_text: str,
    result_tag: str,
    result_keys_text: str,
    working_folder: Path,
    script_name: str,
    caller_key: str,
    ledger: Ledger,
    agent_context: AgentContext,
) -> CoderResult:
    """Implement and execute a coding task."""
    script_path = working_folder / script_name
    caller_name = caller_key.replace("_", " ").title()
    agent_context.event_bus.emit(
        EventType.AGENT_STARTED, AGENT_NAME, f"task from {caller_name}: {first_line(task_text)}"
    )

    history_entries: list[str] = []
    attempts_used = 0
    lookups_left = MAX_LIBRARY_LOOKUPS
    previous_error_line = ""
    outcome = build_failure("The Coder did not run.")
    # Build clean context digest for coder
    digest = build_coder_digest(ledger)

    max_code_attempts = agent_context.config.limits.max_code_attempts
    data_folder = str(agent_context.workspace.project_folder)
    working_folder_str = str(working_folder)
    dataset_columns = extract_dataset_columns(ledger, agent_context.workspace.project_folder)

    while attempts_used < max_code_attempts:
        recent_history = "\n\n".join(history_entries[-2:]) if history_entries else "First attempt."
        prompt = CODER_PROMPT.format(
            random_seed=RANDOM_SEED,
            shell_name=agent_context.environment.shell_name,
            lookups_left=lookups_left,
            result_tag=result_tag,
            result_keys_text=result_keys_text,
            working_folder=working_folder_str,
            script_path=script_path,
            digest=digest,
            caller_name=caller_name,
            task_text=task_text,
            attempt_number=attempts_used + 1,
            max_code_attempts=max_code_attempts,
            history_text=recent_history,
            data_folder=data_folder,
            dataset_columns=dataset_columns,
        )

        answer_text = agent_context.llm_client.ask_text(AGENT_NAME, prompt)
        code_blocks = extract_code_blocks(answer_text)

        lookup_names = collect_lookup_names(code_blocks)
        if (
            lookup_names
            and all(b.language == LOOKUP_LANGUAGE for b in code_blocks)
            and lookups_left > 0
        ):
            names_to_lookup = lookup_names[:lookups_left]
            lookups_left -= len(names_to_lookup)
            history_entries.append(
                run_library_lookups(names_to_lookup, working_folder, agent_context)
            )
            continue

        attempts_used += 1
        outcome = execute_code_blocks(
            code_blocks, script_path, result_tag, working_folder, agent_context
        )
        if outcome.succeeded:
            break

        error_line = last_error_line(outcome.error_text)
        agent_context.event_bus.emit(
            EventType.WARNING,
            AGENT_NAME,
            f"attempt {attempts_used} of {max_code_attempts} failed: {error_line[:120]}",
        )
        history_entries.append(f"Attempt {attempts_used} failed:\n{outcome.message_for_model}")
        if error_line == previous_error_line:
            hint = (
                "CRITICAL WARNING: The script failed with the exact same error as the previous attempt! "
                "You must change your approach:\n"
            )
            err_lower = error_line.lower()
            if (
                "feature names" in err_lower
                or "unseen at fit time" in err_lower
                or "missing" in err_lower
            ):
                hint += (
                    "- Feature name mismatch detected! Call: X_va = X_va.reindex(columns=X_tr.columns, fill_value=0.0) "
                    "and X_test = X_test.reindex(columns=X_full.columns, fill_value=0.0)\n"
                )
            if "early stopping" in err_lower or "validation dataset" in err_lower:
                hint += "- Early stopping error! When using GBDT early stopping, provide eval_set=[(X_va, y_va)] to .fit(), or remove early_stopping.\n"
            if (
                "isinf" in err_lower
                or "isnan" in err_lower
                or "could not convert string" in err_lower
            ):
                hint += (
                    "- Non-numeric columns detected in matrix! Encode categorical columns with pd.get_dummies, "
                    "drop target/id, and call df.fillna(0.0) instead of np.isinf on string columns.\n"
                )
            if "keyerror" in err_lower or "not in index" in err_lower:
                hint += "- Column missing from DataFrame! Only select columns that exist in df.columns: [c for c in cols if c in df.columns].\n"
            history_entries.append(hint)
        previous_error_line = error_line

    if outcome.succeeded:
        agent_context.event_bus.emit(
            EventType.AGENT_FINISHED, AGENT_NAME, f"succeeded after {attempts_used} attempt(s)"
        )
    else:
        agent_context.event_bus.emit(
            EventType.AGENT_FINISHED, AGENT_NAME, f"failed after {attempts_used} attempt(s)"
        )

    return CoderResult(
        succeeded=outcome.succeeded,
        result_data=outcome.result_data,
        script_path=script_path,
        stdout=outcome.stdout,
        error_text=keep_end_of_text(outcome.error_text, MAX_ERROR_TEXT_CHARS),
        attempts_used=attempts_used,
    )
