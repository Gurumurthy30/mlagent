"""The Coder: the sub-agent that executes coding tasks.

Writes scripts, runs them, reads errors, and fixes the script until it succeeds or attempts expire.
Supports shell commands (e.g. pip install) and library lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from mlagent.agents.base import AgentContext
from mlagent.code_runner import extract_tagged_json, keep_end_of_text
from mlagent.digest import build_digest
from mlagent.events import EventType
from mlagent.settings import (
    MAX_CODE_ATTEMPTS,
    MAX_ERROR_TEXT_CHARS,
    MAX_LIBRARY_LOOKUPS,
    RANDOM_SEED,
)
from mlagent.state import Ledger

AGENT_KEY = "coder"
AGENT_NAME = "Coder"

PYTHON_LANGUAGES = {"python", "py", ""}
SHELL_LANGUAGES = {"bash", "sh", "shell", "powershell", "ps1", "cmd"}
LOOKUP_LANGUAGE = "lookup"

CODE_BLOCK_PATTERN = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
LIBRARY_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
LIBRARY_LOOKUP_MODULE = "mlagent.library_lookup"
OUTPUT_END_CHARS_FOR_MODEL = 1500

CODER_PROMPT = """\
You are the Coder, an expert ML engineer implementing scripts for the team.
You receive a task and write ONE complete Python script. You cannot see the screen or ask questions.

HOW TO ANSWER:
1. Exactly one ```python block with the complete script.
   Starts with imports, runs under if __name__ == "__main__":, uses random seed {random_seed}.
2. Optional ```{shell_name} block with shell commands (e.g. python -m pip install) if a library is missing.
3. Optional ```lookup block with dotted names (e.g. lightgbm.LGBMClassifier) to inspect documentation ({lookups_left} left).

SCRIPT RULES:
- Clean, readable code. Output predictions must be written inside working folder: val_predictions.csv and test_predictions.csv.
- Model files must be saved inside working folder as model.<ext>.
- Helpers: `from mlagent_tools import load_splits, load_val_preds_with_labels` is available.
- Bound work: print short progress lines after steps with seconds elapsed.
- Final output: print exactly one line at the end:
  <{result_tag}>JSON</{result_tag}>
  containing:
{result_keys_text}

WHERE THINGS ARE:
- Working folder (save outputs here): {working_folder}
- Script will be saved as: {script_path}

SITUATION & BUDGET:
{digest}

TASK FROM {caller_name}:
{task_text}

ATTEMPT {attempt_number} OF {max_code_attempts}.
{history_text}"""


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
        return build_failure(
            f"Expected exactly one ```python block, found {len(python_blocks)}."
        )

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
    digest = build_digest(AGENT_KEY, ledger)

    while attempts_used < MAX_CODE_ATTEMPTS:
        prompt = CODER_PROMPT.format(
            random_seed=RANDOM_SEED,
            shell_name=agent_context.environment.shell_name,
            lookups_left=lookups_left,
            result_tag=result_tag,
            result_keys_text=result_keys_text,
            working_folder=working_folder,
            script_path=script_path,
            digest=digest,
            caller_name=caller_name,
            task_text=task_text,
            attempt_number=attempts_used + 1,
            max_code_attempts=MAX_CODE_ATTEMPTS,
            history_text="\n\n".join(history_entries) or "First attempt.",
        )

        answer_text = agent_context.llm_client.ask_text(AGENT_NAME, prompt)
        code_blocks = extract_code_blocks(answer_text)

        lookup_names = collect_lookup_names(code_blocks)
        if lookup_names and all(b.language == LOOKUP_LANGUAGE for b in code_blocks) and lookups_left > 0:
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
            f"attempt {attempts_used} of {MAX_CODE_ATTEMPTS} failed: {error_line[:120]}",
        )
        history_entries.append(f"Attempt {attempts_used} failed:\n{outcome.message_for_model}")
        if error_line == previous_error_line:
            agent_context.event_bus.emit(
                EventType.WARNING, AGENT_NAME, "identical error repeated, stopping"
            )
            break
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
