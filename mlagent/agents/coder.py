"""The Coder: the sub-agent that executes coding tasks.

Writes scripts, runs them, reads errors, and fixes the script until it succeeds or attempts expire.
Supports shell commands (e.g. pip install) and library lookups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mlagent.agents.base import AgentContext
from mlagent.code_runner import extract_tagged_json, keep_end_of_text
from mlagent.digest import build_digest
from mlagent.events import EventType
from mlagent.settings import (
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
You are the Coder, a principal competitive ML engineer implementing production-grade Python scripts \
for the autonomous ML team. You receive a task and write ONE complete, self-contained Python script.

RESPONSE FORMAT:
1. Optional ```{shell_name} block: shell commands (e.g. `python -m pip install <pkg>`) ONLY if a library is missing.
2. Optional ```lookup block: dotted names to inspect docstrings ({lookups_left} lookups left).
3. Exactly ONE ```python block containing the complete runnable script:
```python
import os, sys, json, time, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
# from mlagent_tools import load_splits, load_val_preds_with_labels

def main():
    start_time = time.time()
    np.random.seed({random_seed})
    random.seed({random_seed})

    # Execute task with fold-isolation and pre-flight assertions...

    result = {{"key": "value"}}  # convert all numpy scalars to native float/int!
    print(f"<{result_tag}>{{json.dumps(result)}}</{result_tag}>")

if __name__ == "__main__":
    main()
```

CRASH-FREE COMPETITIVE ML CODING IDIOMS (follow strictly):
1. VERSION-SAFE GBDT APIS (do not pass deprecated `verbose` or `early_stopping_rounds` kwargs to `.fit()` in LightGBM):
   - LightGBM: `model = lgb.LGBMClassifier(**params, random_state={random_seed}, verbosity=-1, n_jobs=-1)` and `model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])`.
   - CatBoost: `model = cb.CatBoostClassifier(**params, random_seed={random_seed}, verbose=0, allow_writing_files=False)` and `model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)`.
   - XGBoost: `model = xgb.XGBClassifier(**params, random_state={random_seed}, early_stopping_rounds=50, n_jobs=-1, verbosity=0)` and `model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)`.
2. SAFE MATH, DATAFRAME INDEXING & FOLD ISOLATION:
   - Always call `.reset_index(drop=True)` when slicing fold DataFrames (`train_fold`, `val_fold`) before assigning derived columns or concatenating arrays, while keeping `example_id` aligned with `splits.json`.
   - Guard all divisions with `np.where(np.abs(denom) > 1e-9, num / denom, 0.0)` and sanitize matrices with `df.replace([np.inf, -np.inf], np.nan).fillna(0.0)`.
   - Never fit encoders, imputers, target encodings, or `fit_preparation` on full data before splitting; always fit on `train_fold` and apply to `val_fold` and `test`.
3. PRE-FLIGHT SELF-VERIFICATION ASSERTIONS (run before printing `<{result_tag}>`):
   - Cast all numbers in `result` dict to native Python `float(x)` / `int(x)` (never `np.float32`, `np.int64`, `NaN`, or `Inf`).
   - When writing `val_predictions.csv` and `test_predictions.csv` in `{working_folder}`, assert both files exist on disk, contain zero `NaN`s, and match the required column names and row counts.
4. DEBUGGING ON ATTEMPT > 1:
   - If `ATTEMPT > 1`, inspect the exact traceback in `history_text`, fix the root cause (e.g., column name, shape, dtype, API call) without stripping away the intended ML logic, and output the complete repaired script.
   - Read-only project folder: never write inside the project folder; save all files inside `{working_folder}`.
   - The very last stdout line MUST be `<{result_tag}>JSON</{result_tag}>` with keys:
{result_keys_text}

WHERE THINGS ARE:
- Working folder: {working_folder}
- Script path: {script_path}

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

    max_code_attempts = agent_context.config.limits.max_code_attempts

    while attempts_used < max_code_attempts:
        recent_history = "\n\n".join(history_entries[-2:]) if history_entries else "First attempt."
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
            max_code_attempts=max_code_attempts,
            history_text=recent_history,
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
            f"attempt {attempts_used} of {max_code_attempts} failed: {error_line[:120]}",
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