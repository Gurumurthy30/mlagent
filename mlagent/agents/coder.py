"""The Coder: the sub-agent that every other agent hands its coding tasks to.

It writes a script, runs it, reads the error, and fixes the script until it works or the
attempts run out. It can also run shell commands (for example pip install) and look up
library documentation on this machine.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from mlagent.agents.base import (
    AgentContext,
    build_common_prompt_values,
    display_name,
    render_prompt,
)
from mlagent.code_runner import extract_tagged_json, keep_end_of_text
from mlagent.events import EventType
from mlagent.settings import (
    MAX_CODE_ATTEMPTS,
    MAX_ERROR_TEXT_CHARS,
    MAX_LIBRARY_LOOKUPS,
    RANDOM_SEED,
)
from mlagent.state import Ledger

AGENT_KEY = "coder"
AGENT_NAME = display_name(AGENT_KEY)

PYTHON_LANGUAGES = {"python", "py", ""}
SHELL_LANGUAGES = {"bash", "sh", "shell", "powershell", "ps1", "cmd"}
LOOKUP_LANGUAGE = "lookup"

CODE_BLOCK_PATTERN = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
LIBRARY_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")

LIBRARY_LOOKUP_MODULE = "mlagent.library_lookup"
OUTPUT_END_CHARS_FOR_MODEL = 1500

CODER_PROMPT = """\
You are the Coder, a senior machine learning engineer. Other agents give you a coding task. \
You write the script, it is run for you, and you fix it until it works. You cannot see the \
screen or ask questions: the task, the team notes and the run output are all you get. Where \
the team notes and the task disagree, follow the task. The notes are background, not orders.

HOW TO ANSWER: code blocks, plus at most two short sentences. No long explanations.
1. Exactly one ```python block with the complete script, never a fragment and never a diff. \
It starts with its imports, puts its work under if __name__ == "__main__":, uses the random \
seed {random_seed} (set it for every library that uses randomness), and finishes well inside \
the time limit in the rules above. Never write three backticks inside the script, and never \
add a second python block, not even as an example.
2. Before it you may add ```{shell_name} blocks with shell commands, one command per line, \
for example python -m pip install for a library that is really missing. Use them only when an \
import failed or the task needs a library that is not installed. Install only what is \
missing. Never delete files, never use sudo or admin rights, and never touch anything \
outside the workspace folder.
3. To read a library's documentation before you write code, answer with only a ```lookup \
block with one dotted name per line, for example lightgbm.LGBMClassifier.fit. Use it when you \
are not sure of a function name, an argument or a version difference. You have {lookups_left} \
lookups left. A lookup does not use an attempt.

The shape of the script (the shape only, the content comes from the task):
```python
import json
import random
from pathlib import Path

import numpy as np

SEED = {random_seed}
WORKSPACE = Path(...)  # the workspace folder given below


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    # read the data, do the work, print one short progress line with seconds per major step
    # write every output file inside WORKSPACE
    result_object = dict(...)  # the keys of the result line, real values from this script
    print("<{result_tag}>" + json.dumps(result_object, allow_nan=False) + "</{result_tag}>")


if __name__ == "__main__":
    main()
```

HOW TO WRITE THE SCRIPT:
- Simple, readable code with clear descriptive names. No clever one-liners, no divider \
comments. Do exactly the task and add no extras.
- Estimate the work before you start and keep it inside the time limit: subsample big data, \
use a fast model, few epochs or a bounded number of trials when the task allows. Never write \
a loop without a clear end. Never fully load a file that is too big for the machine memory: \
read chunks, a sample or counts, and say so in the result.
- Print one short progress line after each major step with its seconds, so a timeout still \
shows how far the script got. Never print whole tables or arrays.
- No input(), no windows or plots that wait for a person. If a plot is needed, use the Agg \
backend and save it to a file. On Windows keep worker processes at zero or one unless the task \
needs more.
- Build paths with pathlib, open text files with encoding="utf-8", and write code that runs on \
Windows, Linux and macOS. Write every output file only inside the workspace folder. Create \
the folders you need.
- Do not use try and except to hide a failure. If something important cannot be computed, \
let the script fail with a clear error message. Never print made-up, guessed or default \
numbers as results.
- Shared helpers: mlagent_tools.py is available (from mlagent_tools import load_splits, \
load_val_preds_with_labels). Use load_splits to load splits.json, and load_val_preds_with_labels \
to accurately align validation predictions with ground truth without guessing column schemas.

WHEN A RUN FAILED: read the error first, find the cause, and send a complete new script that \
fixes that cause.
- ModuleNotFoundError or ImportError: a shell block that installs only that library.
- AttributeError, TypeError or an unexpected keyword argument: a lookup of that function, \
then use the signature it shows. A version difference is the usual cause.
- KeyError or a ValueError about shapes or types: look at the data before the failing line. \
Print the column names, types or shapes in one short line, then fix the code. Do not wrap the \
line in try.
- MemoryError, a killed process or out of memory (also on the GPU): a smaller sample, chunks \
or a smaller batch. Say so in the result.
- A timeout, or no result line: use the progress lines to find the slow step, then cut that \
step's work by half or more (rows, trees, epochs, trials). Do not just run it again.
- "Out of range float values" from json: a score is NaN or infinite. Find where it comes \
from. Never replace it with 0.
- FileNotFoundError or PermissionError: use the full paths given. Outputs go only into the \
workspace folder.
- The same error as the last attempt: your fix did not reach the cause. Change your approach.
Never change the task, drop a required key or weaken the result to get a pass.

BEFORE YOU SEND, CHECK: one python block and it is complete, the work sits under the main \
guard, the seed is set, every output goes into the workspace folder, the result line is \
printed last and only once, and it holds every key with a real value.

{run_rules}

WHERE THINGS ARE:
- Project folder (read only, never write or delete anything in it): {project_path}
- Workspace folder (all outputs go here): {workspace_path}
- Your script will be saved as: {script_path}
- Your script runs with this working folder: {working_folder}
Use the full paths given in the task and above.

WHAT THE TEAM KNOWS SO FAR:
{digest}

TASK FROM THE {caller_name}:
{task_text}

THE RESULT LINE: the last thing the script prints must be one single line, on its own line, \
printed after everything else:
<{result_tag}>JSON</{result_tag}>
where JSON is one object with these keys:
{result_keys_text}
The values must be real results computed by this script, with plain Python types: convert \
numpy and pandas values with float(), int() or .tolist() first. Print the line like this, \
with allow_nan=False so a missing score fails loudly instead of being sent on:
print("<{result_tag}>" + json.dumps(result_object, allow_nan=False) + "</{result_tag}>")
Print that tag only once and nowhere else.

ATTEMPT {attempt_number} OF {max_code_attempts}.
{history_text}
If the history above shows an earlier attempt, do not send the same code again. Now write \
your answer."""


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
    non_empty_lines = [line.strip() for line in error_text.splitlines() if line.strip()]
    return non_empty_lines[-1] if non_empty_lines else ""


def first_line(text: str, character_limit: int = 90) -> str:
    lines = text.strip().splitlines()
    return lines[0][:character_limit] if lines else ""


def build_failure(message_for_model: str) -> AttemptOutcome:
    return AttemptOutcome(False, {}, "", message_for_model, message_for_model)


def run_library_lookups(
    lookup_names: list[str], working_folder: Path, agent_context: AgentContext
) -> str:
    """Look up each name in a separate process and return the text to show the model."""
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
    """Run the shell blocks, then the one python block, and check that it printed its result."""
    python_blocks = [block for block in code_blocks if block.language in PYTHON_LANGUAGES]
    shell_blocks = [block for block in code_blocks if block.language in SHELL_LANGUAGES]
    if len(python_blocks) != 1:
        lookup_note = " No lookups are left." if collect_lookup_names(code_blocks) else ""
        return build_failure(
            f"Your answer had {len(python_blocks)} python blocks. "
            f"Reply with exactly one ```python block that holds the complete script.{lookup_note}"
        )

    for shell_block in shell_blocks:
        agent_context.event_bus.emit(
            EventType.CODE_RUN, AGENT_NAME, f"shell: {first_line(shell_block.code)}"
        )
        shell_result = agent_context.code_runner.run_shell_command(shell_block.code, working_folder)
        if not shell_result.succeeded:
            return build_failure(
                f"A shell command failed, so the script was not run.\n"
                f"Command: {shell_block.code.strip()}\nError: {shell_result.error_text}"
            )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(python_blocks[0].code, encoding="utf-8")
    agent_context.event_bus.emit(EventType.CODE_RUN, AGENT_NAME, f"running {script_path.name}")
    script_result = agent_context.code_runner.run_python_script(script_path, working_folder)

    if not script_result.succeeded:
        message = f"The script failed.\n{script_result.error_text}"
        if script_result.timed_out:
            message += "\nMake it faster: less data, fewer epochs or trees, early stopping."
        return AttemptOutcome(False, {}, script_result.stdout, script_result.error_text, message)

    result_data = extract_tagged_json(script_result.stdout, result_tag)
    if result_data is None:
        end_of_output = keep_end_of_text(script_result.stdout, OUTPUT_END_CHARS_FOR_MODEL)
        message = (
            f"The script ran but did not print a valid <{result_tag}>JSON</{result_tag}> line.\n"
            f"End of its output:\n{end_of_output}"
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
    """Do one coding task for another agent. Never raises for code problems: check succeeded."""
    script_path = working_folder / script_name
    caller_name = display_name(caller_key)
    agent_context.event_bus.emit(
        EventType.AGENT_STARTED, AGENT_NAME, f"task from {caller_name}: {first_line(task_text)}"
    )
    common_values = build_common_prompt_values(AGENT_KEY, ledger, agent_context)

    history_entries: list[str] = []
    attempts_used = 0
    lookups_left = MAX_LIBRARY_LOOKUPS
    previous_error_line = ""
    outcome = build_failure("The Coder did not run.")

    while attempts_used < MAX_CODE_ATTEMPTS:
        prompt = render_prompt(
            CODER_PROMPT,
            **common_values,
            caller_name=caller_name,
            task_text=task_text,
            script_path=script_path,
            working_folder=working_folder,
            random_seed=RANDOM_SEED,
            shell_name=agent_context.environment.shell_name,
            lookups_left=lookups_left,
            result_tag=result_tag,
            result_keys_text=result_keys_text,
            attempt_number=attempts_used + 1,
            max_code_attempts=MAX_CODE_ATTEMPTS,
            history_text="\n\n".join(history_entries) or "This is your first attempt.",
        )
        answer_text = agent_context.llm_client.ask_text(AGENT_NAME, prompt)
        code_blocks = extract_code_blocks(answer_text)

        lookup_names = collect_lookup_names(code_blocks)
        asked_only_for_lookups = bool(lookup_names) and all(
            block.language == LOOKUP_LANGUAGE for block in code_blocks
        )
        if asked_only_for_lookups and lookups_left > 0:
            names_to_look_up = lookup_names[:lookups_left]
            lookups_left -= len(names_to_look_up)
            history_entries.append(
                run_library_lookups(names_to_look_up, working_folder, agent_context)
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
        history_entries.append(f"Attempt {attempts_used} failed.\n{outcome.message_for_model}")
        if error_line == previous_error_line:
            agent_context.event_bus.emit(
                EventType.WARNING, AGENT_NAME, "the same error came twice in a row, stopping"
            )
            break
        previous_error_line = error_line

    if outcome.succeeded:
        agent_context.event_bus.emit(
            EventType.AGENT_FINISHED, AGENT_NAME, f"done after {attempts_used} attempt(s)"
        )
    else:
        agent_context.event_bus.emit(
            EventType.AGENT_FINISHED, AGENT_NAME, f"gave up after {attempts_used} attempt(s)"
        )
    return CoderResult(
        succeeded=outcome.succeeded,
        result_data=outcome.result_data,
        script_path=script_path,
        stdout=outcome.stdout,
        error_text=keep_end_of_text(outcome.error_text, MAX_ERROR_TEXT_CHARS),
        attempts_used=attempts_used,
    )
