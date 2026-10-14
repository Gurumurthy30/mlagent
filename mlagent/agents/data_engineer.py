"""The Data Engineer: opens the project folder like a new colleague and writes the project brief.

Each round the Coder runs one inspection script, then the LLM reads what it found and updates
the brief. It stops when the key facts are known or after MAX_DATA_ENGINEER_ROUNDS rounds.
"""

from pydantic import BaseModel

from mlagent.agents.base import (
    AgentContext,
    AgentFailedError,
    build_common_prompt_values,
    display_name,
    facts_to_text,
    format_numbered_list,
    render_prompt,
)
from mlagent.agents.coder import run_coder
from mlagent.events import EventType
from mlagent.settings import (
    ARCHIVED_WORKSPACE_PREFIX,
    BIG_FILE_MB,
    IGNORED_FOLDER_NAMES,
    MAX_DATA_ENGINEER_ROUNDS,
    MAX_FILES_IN_LISTING,
)
from mlagent.state import Ledger, ProjectBrief

AGENT_KEY = "data_engineer"
AGENT_NAME = display_name(AGENT_KEY)
RESULT_TAG = "BRIEF"

DEFAULT_INSPECTION_TASKS = """\
1. List the files recursively with size in MB and a kind (table, image, audio, video, text, \
document, model, other). For a folder with very many files, give one entry per extension with \
the count and total size.
2. Read the text documents (README, task description, data dictionary, .md, .txt) and report \
the first 1500 characters of each.
3. Peek at the data by type. Tables: columns, dtypes, row count, first 3 rows. Images: count, \
extensions, a few image sizes. Audio: count, and durations or sample rates of a few files. \
Text: a few sample lines. Anything else: say what it is.
4. Find which files are for training, for testing, and any sample submission or label file. \
Report the columns or format of a sample submission. Note labels kept in folder names or in \
separate files. For tables, also report which columns are in the train file but not in the \
test file.
5. Report anything odd: empty or unreadable files, mixed formats, missing pieces."""

RESULT_KEYS_TEXT = """\
- "files": list of objects {"path": path relative to the project folder, "size_mb": number, \
"kind": one of table, image, audio, video, text, document, model, other}
- "documents": list of objects {"path": ..., "excerpt": the first 1500 characters}
- "peeks": object mapping a file or folder path to a short factual description of its content
- "notes": list of other facts or problems you found"""

DATA_ENGINEER_READ_PROMPT = """\
You are the Data Engineer of an ML team. You opened the project folder like a new colleague \
and you write the project brief that every other agent relies on. You record facts and the \
user's wishes. You do not plan experiments and you do not choose models.

{run_rules}

WHAT THE USER WROTE. Their words always win over what you find in files. If the files \
disagree, follow the user and add the conflict to open_questions.
Task: {user_task}
Metrics: {user_metrics}

BRIEF SO FAR:
{brief_so_far}

INSPECTION ROUND {round_number} OF {max_rounds}. The inspection script found:
{facts_text}

Update the brief using only these facts and the user's words.
- Never guess a fact that is not in them. Leave that field empty and add the question to \
open_questions.
- Copy every field that did not change from BRIEF SO FAR exactly as it is, so nothing is lost \
between rounds.

How to read the facts:
- A test file usually has the same columns as the train file, without the target. A column \
that is in train but not in test is probably the target.
- A sample submission shows the exact shape of the deliverable.
- If labels are kept in folder names or in a separate file, say so in target.
- Documents describe the task. The user's words still win.

Meaning of the fields:
- summary: two sentences: what the data is and what is to be predicted.
- modality: your own words, for example tabular, image, text, audio, video, time_series
- task_family: your own words, for example classification, regression, clustering, \
forecasting, detection, segmentation, ranking, generation, speech_recognition
- target: the column or label to predict. Empty for tasks without one.
- id_column: the column that identifies a row or example. Empty if none.
- primary_metric: ONE metric in lowercase snake_case that decides which model is best, for \
example "macro F1" becomes macro_f1 and "area under the ROC curve" becomes auc. Take it from \
the user's metrics, else from the documents. If still unknown, pick the usual metric for this \
task family and add that choice to open_questions.
- higher_is_better: true when a bigger primary_metric value is better. It is false for error \
metrics such as rmse, mae and log_loss.
- extra_metrics: the other metrics the user wants reported
- deliverable_format: exactly what the final predictions file must look like (file name, \
columns, what one row means). For example: predictions.csv with the columns customer_id and \
churn, one row per row of test.csv, churn is 0 or 1.
- files: at most {max_files} entries, grouping huge folders into one entry. role is one of \
train, test, sample_submission, description, other.

Return JSON: brief (all fields), done (true when files, modality, task_family, primary_metric \
and deliverable_format are all known), next_tasks (what the next inspection should find out: \
each task concrete and answerable by one short script, empty when done)."""


class BriefReading(BaseModel):
    brief: ProjectBrief
    done: bool
    next_tasks: list[str] = []


def describe_ignored_folders() -> str:
    names = ", ".join(IGNORED_FOLDER_NAMES)
    return f"{names}, and any folder whose name starts with {ARCHIVED_WORKSPACE_PREFIX}"


def build_inspection_task(
    task_list: str, ledger: Ledger, agent_context: AgentContext, last_failure: str
) -> str:
    task_text = (
        "Inspect the project folder and report facts. Do these tasks:\n"
        f"{task_list}\n\n"
        "Rules:\n"
        f"- Project folder to inspect: {agent_context.workspace.project_folder}\n"
        f"- Skip these folders: {describe_ignored_folders()}\n"
        "- Only read. Never change, move or delete anything.\n"
        f"- Never fully load a file larger than {BIG_FILE_MB} MB. "
        "Read its head, a sample or a count.\n"
        "- Keep every text excerpt short.\n\n"
        f"The user's task, for context: {ledger.user_task}"
    )
    if last_failure:
        task_text += f"\n\nThe previous inspection script failed with:\n{last_failure}"
    return task_text


def find_missing_fields(brief: ProjectBrief) -> list[str]:
    checks = {
        "files": bool(brief.files),
        "modality": bool(brief.modality),
        "task_family": bool(brief.task_family),
        "primary_metric": bool(brief.primary_metric),
        "deliverable_format": bool(brief.deliverable_format),
    }
    return [field_name for field_name, is_known in checks.items() if not is_known]


def clean_brief(brief: ProjectBrief) -> ProjectBrief:
    """Keep the file list short and compute the large-file flag in code, not by the LLM."""
    cleaned_brief = brief.model_copy(deep=True)
    cleaned_brief.files = cleaned_brief.files[:MAX_FILES_IN_LISTING]
    cleaned_brief.has_large_files = any(file.size_mb >= BIG_FILE_MB for file in cleaned_brief.files)
    return cleaned_brief


def run_data_engineer(ledger: Ledger, agent_context: AgentContext) -> ProjectBrief:
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "inspecting the project folder")
    working_folder = agent_context.workspace.artifacts_folder / AGENT_KEY
    common_values = build_common_prompt_values(AGENT_KEY, ledger, agent_context)

    brief = ledger.brief
    task_list = DEFAULT_INSPECTION_TASKS
    last_failure = ""
    for round_number in range(1, MAX_DATA_ENGINEER_ROUNDS + 1):
        coder_result = run_coder(
            task_text=build_inspection_task(task_list, ledger, agent_context, last_failure),
            result_tag=RESULT_TAG,
            result_keys_text=RESULT_KEYS_TEXT,
            working_folder=working_folder,
            script_name=f"inspect_project_round_{round_number}.py",
            caller_key=AGENT_KEY,
            ledger=ledger,
            agent_context=agent_context,
        )
        if not coder_result.succeeded:
            last_failure = coder_result.error_text
            event_bus.emit(
                EventType.WARNING, AGENT_NAME, f"round {round_number}: inspection failed"
            )
            continue
        last_failure = ""

        prompt = render_prompt(
            DATA_ENGINEER_READ_PROMPT,
            **common_values,
            user_task=ledger.user_task,
            user_metrics=ledger.user_metrics_text or "not given",
            brief_so_far=brief.model_dump_json(),
            round_number=round_number,
            max_rounds=MAX_DATA_ENGINEER_ROUNDS,
            facts_text=facts_to_text(coder_result.result_data),
            max_files=MAX_FILES_IN_LISTING,
        )
        reading = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, BriefReading)
        brief = clean_brief(reading.brief)
        missing_fields = find_missing_fields(brief)
        event_bus.emit(
            EventType.INFO,
            AGENT_NAME,
            f"round {round_number} of {MAX_DATA_ENGINEER_ROUNDS}: modality "
            f"{brief.modality or '?'}, task {brief.task_family or '?'}, "
            f"still unknown: {', '.join(missing_fields) or 'nothing'}",
        )
        if reading.done and not missing_fields:
            break
        next_tasks = reading.next_tasks or [f"Find out: {', '.join(missing_fields)}"]
        task_list = format_numbered_list(next_tasks)

    return finish_brief(brief, last_failure, agent_context)


def finish_brief(
    brief: ProjectBrief, last_failure: str, agent_context: AgentContext
) -> ProjectBrief:
    missing_fields = find_missing_fields(brief)
    must_have_fields = [
        name for name in missing_fields if name in ("files", "modality", "task_family")
    ]
    if must_have_fields:
        reason = f"The Data Engineer could not find out: {', '.join(must_have_fields)}."
        if last_failure:
            reason += f" The last inspection script failed with: {last_failure}"
        raise AgentFailedError(reason)

    for field_name in missing_fields:
        brief.open_questions.append(f"Could not determine: {field_name}")
    agent_context.event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"brief ready: {len(brief.files)} file entries, metric {brief.primary_metric or 'unknown'}",
    )
    return brief
