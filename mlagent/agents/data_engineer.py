"""The Data Engineer: opens the project folder like a new colleague and writes the project brief.

Each round the Coder runs one inspection script, then the LLM reads what it found and updates
the brief. It stops when the key facts are known or after MAX_DATA_ENGINEER_ROUNDS rounds.
"""

from pydantic import BaseModel

from mlagent.agents.base import AgentContext, AgentFailedError, facts_to_text
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
AGENT_NAME = "Data Engineer"
RESULT_TAG = "BRIEF"

REQUIRED_FIELDS = ("files", "modality", "task_family", "primary_metric", "deliverable_format")
MUST_HAVE_FIELDS = ("files", "modality", "task_family")

FIRST_INSPECTION = """\
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

RESULT_KEYS = """\
- "files": list of objects {"path": path relative to the project folder, "size_mb": number, \
"kind": one of table, image, audio, video, text, document, model, other, "columns": list of column names for tables (at most 60)}
- "documents": list of objects {"path": ..., "excerpt": the first 1500 characters}
- "peeks": object mapping a file or folder path to a short factual description of its content
- "notes": list of other facts or problems you found"""

READ_PROMPT = """\
You are the Data Engineer of an ML team. You write the project brief that every other agent \
relies on. You record facts and the user's wishes. You do not plan experiments or choose models.

THE USER'S WORDS. They win over anything in the files. If the files disagree, follow the user \
and add the conflict to open_questions.
Task: {user_task}
Metrics: {user_metrics}

BRIEF SO FAR:
{brief_so_far}

INSPECTION ROUND {round_number} OF {max_rounds}. The inspection script found:
{facts_text}

RULES
- Use only these facts and the user's words. Never guess: leave the field empty and add the \
question to open_questions.
- Copy every field that did not change from BRIEF SO FAR exactly, so nothing is lost.
- A column that is in train but not in test is probably the target. A sample submission shows \
the exact shape of the deliverable. If labels are in folder names or a separate file, say so \
in target.

FIELDS
- summary: two sentences: what the data is and what is to be predicted.
- modality: for example tabular, image, text, audio, video, time_series
- task_family: for example classification, regression, clustering, forecasting, detection, \
segmentation, ranking, generation, speech_recognition
- target: the column or label to predict. Empty if none.
- id_column: the column that identifies a row or example. Empty if none.
- primary_metric: ONE metric in lowercase snake_case, from the user's metrics, else from the \
documents. If still unknown, use the usual metric for the task family and add that choice to \
open_questions. Example: "macro F1" becomes macro_f1.
- higher_is_better: true when a bigger value is better. False for error metrics such as rmse, \
mae, log_loss.
- extra_metrics: other metrics the user wants reported.
- deliverable_format: exactly what the final predictions file looks like (file name, columns, \
what one row means). Example: predictions.csv with columns customer_id and churn, one row per \
row of test.csv, churn is 0 or 1.
- files: at most {max_files} entries, grouping huge folders into one entry. role is one of \
train, test, sample_submission, description, other. For table files, include columns (up to 60 names).
- group_columns: list of candidate group/cluster/entity ID column names (e.g. user_id, patient_id, group_id).
- time_columns: list of date/time/timestamp column names.
- text_columns: list of long/free text column names.

OUTPUT:
Return JSON with exactly these fields in order:
- brief: project brief object containing all fields (summary, modality, task_family, target, id_column, primary_metric, higher_is_better, extra_metrics, deliverable_format, files, group_columns, time_columns, text_columns).
- done: true when files, modality, task_family, primary_metric and deliverable_format are all known.
- next_tasks: list of what the next inspection must find out (each task concrete and answerable by one short script; empty when done)."""


class BriefReading(BaseModel):
    brief: ProjectBrief
    done: bool
    next_tasks: list[str] = []


def run_data_engineer(ledger: Ledger, agent_context: AgentContext) -> ProjectBrief:
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "inspecting the project folder")

    project_folder = agent_context.workspace.project_folder
    working_folder = agent_context.workspace.artifacts_folder / AGENT_KEY
    skipped_folders = (
        f"{', '.join(IGNORED_FOLDER_NAMES)}, "
        f"and any folder whose name starts with {ARCHIVED_WORKSPACE_PREFIX}"
    )

    brief = ledger.brief
    missing = [name for name in REQUIRED_FIELDS if not getattr(brief, name)]
    inspection_tasks = FIRST_INSPECTION
    last_failure = ""

    for round_number in range(1, MAX_DATA_ENGINEER_ROUNDS + 1):
        # 1. Ask the Coder to inspect the project folder.
        task_text = (
            "Inspect the project folder and report facts. Do these tasks:\n"
            f"{inspection_tasks}\n\n"
            "Rules:\n"
            f"- Project folder to inspect: {project_folder}\n"
            f"- Skip these folders: {skipped_folders}\n"
            "- Only read. Never change, move or delete anything.\n"
            f"- Never fully load a file larger than {BIG_FILE_MB} MB. "
            "Read its head, a sample or a count.\n"
            "- Keep every text excerpt short.\n\n"
            f"The user's task, for context: {ledger.user_task}"
        )
        if last_failure:
            task_text += f"\n\nThe previous inspection script failed with:\n{last_failure}"

        coder_result = run_coder(
            task_text=task_text,
            result_tag=RESULT_TAG,
            result_keys_text=RESULT_KEYS,
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

        # 2. Ask the LLM to update the brief from what the script found.
        prompt = READ_PROMPT.format(
            user_task=ledger.user_task,
            user_metrics=ledger.user_metrics_text or "not given",
            brief_so_far=brief.model_dump_json(),
            round_number=round_number,
            max_rounds=MAX_DATA_ENGINEER_ROUNDS,
            facts_text=facts_to_text(coder_result.result_data),
            max_files=MAX_FILES_IN_LISTING,
        )
        reading = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, BriefReading)

        # 3. Tidy the brief: keep only files that exist, cap the list, flag big files, extract columns.
        brief = reading.brief
        files_on_disk = [item for item in brief.files if (project_folder / item.path).exists()]
        brief.files = (files_on_disk or brief.files)[:MAX_FILES_IN_LISTING]
        brief.has_large_files = any(item.size_mb >= BIG_FILE_MB for item in brief.files)

        # Populate columns from coder result if available
        if isinstance(coder_result.result_data, dict):
            raw_files = coder_result.result_data.get("files", [])
            cols_by_path = {
                f["path"]: f["columns"]
                for f in raw_files
                if isinstance(f, dict) and "path" in f and "columns" in f
            }
            for file_info in brief.files:
                if not file_info.columns and file_info.path in cols_by_path:
                    file_info.columns = cols_by_path[file_info.path]
                file_info.columns = file_info.columns[:60]

        missing = [name for name in REQUIRED_FIELDS if not getattr(brief, name)]
        event_bus.emit(
            EventType.INFO,
            AGENT_NAME,
            f"round {round_number} of {MAX_DATA_ENGINEER_ROUNDS}: modality "
            f"{brief.modality or '?'}, task {brief.task_family or '?'}, "
            f"still unknown: {', '.join(missing) or 'nothing'}",
        )
        if reading.done and not missing:
            break
        next_tasks = reading.next_tasks or [f"Find out: {', '.join(missing)}"]
        inspection_tasks = "\n".join(f"{i}. {task}" for i, task in enumerate(next_tasks, 1))

    # 4. Fail loudly if the essentials are still unknown; otherwise record the gaps.
    blocking = [name for name in MUST_HAVE_FIELDS if name in missing]
    if blocking:
        reason = f"The Data Engineer could not find out: {', '.join(blocking)}."
        if last_failure:
            reason += f" The last inspection script failed with: {last_failure}"
        raise AgentFailedError(reason)

    brief.open_questions += [f"Could not determine: {name}" for name in missing]
    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"brief ready: {len(brief.files)} file entries, metric {brief.primary_metric or 'unknown'}",
    )
    return brief
