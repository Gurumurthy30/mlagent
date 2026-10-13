import pytest
from scripted_answers import coder_answer_printing, two_failed_coder_attempts

from mlagent.agents.base import AgentFailedError
from mlagent.agents.data_engineer import (
    BriefReading,
    find_missing_fields,
    run_data_engineer,
)
from mlagent.events import EventType
from mlagent.memory import create_new_ledger
from mlagent.state import FileInfo, ProjectBrief

FACTS = {
    "files": [{"path": "train.csv", "size_mb": 1.2, "kind": "table"}],
    "documents": [],
    "peeks": {"train.csv": "columns: id, age, churn; 1000 rows"},
    "notes": [],
}


def complete_brief(**changes) -> ProjectBrief:
    values = {
        "summary": "Customer table with a churn label.",
        "files": [FileInfo(path="train.csv", size_mb=1.2, kind="table", role="train")],
        "modality": "tabular",
        "task_family": "classification",
        "target": "churn",
        "id_column": "id",
        "primary_metric": "macro_f1",
        "higher_is_better": True,
        "extra_metrics": ["auc"],
        "deliverable_format": "predictions.csv with columns id, churn",
    }
    values.update(changes)
    return ProjectBrief(**values)


def reading_answer(brief: ProjectBrief, done: bool = True, next_tasks=()) -> BriefReading:
    return BriefReading(brief=brief, done=done, next_tasks=list(next_tasks))


def new_ledger():
    return create_new_ledger("Predict which customers churn.", "macro F1 mainly, also AUC")


def test_one_round_builds_the_brief(make_agent_context):
    agent_context = make_agent_context(
        [coder_answer_printing("BRIEF", FACTS), reading_answer(complete_brief())]
    )

    brief = run_data_engineer(new_ledger(), agent_context)

    assert brief.modality == "tabular"
    assert brief.primary_metric == "macro_f1"
    assert brief.extra_metrics == ["auc"]
    assert brief.open_questions == []
    assert brief.has_large_files is False
    assert len(agent_context.llm_client.received_prompts) == 2


def test_the_coder_task_lists_the_inspection_tasks_and_safety_rules(make_agent_context):
    agent_context = make_agent_context(
        [coder_answer_printing("BRIEF", FACTS), reading_answer(complete_brief())]
    )

    run_data_engineer(new_ledger(), agent_context)

    coder_prompt = agent_context.llm_client.received_prompts[0]
    assert "List the files recursively" in coder_prompt
    assert "Never fully load a file larger than 200 MB" in coder_prompt
    assert "mlagent_workspace" in coder_prompt
    assert "mlagent_workspace_old_" in coder_prompt
    assert "Predict which customers churn." in coder_prompt
    assert "<BRIEF>JSON</BRIEF>" in coder_prompt


def test_the_reading_prompt_has_the_users_words_the_facts_and_the_limits(make_agent_context):
    agent_context = make_agent_context(
        [coder_answer_printing("BRIEF", FACTS), reading_answer(complete_brief())]
    )

    run_data_engineer(new_ledger(), agent_context)

    reading_prompt = agent_context.llm_client.received_prompts[1]
    assert "Task: Predict which customers churn." in reading_prompt
    assert "Metrics: macro F1 mainly, also AUC" in reading_prompt
    assert "their words always win" in reading_prompt.lower()
    assert "columns: id, age, churn; 1000 rows" in reading_prompt
    assert "INSPECTION ROUND 1 OF 3" in reading_prompt
    assert "at most 40 entries" in reading_prompt
    assert "Time limit for one script run: 30 minutes" in reading_prompt


def test_missing_metrics_text_is_shown_as_not_given(make_agent_context):
    agent_context = make_agent_context(
        [coder_answer_printing("BRIEF", FACTS), reading_answer(complete_brief())]
    )

    run_data_engineer(create_new_ledger("Predict churn.", ""), agent_context)

    assert "Metrics: not given" in agent_context.llm_client.received_prompts[1]


def test_next_tasks_from_an_unfinished_round_go_into_the_next_inspection(make_agent_context):
    unfinished_brief = complete_brief(deliverable_format="")
    agent_context = make_agent_context(
        [
            coder_answer_printing("BRIEF", FACTS),
            reading_answer(unfinished_brief, done=False, next_tasks=["Read sample_submission.csv"]),
            coder_answer_printing("BRIEF", FACTS),
            reading_answer(complete_brief()),
        ]
    )

    brief = run_data_engineer(new_ledger(), agent_context)

    second_coder_prompt = agent_context.llm_client.received_prompts[2]
    assert "1. Read sample_submission.csv" in second_coder_prompt
    assert "INSPECTION ROUND 2 OF 3" in agent_context.llm_client.received_prompts[3]
    assert brief.deliverable_format != ""
    assert brief.open_questions == []


def test_done_with_a_missing_key_fact_does_not_end_the_loop(make_agent_context):
    brief_without_metric = complete_brief(primary_metric="")
    agent_context = make_agent_context(
        [
            coder_answer_printing("BRIEF", FACTS),
            reading_answer(brief_without_metric, done=True),
            coder_answer_printing("BRIEF", FACTS),
            reading_answer(complete_brief(), done=True),
        ]
    )

    brief = run_data_engineer(new_ledger(), agent_context)

    assert "Find out: primary_metric" in agent_context.llm_client.received_prompts[2]
    assert brief.primary_metric == "macro_f1"


def test_after_the_last_round_unknown_optional_facts_become_open_questions(make_agent_context):
    brief_without_format = complete_brief(deliverable_format="")
    answers = []
    for _ in range(3):
        answers += [
            coder_answer_printing("BRIEF", FACTS),
            reading_answer(brief_without_format, done=False),
        ]
    agent_context = make_agent_context(answers)

    brief = run_data_engineer(new_ledger(), agent_context)

    assert brief.open_questions == ["Could not determine: deliverable_format"]
    assert len(agent_context.llm_client.received_prompts) == 6


def test_a_failed_inspection_is_retried_with_the_error_in_the_task(make_agent_context):
    agent_context = make_agent_context(
        [
            *two_failed_coder_attempts("disk not readable"),
            coder_answer_printing("BRIEF", FACTS),
            reading_answer(complete_brief()),
        ]
    )

    brief = run_data_engineer(new_ledger(), agent_context)

    retry_prompt = agent_context.llm_client.received_prompts[2]
    assert "The previous inspection script failed with" in retry_prompt
    assert "disk not readable" in retry_prompt
    assert brief.modality == "tabular"


def test_three_failed_inspections_stop_the_run_with_a_clear_message(make_agent_context):
    answers = []
    for _ in range(3):
        answers += two_failed_coder_attempts("disk not readable")
    agent_context = make_agent_context(answers)

    with pytest.raises(AgentFailedError, match="could not find out: files, modality, task_family"):
        run_data_engineer(new_ledger(), agent_context)


def test_a_brief_without_modality_or_task_family_stops_the_run(make_agent_context):
    answers = []
    for _ in range(3):
        answers += [
            coder_answer_printing("BRIEF", FACTS),
            reading_answer(complete_brief(modality="", task_family=""), done=False),
        ]
    agent_context = make_agent_context(answers)

    with pytest.raises(AgentFailedError, match="modality, task_family"):
        run_data_engineer(new_ledger(), agent_context)


def test_the_file_list_is_cut_and_large_files_are_flagged_in_code(make_agent_context):
    many_files = [FileInfo(path=f"img_{number}.png", size_mb=0.1) for number in range(100)]
    many_files.append(FileInfo(path="big.parquet", size_mb=900.0, kind="table"))
    brief_with_many_files = complete_brief(files=many_files[-1:] + many_files[:60])
    agent_context = make_agent_context(
        [
            coder_answer_printing("BRIEF", FACTS),
            reading_answer(brief_with_many_files, done=True),
        ]
    )

    brief = run_data_engineer(new_ledger(), agent_context)

    assert len(brief.files) == 40
    assert brief.has_large_files is True


def test_find_missing_fields_names_every_unknown_key_fact():
    assert find_missing_fields(ProjectBrief()) == [
        "files",
        "modality",
        "task_family",
        "primary_metric",
        "deliverable_format",
    ]
    assert find_missing_fields(complete_brief()) == []


def test_events_show_start_progress_and_finish(make_agent_context):
    agent_context = make_agent_context(
        [coder_answer_printing("BRIEF", FACTS), reading_answer(complete_brief())]
    )
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    run_data_engineer(new_ledger(), agent_context)

    engineer_events = [e for e in received_events if e.agent_name == "Data Engineer"]
    assert engineer_events[0].event_type == EventType.AGENT_STARTED
    assert any("modality tabular" in event.message for event in engineer_events)
    assert engineer_events[-1].event_type == EventType.AGENT_FINISHED
    assert "metric macro_f1" in engineer_events[-1].message
