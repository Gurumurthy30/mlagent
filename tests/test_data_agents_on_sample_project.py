"""Runs the Data Engineer and the Data Analyst on a real sample folder.

The LLM answers are scripted, but the scripts the Coder runs really read the files with pandas,
so the numbers in the prompts come from the data and not from the test.
"""

import json

import numpy as np
import pandas as pd
import pytest

from mlagent.agents.data_analyst import (
    ChecklistDraft,
    CoverageMark,
    EdaSummary,
    NewFinding,
    Question,
    QuestionPlan,
    RoundReading,
    run_data_analyst,
)
from mlagent.agents.data_engineer import BriefReading, run_data_engineer
from mlagent.memory import create_new_ledger
from mlagent.state import FeatureIdea, FileInfo, ProjectBrief

INSPECTION_SCRIPT = """
import json
import os
from pathlib import Path

import pandas as pd

project_folder = Path(PROJECT_FOLDER)
skipped_names = {"mlagent_workspace", ".git", ".venv", "venv", "__pycache__", "node_modules"}
files, documents, peeks = [], [], {}
for current_folder, folder_names, file_names in os.walk(project_folder):
    folder_names[:] = [
        name for name in folder_names
        if name not in skipped_names and not name.startswith("mlagent_workspace_old_")
    ]
    for file_name in file_names:
        path = Path(current_folder) / file_name
        relative_path = path.relative_to(project_folder).as_posix()
        kind = "table" if path.suffix == ".csv" else "document"
        size_mb = round(path.stat().st_size / 1e6, 3)
        files.append({"path": relative_path, "size_mb": size_mb, "kind": kind})
        if path.suffix == ".md":
            excerpt = path.read_text(encoding="utf-8")[:1500]
            documents.append({"path": relative_path, "excerpt": excerpt})
        if path.suffix == ".csv":
            table = pd.read_csv(path)
            column_text = ", ".join(table.columns)
            peeks[relative_path] = "columns: " + column_text + "; rows: " + str(len(table))
facts = {"files": files, "documents": documents, "peeks": peeks, "notes": []}
print("<BRIEF>" + json.dumps(facts) + "</BRIEF>")
"""

EDA_SCRIPT = """
import json
from pathlib import Path

import pandas as pd

train_table = pd.read_csv(Path(PROJECT_FOLDER) / "train.csv")
churn_rate = float(train_table["churn"].mean())
missing_counts = {name: int(count) for name, count in train_table.isna().sum().items() if count}
rounded_rate = str(round(churn_rate, 4))
answer_text = "churn rate " + rounded_rate + "; missing values per column " + str(missing_counts)
answer = {"question": "target and missing values", "result": answer_text}
print("<FINDINGS>" + json.dumps({"answers": [answer], "problems": []}) + "</FINDINGS>")
"""


def as_python_answer(script_text: str, project_folder) -> str:
    filled_script = script_text.replace("PROJECT_FOLDER", repr(str(project_folder)))
    return f"```python\n{filled_script}\n```"


@pytest.fixture
def sample_project(workspace):
    """A small churn project: train.csv with a label, test.csv without, a sample submission."""
    project_folder = workspace.project_folder
    random_generator = np.random.default_rng(0)
    row_count = 200
    train_table = pd.DataFrame(
        {
            "customer_id": range(row_count),
            "age": random_generator.integers(18, 80, row_count),
            "monthly_charges": random_generator.normal(70, 20, row_count).round(2),
            "churn": random_generator.binomial(1, 0.25, row_count),
        }
    )
    train_table.loc[:9, "age"] = np.nan
    train_table.to_csv(project_folder / "train.csv", index=False)
    train_table.drop(columns="churn").head(50).to_csv(project_folder / "test.csv", index=False)
    pd.DataFrame({"customer_id": range(50), "churn": 0}).to_csv(
        project_folder / "sample_submission.csv", index=False
    )
    (project_folder / "README.md").write_text(
        "Predict customer churn. Scored with macro F1.", encoding="utf-8"
    )
    return project_folder, float(train_table["churn"].mean())


def test_both_agents_work_from_the_real_files(make_agent_context, sample_project):
    project_folder, real_churn_rate = sample_project
    ledger = create_new_ledger("Predict which customers churn.", "macro F1 mainly")

    brief_to_return = ProjectBrief(
        summary="Churn table with train, test and a sample submission.",
        files=[FileInfo(path="train.csv", kind="table", role="train")],
        modality="tabular",
        task_family="classification",
        target="churn",
        id_column="customer_id",
        primary_metric="macro_f1",
        deliverable_format="predictions.csv with customer_id and churn",
    )
    engineer_context = make_agent_context(
        [
            as_python_answer(INSPECTION_SCRIPT, project_folder),
            BriefReading(brief=brief_to_return, done=True),
        ],
        timeout_seconds=120,
    )

    brief = run_data_engineer(ledger, engineer_context)

    reading_prompt = engineer_context.llm_client.received_prompts[1]
    assert "train.csv" in reading_prompt
    assert "columns: customer_id, age, monthly_charges, churn; rows: 200" in reading_prompt
    assert "Predict customer churn. Scored with macro F1." in reading_prompt
    assert '"path": "sample_submission.csv"' in reading_prompt
    assert "mlagent_workspace" not in reading_prompt.split("The inspection script found:")[1]
    assert brief.task_family == "classification"

    ledger.brief = brief
    analyst_context = make_agent_context(
        [
            ChecklistDraft(items=["target distribution", "missing values"]),
            QuestionPlan(questions=[Question(text="Churn rate and missing values?")]),
            as_python_answer(EDA_SCRIPT, project_folder),
            RoundReading(
                findings=[NewFinding(statement="About a quarter of customers churn.")],
                coverage=[
                    CoverageMark(item="target distribution", status="covered"),
                    CoverageMark(item="missing values", status="covered"),
                ],
            ),
            EdaSummary(
                issues=["imbalanced target"],
                feature_ideas=[FeatureIdea(description="age bucket", finding_id="finding_001")],
                leakage_risks=[],
                validation_hints=["stratify on churn"],
                model_hints=["gradient boosting"],
            ),
        ],
        timeout_seconds=120,
    )

    eda = run_data_analyst(ledger, analyst_context)

    read_prompt = analyst_context.llm_client.received_prompts[3]
    assert f"churn rate {round(real_churn_rate, 4)}" in read_prompt
    assert "'age': 10" in read_prompt
    assert set(eda.checklist.values()) == {"covered"}
    assert eda.findings[0].finding_id == "finding_001"
    assert eda.eda_loops_used == 1


def test_scripts_and_commands_of_both_agents_are_saved_in_the_workspace(
    make_agent_context, sample_project, workspace
):
    project_folder, _ = sample_project
    brief_to_return = ProjectBrief(
        summary="x",
        files=[FileInfo(path="train.csv")],
        modality="tabular",
        task_family="classification",
        primary_metric="macro_f1",
        deliverable_format="predictions.csv",
    )
    agent_context = make_agent_context(
        [
            as_python_answer(INSPECTION_SCRIPT, project_folder),
            BriefReading(brief=brief_to_return, done=True),
        ],
        timeout_seconds=120,
    )

    run_data_engineer(create_new_ledger("task", "metric"), agent_context)

    saved_script = workspace.artifacts_folder / "data_engineer" / "inspect_project_round_1.py"
    assert saved_script.exists()
    log_lines = workspace.commands_log_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(log_lines[0])["exit_code"] == 0
