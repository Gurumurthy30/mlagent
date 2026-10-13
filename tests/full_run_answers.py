"""The scripted answers for a complete small run on the sample churn project.

The run has an EDA round, a baseline, a second model, two reviews, then the budget stop (2
experiment rounds) and the Submitter. Every script really runs on real files.
"""

from scripted_answers import coder_answer_printing
from test_data_agents_on_sample_project import EDA_SCRIPT, INSPECTION_SCRIPT
from test_modeling_pipeline_on_sample_project import (
    CHECK_SCRIPT,
    FEATURES_SCRIPT,
    SPLITS_SCRIPT,
    TRAIN_SCRIPT,
    as_python_answer,
)

from mlagent.agents.data_analyst import (
    ChecklistDraft,
    CoverageMark,
    EdaSummary,
    NewFinding,
    Question,
    QuestionPlan,
    RoundReading,
)
from mlagent.agents.data_engineer import BriefReading
from mlagent.agents.evaluator import VerdictAnswer
from mlagent.agents.ml_lead import PlanDraft
from mlagent.agents.modeler import ModelChoice
from mlagent.agents.reviewer import ReviewDraft
from mlagent.state import FeatureIdea, FileInfo, ProjectBrief

BLEND_SCRIPT = """
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

train = pd.read_csv(Path(PROJECT_FOLDER) / "train.csv")
first = pd.read_csv(FIRST_PREDICTIONS)
second = pd.read_csv(SECOND_PREDICTIONS)
blend = first.copy()
blend["probability"] = (first["probability"] + second["probability"]) / 2
scores = []
for fold in sorted(set(blend["fold"]) - {-1}):
    rows = blend["fold"] == fold
    labels = train.loc[blend.loc[rows, "example_id"], "churn"]
    predicted = blend.loc[rows, "probability"] > 0.5
    scores.append(float(f1_score(labels, predicted, average="macro")))
output_folder = Path(OUTPUT_FOLDER)
output_folder.mkdir(parents=True, exist_ok=True)
blend_path = output_folder / "blend_predictions.csv"
blend_path.write_text(Path(FIRST_TEST_PREDICTIONS).read_text())
result = {"validation_scores": scores, "test_predictions_path": str(blend_path),
          "method": "average of probabilities"}
print("<BLEND>" + json.dumps(result) + "</BLEND>")
"""

DIAGNOSIS = {"weak_spots": ["recall on churners is weak"], "comparison": "", "notes": []}
REPORT_TEXT = "## Summary\nA logistic regression baseline was trained and checked."


def churn_brief() -> ProjectBrief:
    return ProjectBrief(
        summary="Customer table with a churn label.",
        files=[FileInfo(path="train.csv", kind="table", role="train")],
        modality="tabular",
        task_family="classification",
        target="churn",
        id_column="customer_id",
        primary_metric="macro_f1",
        extra_metrics=["auc"],
        deliverable_format="predictions.csv with customer_id and churn",
    )


def check_answer(workspace, experiment_id: str) -> str:
    predictions_path = workspace.artifacts_folder / experiment_id / "validation_predictions.csv"
    return as_python_answer(CHECK_SCRIPT, workspace, VALIDATION_PREDICTIONS=predictions_path)


def answers_before_the_modeler(workspace) -> list:
    """Data Engineer, Data Analyst, ML Lead and the Feature Engineer's round 0."""
    plan_draft = PlanDraft(
        validation_scheme="stratified_kfold",
        validation_reason="the label is imbalanced",
        primary_metric="macro_f1",
        higher_is_better=True,
        extra_metrics=["auc"],
        baseline_description="logistic regression on the numeric columns",
        playbook="Use class weights.",
    )
    checklist_items = ["target distribution", "missing values"]
    return [
        as_python_answer(INSPECTION_SCRIPT, workspace),
        BriefReading(brief=churn_brief(), done=True),
        ChecklistDraft(items=checklist_items),
        QuestionPlan(questions=[Question(text="Churn rate and missing values?")]),
        as_python_answer(EDA_SCRIPT, workspace),
        RoundReading(
            findings=[NewFinding(statement="About a quarter of customers churn.", priority="high")],
            coverage=[CoverageMark(item=item, status="covered") for item in checklist_items],
        ),
        EdaSummary(
            issues=["imbalanced target"],
            feature_ideas=[FeatureIdea(description="age bucket", finding_id="finding_001")],
            leakage_risks=[],
            validation_hints=["stratify on churn"],
            model_hints=["gradient boosting"],
        ),
        plan_draft,
        as_python_answer(SPLITS_SCRIPT, workspace),
        as_python_answer(FEATURES_SCRIPT, workspace),
    ]


def answers_from_the_modeler(workspace) -> list:
    """Two rounds of model, evaluation and review, then the Submitter."""
    approval = VerdictAnswer(verdict="approve", reasons=["score reproduced"])
    diagnosis = coder_answer_printing("DIAG", DIAGNOSIS)
    first_choice = ModelChoice(model_family="logistic_regression", model_params={"C": 1.0})
    second_choice = ModelChoice(model_family="logistic_regression", model_params={"C": 0.5})
    first_predictions = workspace.artifacts_folder / "experiment_001" / "validation_predictions.csv"
    second_predictions = (
        workspace.artifacts_folder / "experiment_002" / "validation_predictions.csv"
    )
    first_test_predictions = workspace.artifacts_folder / "experiment_001" / "test_predictions.csv"
    blend_script = as_python_answer(
        BLEND_SCRIPT,
        workspace,
        FIRST_PREDICTIONS=first_predictions,
        SECOND_PREDICTIONS=second_predictions,
        FIRST_TEST_PREDICTIONS=first_test_predictions,
        OUTPUT_FOLDER=workspace.artifacts_folder / "submitter",
    )
    return [
        first_choice,
        as_python_answer(TRAIN_SCRIPT, workspace),
        check_answer(workspace, "experiment_001"),
        approval,
        diagnosis,
        ReviewDraft(assessment="baseline works", focus="model", reason="try another setup"),
        second_choice,
        as_python_answer(TRAIN_SCRIPT, workspace),
        check_answer(workspace, "experiment_002"),
        approval,
        diagnosis,
        ReviewDraft(assessment="no further gain", focus="features", reason="try new inputs"),
        blend_script,
        REPORT_TEXT,
    ]
