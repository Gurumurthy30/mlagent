"""Runs the ML Lead, Feature Engineer, Modeler and Evaluator on a real sample project.

The LLM answers are scripted, but every script really runs: it reads the CSV files, builds the
splits, trains a scikit-learn model, saves predictions, and the Evaluator recomputes the score
from those saved predictions. This checks that the pieces of the script contract fit together.
"""

import numpy as np
import pandas as pd
import pytest

from mlagent.agents.evaluator import VerdictAnswer, run_evaluator
from mlagent.agents.feature_engineer import run_feature_engineer
from mlagent.agents.ml_lead import PlanDraft, run_ml_lead
from mlagent.agents.modeler import ModelChoice, run_modeler
from mlagent.memory import create_new_ledger
from mlagent.state import ExperimentRun, FileInfo

SPLITS_SCRIPT = """
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv(Path(PROJECT_FOLDER) / "train.csv")
split_count = 3
fold_of_example = [-1] * len(train)
splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=42)
for fold, (_, valid_index) in enumerate(splitter.split(train, train["churn"])):
    for index in valid_index:
        fold_of_example[index] = fold
splits = {
    "scheme": "stratified_kfold", "seed": 42, "n_splits": split_count,
    "id_description": "row position in train.csv",
    "example_ids": list(range(len(train))), "fold_of_example": fold_of_example,
}
path = Path(SPLITS_PATH)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(splits))
result = {"splits_path": str(path), "n_splits": split_count,
          "n_examples": len(train), "n_holdout": 0}
print("<SPLITS>" + json.dumps(result) + "</SPLITS>")
"""

FEATURES_SCRIPT = '''
"""Basic preparation.

fit_preparation(train_table) -> state   learns the medians from training rows only
apply_preparation(table, state) -> DataFrame of numeric model inputs
"""
import json
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = ["age", "monthly_charges", "tenure"]


def fit_preparation(train_table):
    return {column: float(train_table[column].median()) for column in FEATURE_COLUMNS}


def apply_preparation(table, state):
    return table[FEATURE_COLUMNS].fillna(value=state)


if __name__ == "__main__":
    sample = pd.read_csv(Path(PROJECT_FOLDER) / "train.csv").head(100)
    prepared = apply_preparation(sample, fit_preparation(sample))
    assert not prepared.isna().any().any()
    result = {"module_path": __file__, "functions": ["fit_preparation", "apply_preparation"],
              "description": "median imputation of three numeric columns"}
    print("<FEATURES>" + json.dumps(result) + "</FEATURES>")
'''

TRAIN_SCRIPT = """
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, FEATURES_FOLDER)
import features_v1

project = Path(PROJECT_FOLDER)
train = pd.read_csv(project / "train.csv")
test = pd.read_csv(project / "test.csv")
splits = json.loads(Path(SPLITS_PATH).read_text())
fold = np.array(splits["fold_of_example"])
target = train["churn"].to_numpy()

f1_scores, auc_scores = [], []
validation_probability = np.zeros(len(train))
for split_number in range(splits["n_splits"]):
    train_rows = (fold != split_number) & (fold != -1)
    valid_rows = fold == split_number
    state = features_v1.fit_preparation(train.loc[train_rows])
    model = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
    model.fit(features_v1.apply_preparation(train.loc[train_rows], state), target[train_rows])
    valid_inputs = features_v1.apply_preparation(train.loc[valid_rows], state)
    probability = model.predict_proba(valid_inputs)[:, 1]
    validation_probability[valid_rows] = probability
    f1_scores.append(float(f1_score(target[valid_rows], probability > 0.5, average="macro")))
    auc_scores.append(float(roc_auc_score(target[valid_rows], probability)))

folder = Path.cwd()
pd.DataFrame(
    {"example_id": splits["example_ids"], "fold": fold, "probability": validation_probability}
).to_csv(folder / "validation_predictions.csv", index=False)
state = features_v1.fit_preparation(train)
final_model = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
final_model.fit(features_v1.apply_preparation(train, state), target)
predicted = final_model.predict(features_v1.apply_preparation(test, state))
pd.DataFrame({"customer_id": test["customer_id"], "churn": predicted}).to_csv(
    folder / "test_predictions.csv", index=False
)
joblib.dump(final_model, folder / "model.joblib")
result = {
    "validation_scores": f1_scores, "holdout_score": None,
    "extra_scores": {"auc": float(np.mean(auc_scores))},
    "validation_predictions_path": str(folder / "validation_predictions.csv"),
    "test_predictions_path": str(folder / "test_predictions.csv"),
    "model_path": str(folder / "model.joblib"), "notes": "logistic regression baseline",
}
print("<RESULT>" + json.dumps(result) + "</RESULT>")
"""

CHECK_SCRIPT = """
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

train = pd.read_csv(Path(PROJECT_FOLDER) / "train.csv")
predictions = pd.read_csv(VALIDATION_PREDICTIONS)
split_scores = []
for split_number in sorted(set(predictions["fold"]) - {-1}):
    rows = predictions["fold"] == split_number
    labels = train.loc[predictions.loc[rows, "example_id"], "churn"]
    predicted = predictions.loc[rows, "probability"] > 0.5
    split_scores.append(float(f1_score(labels, predicted, average="macro")))

flags = []
for column in train.columns.drop(["customer_id", "churn"]):
    values = train[column].fillna(train[column].median())
    auc = roc_auc_score(train["churn"], values)
    if max(auc, 1 - auc) > 0.98:
        flags.append({"description": f"{column} alone gives AUC {auc:.3f}", "severity": "high"})
checks = {
    "recomputed_score": sum(split_scores) / len(split_scores),
    "recomputed_holdout_score": None, "leakage_flags": flags,
    "train_validation_overlap": 0, "notes": ["recomputed from the saved predictions"],
}
print("<CHECKS>" + json.dumps(checks) + "</CHECKS>")
"""


def as_python_answer(script_text: str, workspace, **extra_values) -> str:
    values = {
        "PROJECT_FOLDER": repr(str(workspace.project_folder)),
        "SPLITS_PATH": repr(str(workspace.splits_path)),
        "FEATURES_FOLDER": repr(str(workspace.features_folder)),
        **{name: repr(str(value)) for name, value in extra_values.items()},
    }
    filled_script = script_text
    for name, value in values.items():
        filled_script = filled_script.replace(name, value)
    return f"```python\n{filled_script}\n```"


def write_sample_project(project_folder, leaky_column: bool = False):
    random_generator = np.random.default_rng(1)
    row_count = 300
    monthly_charges = random_generator.normal(70, 20, row_count)
    churn = (monthly_charges + random_generator.normal(0, 12, row_count) > 75).astype(int)
    train = pd.DataFrame(
        {
            "customer_id": range(row_count),
            "age": random_generator.integers(18, 80, row_count).astype(float),
            "monthly_charges": monthly_charges.round(2),
            "tenure": random_generator.integers(1, 72, row_count),
            "churn": churn,
        }
    )
    train.loc[:14, "age"] = np.nan
    if leaky_column:
        train["refund_amount"] = churn * 50.0
    train.to_csv(project_folder / "train.csv", index=False)
    train.drop(columns=["churn", "refund_amount"], errors="ignore").head(60).to_csv(
        project_folder / "test.csv", index=False
    )


@pytest.fixture
def churn_ledger(workspace):
    write_sample_project(workspace.project_folder)
    ledger = create_new_ledger("Predict which customers churn.", "macro F1 mainly, also AUC")
    ledger.brief.summary = "Customer table with a churn label."
    ledger.brief.files = [FileInfo(path="train.csv", kind="table", role="train")]
    ledger.brief.modality = "tabular"
    ledger.brief.task_family = "classification"
    ledger.brief.target = "churn"
    ledger.brief.id_column = "customer_id"
    ledger.brief.primary_metric = "macro_f1"
    ledger.brief.extra_metrics = ["auc"]
    ledger.brief.deliverable_format = "predictions.csv with customer_id and churn"
    return ledger


def test_the_four_agents_work_together_on_real_files(make_agent_context, workspace, churn_ledger):
    ledger = churn_ledger
    plan_draft = PlanDraft(
        validation_scheme="stratified_kfold",
        validation_reason="the label is imbalanced",
        primary_metric="macro_f1",
        higher_is_better=True,
        extra_metrics=["auc"],
        baseline_description="logistic regression on the numeric columns",
        playbook="Use class weights.",
    )
    context = make_agent_context([plan_draft, as_python_answer(SPLITS_SCRIPT, workspace)], 180)
    ledger.plan = run_ml_lead(ledger, context)
    assert ledger.plan.splits_path == str(workspace.splits_path)

    context = make_agent_context([as_python_answer(FEATURES_SCRIPT, workspace)], 180)
    features_result = run_feature_engineer(ledger, context)
    ledger.features = features_result.features
    ledger.current_feature_version = features_result.current_feature_version
    assert features_result.current_feature_version == 1
    assert workspace.features_file(1).exists()

    choice = ModelChoice(
        model_family="logistic_regression", model_params={"C": 1.0}, reason="simple baseline"
    )
    context = make_agent_context([choice, as_python_answer(TRAIN_SCRIPT, workspace)], 180)
    run = run_modeler(ledger, context)
    assert not run.crashed, run.error
    ledger.runs.append(run)
    assert len(run.validation_scores) == 3
    assert run.score_mean > 0.6
    assert run.extra_scores["auc"] > 0.7
    assert pd.read_csv(run.test_predictions_path).shape == (60, 2)

    check_script = as_python_answer(
        CHECK_SCRIPT, workspace, VALIDATION_PREDICTIONS=run.validation_predictions_path
    )
    approval = VerdictAnswer(verdict="approve", reasons=["score reproduced from predictions"])
    context = make_agent_context([check_script, approval], 180)
    evaluation = run_evaluator(ledger, context, run.experiment_id)
    ledger.evaluations.append(evaluation)

    assert evaluation.verdict == "approve"
    assert evaluation.checks["recomputed_score"] == pytest.approx(run.score_mean)
    assert evaluation.checks["leakage_flags"] == []
    assert ledger.best_approved_run().experiment_id == "experiment_001"


def test_the_evaluator_catches_a_leaky_column_with_a_real_check_script(
    make_agent_context, workspace
):
    write_sample_project(workspace.project_folder, leaky_column=True)
    train = pd.read_csv(workspace.project_folder / "train.csv")
    experiment_folder = workspace.experiment_folder("experiment_001")
    perfect_predictions = pd.DataFrame(
        {"example_id": train.index, "fold": train.index % 3, "probability": train["churn"] * 1.0}
    )
    predictions_path = experiment_folder / "validation_predictions.csv"
    perfect_predictions.to_csv(predictions_path, index=False)

    ledger = create_new_ledger("Predict churn.", "macro F1")
    ledger.plan.primary_metric = "macro_f1"
    ledger.runs = [
        ExperimentRun(
            experiment_id="experiment_001",
            model_family="lightgbm",
            score_mean=1.0,
            score_std=0.0,
            validation_scores=[1.0, 1.0, 1.0],
            validation_predictions_path=str(predictions_path),
            test_predictions_path=str(experiment_folder / "test_predictions.csv"),
        )
    ]
    check_script = as_python_answer(
        CHECK_SCRIPT, workspace, VALIDATION_PREDICTIONS=predictions_path
    )
    context = make_agent_context([check_script], 180)

    evaluation = run_evaluator(ledger, context, "experiment_001")

    assert evaluation.verdict == "reject"
    assert "refund_amount alone gives AUC" in evaluation.reasons[0]
    assert len(context.llm_client.received_prompts) == 1
