"""End-to-end integration tests for the LangGraph pipeline with Fake LLM and fake Coder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mlagent.agents.base import AgentContext
from mlagent.agents.coder import CoderResult
from mlagent.agents.data_analyst import ReadDraft, SummaryDraft, ThinkDraft
from mlagent.agents.data_engineer import BriefReading
from mlagent.agents.evaluator import VerdictAnswer
from mlagent.agents.feature_engineer import (
    CandidateList,
    CandidateSpec,
    FoundationDraft,
    FoundationStep,
)
from mlagent.agents.ml_lead import PlanDraft
from mlagent.agents.modeler import ModelChoice
from mlagent.agents.reviewer import ReviewDraft
from mlagent.config import AppConfig
from mlagent.environment_probe import EnvironmentInfo
from mlagent.events import EventBus
from mlagent.graph import run_pipeline
from mlagent.llm_client import LlmUnavailableError
from mlagent.memory import Workspace, create_new_ledger, load_ledger, save_ledger
from mlagent.state import FileInfo, Finding, Ledger, ProjectBrief
from tests.conftest import FakeLlmClient


def setup_fake_environment(tmp_path: Path) -> tuple[Workspace, Ledger, AgentContext]:
    """Create a minimal workspace, dataset, and agent context for pipeline tests."""
    project_folder = tmp_path / "project_data"
    project_folder.mkdir(parents=True, exist_ok=True)
    train_file = project_folder / "train.csv"
    train_file.write_text("id,feat1,feat2,target\n1,0.1,0.2,0\n2,0.3,0.4,1\n3,0.5,0.6,0\n", encoding="utf-8")
    test_file = project_folder / "test.csv"
    test_file.write_text("id,feat1,feat2\n4,0.7,0.8\n5,0.9,1.0\n", encoding="utf-8")

    workspace = Workspace(tmp_path / "workspace")
    workspace.create_folders()

    ledger = create_new_ledger(
        user_task="Predict target label from features",
        user_metrics_text="f1",
    )
    save_ledger(workspace, ledger)

    config = AppConfig()
    config.limits.max_experiment_loops = 3
    config.limits.max_eda_loops = 2

    env = EnvironmentInfo(
        os_name="Windows",
        shell_name="powershell",
        python_version="3.12.0",
        cpu_count=4,
        ram_total_gb=16.0,
        ram_free_gb=8.0,
        has_internet=True,
        can_train_on_gpu=False,
    )

    ctx = AgentContext(
        workspace=workspace,
        event_bus=EventBus(),
        code_runner=None,  # type: ignore[arg-type]
        llm_client=None,  # type: ignore[arg-type]
        environment=env,
        config=config,
    )
    return workspace, ledger, ctx


def build_canned_coder(workspace: Workspace) -> Any:
    """Return a fake run_coder function that produces valid contract outputs for each agent."""
    def fake_run_coder(
        task_text: str,
        result_tag: str,
        result_keys_text: str,
        working_folder: Path,
        script_name: str,
        caller_key: str,
        ledger: Ledger,
        agent_context: AgentContext,
    ) -> CoderResult:
        working_folder.mkdir(parents=True, exist_ok=True)
        script_path = working_folder / script_name

        if caller_key == "data_engineer":
            return CoderResult(
                succeeded=True,
                result_data={
                    "files": [
                        {
                            "path": str(workspace.project_folder / "train.csv"),
                            "role": "train",
                            "size_bytes": 1024,
                            "row_count": 100,
                            "columns": ["id", "feat1", "feat2", "target"],
                        },
                        {
                            "path": str(workspace.project_folder / "test.csv"),
                            "role": "test",
                            "size_bytes": 512,
                            "row_count": 50,
                            "columns": ["id", "feat1", "feat2"],
                        },
                    ],
                    "documents": [],
                    "peeks": {},
                    "notes": [],
                },
                script_path=script_path,
                stdout="<BRIEF_FACTS>{}</BRIEF_FACTS>",
                error_text="",
                attempts_used=1,
            )

        if caller_key == "data_analyst":
            return CoderResult(
                succeeded=True,
                result_data={
                    "answers": [
                        {"question": "target distribution", "result": "binary classification 50/50 balance"}
                    ],
                    "problems": [],
                },
                script_path=script_path,
                stdout="",
                error_text="",
                attempts_used=1,
            )

        if caller_key == "ml_lead":
            splits_file = workspace.splits_path
            splits_file.parent.mkdir(parents=True, exist_ok=True)
            import json
            splits_file.write_text(
                json.dumps({"n_splits": 2, "example_ids": [1, 2, 3], "fold_of_example": [0, 1, 0]}),
                encoding="utf-8",
            )
            return CoderResult(
                succeeded=True,
                result_data={
                    "splits_path": str(splits_file),
                    "n_splits": 2,
                    "n_examples": 3,
                    "n_holdout": 0,
                },
                script_path=script_path,
                stdout="",
                error_text="",
                attempts_used=1,
            )

        if caller_key == "feature_engineer":
            if "foundation" in task_text.lower() or "features_v1" in task_text.lower():
                feat_v1 = workspace.features_file(1)
                feat_v1.parent.mkdir(parents=True, exist_ok=True)
                feat_v1.write_text("def fit_preparation(d, s=42): return {}\ndef apply_preparation(d, s): return d\n", encoding="utf-8")
                return CoderResult(
                    succeeded=True,
                    result_data={"status": "ok"},
                    script_path=script_path,
                    stdout="",
                    error_text="",
                    attempts_used=1,
                )
            # Screening / adding features
            feat_v2 = workspace.features_file(2)
            feat_v2.parent.mkdir(parents=True, exist_ok=True)
            feat_v2.write_text("def fit_preparation(d, s=42): return {}\ndef apply_preparation(d, s): return d\n", encoding="utf-8")
            return CoderResult(
                succeeded=True,
                result_data={
                    "baseline_scores": [0.80, 0.82],
                    "candidates": [
                        {"name": "ratio_feat", "scores": [0.85, 0.86]},
                        {"name": "noise_feat", "scores": [0.79, 0.80]},
                    ],
                },
                script_path=script_path,
                stdout="",
                error_text="",
                attempts_used=1,
            )

        if caller_key == "modeler":
            val_p = working_folder / "val_predictions.csv"
            val_p.write_text("id,pred,label\n1,0,0\n2,1,1\n3,0,0\n", encoding="utf-8")
            test_p = working_folder / "test_predictions.csv"
            test_p.write_text("id,pred\n4,0\n5,1\n", encoding="utf-8")
            return CoderResult(
                succeeded=True,
                result_data={
                    "validation_scores": [0.83, 0.85],
                    "holdout_score": 0.84,
                    "extra_scores": {},
                    "validation_predictions_path": str(val_p),
                    "test_predictions_path": str(test_p),
                    "model_path": "",
                    "notes": "LightGBM baseline",
                },
                script_path=script_path,
                stdout="",
                error_text="",
                attempts_used=1,
            )

        if caller_key == "evaluator":
            return CoderResult(
                succeeded=True,
                result_data={
                    "recomputed_score": 0.84,
                    "recomputed_holdout_score": 0.84,
                    "leakage_flags": [],
                    "train_validation_overlap": 0,
                    "notes": ["Checks passed"],
                },
                script_path=script_path,
                stdout="",
                error_text="",
                attempts_used=1,
            )

        if caller_key == "tuner":
            val_p = working_folder / "val_predictions.csv"
            val_p.write_text("id,pred,label\n1,0,0\n2,1,1\n3,0,0\n", encoding="utf-8")
            test_p = working_folder / "test_predictions.csv"
            test_p.write_text("id,pred\n4,0\n5,1\n", encoding="utf-8")
            return CoderResult(
                succeeded=True,
                result_data={
                    "validation_scores": [0.86, 0.87],
                    "holdout_score": 0.865,
                    "extra_scores": {},
                    "validation_predictions_path": str(val_p),
                    "test_predictions_path": str(test_p),
                    "model_path": "",
                    "model_params": {"max_depth": 4},
                    "trials_run": 5,
                    "notes": "Tuned LightGBM",
                },
                script_path=script_path,
                stdout="",
                error_text="",
                attempts_used=1,
            )

        if caller_key == "submitter":
            blended = working_folder / "blended_predictions.csv"
            blended.write_text("id,pred\n4,0\n5,1\n", encoding="utf-8")
            return CoderResult(
                succeeded=True,
                result_data={
                    "validation_scores": [0.865, 0.875],
                    "test_predictions_path": str(blended),
                    "method": "Simple average",
                },
                script_path=script_path,
                stdout="",
                error_text="",
                attempts_used=1,
            )

        return CoderResult(True, {}, script_path, "", "", 1)

    return fake_run_coder


def build_pipeline_fake_llm() -> FakeLlmClient:
    """Build FakeLlmClient with responses tailored for an entire pipeline progression."""
    review_call_count = 0

    def mock_reviewer_draft(prompt: str) -> ReviewDraft:
        nonlocal review_call_count
        review_call_count += 1
        if review_call_count == 1:
            # Round 0 reviewed -> next focus is features
            return ReviewDraft(
                assessment="Baseline model trained successfully.",
                lesson="Tree model performs well on tabular data.",
                focus="features",
                reason="Engineer ratios and interactions.",
                instructions=["Create interaction features."],
            )
        elif review_call_count == 2:
            # Round 1 reviewed -> next focus is tune
            return ReviewDraft(
                assessment="Features gave clear gain.",
                lesson="Ratio feature improved validation F1.",
                focus="tune",
                reason="Tune tree depth and learning rate.",
                instructions=["Tune LightGBM hyperparameters."],
            )
        else:
            # Round 2 reviewed -> finish
            return ReviewDraft(
                assessment="Tuning saturated gains.",
                lesson="Optimal parameters found.",
                focus="finish",
                reason="Target performance reached.",
                instructions=[],
            )

    responses: dict[str, Any] = {
        "Data Engineer": BriefReading(
            brief=ProjectBrief(
                summary="Binary classification of customer churn.",
                modality="tabular",
                task_family="classification",
                target="target",
                id_column="id",
                primary_metric="f1",
                higher_is_better=True,
                deliverable_format="predictions.csv with id and pred columns",
                files=[
                    FileInfo(path="train.csv", role="train", row_count=100, columns=["id", "feat1", "feat2", "target"]),
                    FileInfo(path="test.csv", role="test", row_count=50, columns=["id", "feat1", "feat2"]),
                ],
            ),
            done=True,
            next_tasks=[],
        ),
        "Data Analyst": lambda prompt, schema=None: (
            SummaryDraft(
                summary="Clean tabular dataset with 2 continuous features.",
                issues=[],
                feature_ideas=[{"description": "Ratio feature", "finding_id": "finding_001"}],
                leakage_risks=[],
                validation_hints=["stratified_kfold"],
                model_hints=["lightgbm"],
            )
            if (
                schema is SummaryDraft
                or "closing the exploration" in prompt.lower()
                or "summarizing" in prompt.lower()
                or "handover" in prompt.lower()
            )
            else (
                ReadDraft(
                    findings=[
                        Finding(
                            statement="Target classes are balanced 50/50.",
                            evidence="1:1 ratio in train.csv",
                            implication="Stratified K-fold CV is appropriate.",
                            priority="high",
                        )
                    ],
                    open_questions=[],
                )
                if (
                    schema is ReadDraft
                    or "answered your questions" in prompt.lower()
                    or "reading step" in prompt.lower()
                )
                else ThinkDraft(
                    thinking="Initial exploration of data distributions.",
                    open_questions=["Check target balance", "Check missing values"],
                    questions_to_ask=["What is target balance?"],
                    done=False if "round 1" in prompt.lower() else True,
                )
            )
        ),
        "ML Lead": PlanDraft(
            thinking="Standard supervised classification setup.",
            validation_scheme="stratified_kfold",
            validation_reason="Class balance preservation across splits.",
            baseline_description="LightGBM classifier with default parameters.",
            use_gpu=False,
            risks=["Overfitting on small dataset"],
            playbook="Start with tree ensemble, test candidate features.",
        ),
        "Feature Engineer": lambda prompt, schema=None: (
            CandidateList(
                thinking="Propose ratio feature targeting classification error.",
                candidates=[
                    CandidateSpec(
                        name="ratio_feat",
                        family="ratio",
                        reason="Feature ratio separates classes well.",
                        finding_id="finding_001",
                        idea="feat1 / (feat2 + 1e-5)",
                    ),
                    CandidateSpec(
                        name="noise_feat",
                        family="noise",
                        reason="Testing noise rejection.",
                        finding_id="finding_001",
                        idea="feat1 * 0.0",
                    ),
                ],
            )
            if (
                schema is CandidateList
                or "propose" in prompt.lower()
                or "candidates" in prompt.lower()
            )
            else FoundationDraft(
                thinking="Prepare numeric scaling and null handling.",
                steps=[
                    FoundationStep(name="impute_scale", what_to_do="StandardScale features", why="baseline stability")
                ],
            )
        ),
        "Modeler": ModelChoice(
            thinking="LightGBM classifier on current feature set.",
            model_family="lightgbm",
            model_params={"n_estimators": 50, "learning_rate": 0.1},
            reason="Fast gradient boosting baseline",
        ),
        "Evaluator": VerdictAnswer(
            thinking="Recomputed score matches script output perfectly. No leakage detected.",
            reasons=["Score difference within tolerance", "Zero train/val overlap"],
            verdict="approve",
        ),
        "Reviewer": lambda prompt: (
            (
                "# Technical Report: Customer Churn Prediction\n\n"
                "## In brief\n"
                "LightGBM baseline and tuned models trained on 2-fold CV achieved 0.8650 F1 score.\n\n"
                "## Key numbers\n"
                "- Best F1 score: 0.8650\n"
                "- Folds: 2\n"
                "- Total runs: 3\n\n"
                "## What the data showed\n"
                "Balanced target classes and strong ratio feature importance.\n\n"
                "## What the team did\n"
                "Explored distributions, trained baseline, screened features, and tuned parameters.\n\n"
                "## The final predictions\n"
                "Predictions saved to predictions.csv with id and pred columns.\n"
            )
            if "technical report" in prompt.lower() or "sections" in prompt.lower()
            else mock_reviewer_draft(prompt)
        ),
    }
    return FakeLlmClient(responses=responses)


@pytest.mark.integration
def test_full_pipeline_executes_all_agents_and_verifies_invariants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full graph executes every agent cleanly, terminating with valid artifacts and unique IDs."""
    workspace, initial_ledger, ctx = setup_fake_environment(tmp_path)
    fake_coder = build_canned_coder(workspace)
    fake_llm = build_pipeline_fake_llm()

    # Monkeypatch run_coder in all agent modules to route to canned coder
    monkeypatch.setattr("mlagent.agents.data_engineer.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.data_analyst.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.ml_lead.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.feature_engineer.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.modeler.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.evaluator.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.tuner.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.submitter.run_coder", fake_coder)

    ctx.llm_client = fake_llm

    # Run the entire pipeline
    final_ledger = run_pipeline(initial_ledger, ctx)

    # 1. Pipeline termination & status
    assert final_ledger.status == "done"
    assert final_ledger.phase == "done"
    assert final_ledger.final is not None
    assert final_ledger.final.headline != ""
    assert final_ledger.final.report_path != ""

    # 2. Report verification: verify report exists and has required sections
    report_file = Path(final_ledger.final.report_path)
    assert report_file.is_file()
    report_content = report_file.read_text(encoding="utf-8")
    for section in ["## In brief", "## Key numbers", "## What the data showed", "## What the team did", "## The final predictions"]:
        assert section in report_content, f"Missing section '{section}' in report.md"

    # 3. User invariant: Every Finding has non-empty unique id
    findings = final_ledger.eda.findings
    assert len(findings) > 0
    finding_ids = [f.finding_id for f in findings]
    assert all(bool(fid) for fid in finding_ids)
    assert len(set(finding_ids)) == len(finding_ids), "Finding IDs must be unique"

    # 4. User invariant: Every FeatureRecord has non-empty unique id
    features = final_ledger.features
    assert len(features) > 0
    feature_ids = [f.feature_id for f in features]
    assert all(bool(fid) for fid in feature_ids)
    assert len(set(feature_ids)) == len(feature_ids), "FeatureRecord IDs must be unique"

    # 5. Non-crashed runs all have verdicts
    for run in final_ledger.runs:
        if not run.crashed:
            assert run.verdict in ("approve", "reject"), f"Run {run.experiment_id} missing verdict"

    # 6. User invariant: Every agent ran at least once
    called_agents = {c["agent"] for c in fake_llm.calls}
    expected_agents = {"Data Engineer", "Data Analyst", "ML Lead", "Feature Engineer", "Modeler", "Evaluator", "Reviewer"}
    missing = expected_agents - called_agents
    assert not missing, f"Expected all agents to be called, missing: {missing}"


@pytest.mark.integration
def test_pipeline_resumes_without_redoing_finished_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pipeline interrupted mid-run can be resumed from saved ledger without repeating finished work."""
    workspace, initial_ledger, ctx = setup_fake_environment(tmp_path)
    fake_coder = build_canned_coder(workspace)
    fake_llm = build_pipeline_fake_llm()

    monkeypatch.setattr("mlagent.agents.data_engineer.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.data_analyst.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.ml_lead.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.feature_engineer.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.modeler.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.evaluator.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.tuner.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.submitter.run_coder", fake_coder)

    ctx.llm_client = fake_llm

    # Simulate interruption by raising during the first call to Feature Engineer
    fe_call_count = 0
    original_ask_structured = fake_llm.ask_structured

    def interrupting_ask_structured(agent_name: str, prompt: str, schema: type[Any]) -> Any:
        nonlocal fe_call_count
        if agent_name == "Feature Engineer":
            fe_call_count += 1
            if fe_call_count == 1:
                raise RuntimeError("Simulated mid-pipeline interruption!")
        return original_ask_structured(agent_name, prompt, schema)

    fake_llm.ask_structured = interrupting_ask_structured  # type: ignore[assignment]

    # First run: should crash and fail at feature_engineer
    with pytest.raises(RuntimeError, match="Simulated mid-pipeline interruption"):
        run_pipeline(initial_ledger, ctx)

    # Inspect saved ledger
    saved_ledger = load_ledger(workspace)
    # data_engineer and ml_lead should be finished
    assert saved_ledger.brief.task_family == "classification"
    assert saved_ledger.plan.validation_scheme == "stratified_kfold"
    initial_call_count = len(fake_llm.calls)

    # Restore normal operation and resume pipeline
    fake_llm.ask_structured = original_ask_structured  # type: ignore[assignment]
    resumed_ledger = run_pipeline(saved_ledger, ctx)

    assert resumed_ledger.status == "done"
    assert resumed_ledger.phase == "done"

    # Verify that data_engineer and ml_lead were NOT re-executed after resume
    resumed_calls = fake_llm.calls[initial_call_count:]
    resumed_agents = [c["agent"] for c in resumed_calls]
    assert "Data Engineer" not in resumed_agents
    assert "ML Lead" not in resumed_agents


@pytest.mark.integration
def test_pipeline_failure_recovery_diverts_to_submit_when_approved_runs_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When an unrecoverable failure (LlmUnavailableError or AgentFailedError) occurs, approved work still submits."""
    workspace, initial_ledger, ctx = setup_fake_environment(tmp_path)
    fake_coder = build_canned_coder(workspace)
    fake_llm = build_pipeline_fake_llm()

    monkeypatch.setattr("mlagent.agents.data_engineer.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.data_analyst.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.ml_lead.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.feature_engineer.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.modeler.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.evaluator.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.tuner.run_coder", fake_coder)
    monkeypatch.setattr("mlagent.agents.submitter.run_coder", fake_coder)

    ctx.llm_client = fake_llm

    # Trigger LlmUnavailableError after 1 approved run is recorded (during Reviewer call)
    original_ask_structured = fake_llm.ask_structured

    def failing_ask_structured(agent_name: str, prompt: str, schema: type[Any]) -> Any:
        if agent_name == "Reviewer":
            raise LlmUnavailableError("Ollama service unreachable!")
        return original_ask_structured(agent_name, prompt, schema)

    fake_llm.ask_structured = failing_ask_structured  # type: ignore[assignment]

    ledger = run_pipeline(initial_ledger, ctx)

    # Ledger should record stop reason 'llm_unavailable' and proceed to submit/report
    assert ledger.stop_reason == "llm_unavailable"
    assert ledger.final is not None
    assert ledger.final.predictions_path != ""
    assert Path(ledger.final.report_path).is_file()
