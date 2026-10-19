"""Hyperparameter Tuner agent: systematic search to close generalization gaps.

Conducts targeted hyperparameter searches on competitive Champion models.
Tailors search space to the train-vs-CV gap: prioritizing regularization when
overfitting, or increasing model capacity when underfitting.
"""

from __future__ import annotations

import json

# Agent execution context and exceptions
from mlagent.agents.base import AgentContext, AgentFailedError

# Script execution sub-agent
from mlagent.agents.coder import run_coder

# Modeler diagnostic parsing
from mlagent.agents.modeler import _parse_diagnostic_report

# Master graph state models
from mlagent.graph.state import (
    ErrorSlice,
    ExperimentRun,
    Ledger,
    TuningPlan,
)

# Event logging and notifications
from mlagent.helper.events import EventType

# Script rules text from contract
from mlagent.helper.script_contract import MODEL_SCRIPT_RULES_TEXT

# Prompt templates and role definitions
from mlagent.memory.assembler import MemoryAssembler
from mlagent.memory.cards import ExperimentRunCard
from mlagent.memory.store import MemoryStore
from mlagent.prompts.tuner import TUNER_ROLE

AGENT_KEY = "tuner"
AGENT_NAME = "Tuner"
RESULT_TAG = "TUNER_RESULT"

RESULT_KEYS = """\
- "best_params": dict of winning hyperparameters
- "train_score_mean": float
- "cv_fold_scores": list of floats
- "cv_mean": float
- "cv_std": float
- "holdout_score": float or null
- "secondary_metrics": dict mapping metric name to float
- "top_feature_importances": dict mapping feature name to importance float
- "worst_error_slices": list of objects {"slice_condition": str, "sample_count": int, "slice_metric_score": float, "degradation_vs_overall": float}
- "diagnostic_report": dict with task family metrics
- "val_predictions_path": "val_predictions.csv"
- "test_predictions_path": "test_predictions.csv"
- "model_path": "model.pkl"
- "trials_completed": int"""


def run_tuner(ledger: Ledger, agent_context: AgentContext) -> TuningPlan:
    # Plan and execute hyperparameter optimization for Champion via MemoryAssembler
    event_bus = agent_context.event_bus
    champion = ledger.champion_run()
    if not champion or champion.cv_mean is None:
        # Fall back to latest approved run if available
        approved = [r for r in ledger.runs if r.approved]
        if approved:
            champion = approved[-1]
        else:
            raise AgentFailedError("Cannot tune: no approved Champion run exists.")

    exp_id = ledger.next_experiment_id()
    event_bus.emit(
        EventType.AGENT_STARTED,
        AGENT_NAME,
        f"tuning hyperparameters of Champion {champion.experiment_id} ({champion.model_family}) in {exp_id}",
        ledger=ledger,
    )

    store = MemoryStore(agent_context.workspace.root)
    assembler = MemoryAssembler(store)

    task_text = (
        f"Perform targeted hyperparameter optimization for Champion {champion.experiment_id} ({champion.model_family}) "
        f"on features v{champion.feature_version}. Optimize to close generalization gaps on primary metric."
    )
    prompt = assembler.build_prompt(
        agent_name="tuner",
        ledger=ledger,
        task=task_text,
        schema_hint=str(TuningPlan.model_json_schema()),
    )

    try:
        plan = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=TuningPlan,
            system_message=TUNER_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(f"Tuning planning failed for {exp_id}: {exc}") from exc

    # Set up experiment directory
    exp_folder = agent_context.workspace.artifacts_folder / "experiments" / exp_id
    exp_folder.mkdir(parents=True, exist_ok=True)
    splits_file = agent_context.workspace.artifacts_folder / "splits.json"
    features_script = agent_context.workspace.artifacts_folder / "features" / f"features_v{champion.feature_version}.py"
    project_folder = agent_context.workspace.project_folder

    primary_metric = ledger.brief.primary_metric
    higher_is_better = ledger.brief.higher_is_better

    target_col = ledger.brief.target_column or "target"
    id_cols = ledger.brief.id_columns
    id_col = id_cols[0] if id_cols else "Id"

    coder_task = (
        f"Tune hyperparameters for experiment {exp_id} on base model {champion.model_family}.\n\n"
        f"Working Directory: {exp_folder}\n"
        f"Splits JSON: {splits_file}\n"
        f"Features Module: {features_script}\n"
        f"Project Data: {project_folder}\n"
        f"Target Column: {target_col}\n"
        f"ID Column: {id_col}\n"
        f"Primary Metric: {primary_metric} (higher is better: {higher_is_better})\n"
        f"Search Ranges: {json.dumps([r.model_dump() for r in plan.hyperparameter_ranges], indent=2)}\n"
        f"Trial Budget: {plan.search_budget_trials}\n"
        f"Early Stopping Rounds: {plan.early_stopping_rounds}\n\n"
        f"Rules:\n{MODEL_SCRIPT_RULES_TEXT}\n\n"
        "CRITICAL CODE PATTERNS (FOLLOW EXACTLY):\n"
        "1. DYNAMIC FEATURES IMPORT:\n"
        "   import sys, os\n"
        f"   features_dir = r'{features_script.parent}'\n"
        "   if features_dir not in sys.path:\n"
        "       sys.path.insert(0, features_dir)\n"
        f"   from {features_script.stem} import fit_preparation, apply_preparation\n\n"
        "2. LOAD SPLITS & SLICE FOLDS:\n"
        f"   with open(r'{splits_file}', 'r', encoding='utf-8') as f:\n"
        "       splits_data = json.load(f)\n"
        "   example_ids = np.array(splits_data['example_ids'])\n"
        "   fold_of_example = np.array(splits_data['fold_of_example'])\n"
        "   n_splits = int(splits_data['n_splits'])\n"
        "   # Slicing folds:\n"
        "   for fold in range(n_splits):\n"
        "       val_idx = example_ids[fold_of_example == fold]\n"
        "       train_idx = example_ids[(fold_of_example != fold) & (fold_of_example != -1)]\n"
        "       train_fold = train_df.iloc[train_idx].reset_index(drop=True)\n"
        "       val_fold = train_df.iloc[val_idx].reset_index(drop=True)\n"
        "       prep_state = fit_preparation(train_fold)\n"
        "       X_tr = apply_preparation(train_fold, prep_state)\n"
        f"       y_tr = train_fold['{target_col}']\n"
        "       X_va = apply_preparation(val_fold, prep_state)\n"
        f"       y_va = val_fold['{target_col}']\n\n"
        "3. REQUIRED FILES TO WRITE IN WORKING DIRECTORY:\n"
        f"   - val_predictions.csv (columns: ['{id_col}', 'fold', 'prediction'])\n"
        f"   - test_predictions.csv (columns: ['{id_col}', 'prediction'])\n"
        "   - model.pkl (saved via joblib.dump)\n"
        "4. Output JSON tagged with <TUNER_RESULT>."
    )

    coder_result = run_coder(
        task_text=coder_task,
        result_tag=RESULT_TAG,
        result_keys_text=RESULT_KEYS,
        working_folder=exp_folder,
        script_name="tune_model.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )

    hyp = ledger.hypotheses[-1] if ledger.hypotheses else None
    run_record = ExperimentRun(
        experiment_id=exp_id,
        iteration=ledger.iteration,
        hypothesis_id=hyp.hypothesis_id if hyp else "",
        route_type="tune_champion",
        model_family=f"{champion.model_family}_tuned",
        complexity_tier=champion.complexity_tier,
        model_params=plan.baseline_hyperparameters,
        feature_version=champion.feature_version,
        script_path=str(exp_folder / "tune_model.py"),
        artifact_dir=str(exp_folder),
        model_path=str(exp_folder / "model.pkl"),
        summary=plan.cli_summary,
    )

    if not coder_result.succeeded:
        run_record.error_traceback = coder_result.error_text
        ledger.runs.append(run_record)
        event_bus.emit(
            EventType.WARNING,
            AGENT_NAME,
            f"Tuning experiment {exp_id} failed: {coder_result.error_text[:120]}",
        )
        return plan

    data = coder_result.result_data
    run_record.model_params = data.get("best_params", run_record.model_params)
    run_record.train_score_mean = data.get("train_score_mean")
    run_record.cv_fold_scores = [float(s) for s in data.get("cv_fold_scores", [])]
    run_record.cv_mean = float(data.get("cv_mean")) if data.get("cv_mean") is not None else None
    run_record.cv_std = float(data.get("cv_std")) if data.get("cv_std") is not None else None
    run_record.holdout_score = float(data.get("holdout_score")) if data.get("holdout_score") is not None else None
    run_record.secondary_metrics = {k: float(v) for k, v in data.get("secondary_metrics", {}).items()}
    run_record.top_feature_importances = {k: float(v) for k, v in data.get("top_feature_importances", {}).items()}

    # Extract error slices
    raw_slices = data.get("worst_error_slices", [])
    run_record.worst_error_slices = [
        ErrorSlice(
            slice_condition=s.get("slice_condition", "unknown"),
            sample_count=int(s.get("sample_count", 0)),
            slice_metric_score=float(s.get("slice_metric_score", 0.0)),
            degradation_vs_overall=float(s.get("degradation_vs_overall", 0.0)),
        )
        for s in raw_slices
        if isinstance(s, dict)
    ]

    # Parse task-family specific diagnostic report
    raw_diag = data.get("diagnostic_report", {})
    if isinstance(raw_diag, dict):
        diag_attrs = _parse_diagnostic_report(ledger.brief.task_family, raw_diag)
        for attr, val in diag_attrs.items():
            setattr(run_record, attr, val)

    ledger.runs.append(run_record)
    ledger.last_tuned_iteration = ledger.iteration

    # Append ExperimentRunCard to MemoryStore
    cv_val = run_record.cv_mean if run_record.cv_mean is not None else 0.0
    cv_err = run_record.cv_std if run_record.cv_std is not None else 0.0
    holdout_val = run_record.holdout_score if run_record.holdout_score is not None else 0.0
    summary_text = (
        f"Run {exp_id} (Tuned): {run_record.model_family} (Tier {champion.complexity_tier}, FV {champion.feature_version}). "
        f"CV={cv_val:.4f}±{cv_err:.4f}. Holdout={holdout_val:.4f}. Status={run_record.overfitting_status}."
    )
    run_card = ExperimentRunCard(
        tag="experiment_run",
        card_id=exp_id,
        owner="tuner",
        iteration=ledger.iteration,
        memory_summary=summary_text,
        payload={
            "experiment_id": exp_id,
            "model_family": run_record.model_family,
            "tier": champion.complexity_tier,
            "feature_version": champion.feature_version,
            "cv_mean": run_record.cv_mean,
            "cv_std": run_record.cv_std,
            "holdout_score": run_record.holdout_score,
            "train_score_mean": run_record.train_score_mean,
            "overfitting_status": run_record.overfitting_status,
            "feature_importances": dict(list(run_record.top_feature_importances.items())[:5]),
            "script_path": str(exp_folder / "tune_model.py"),
            "artifact_dir": str(exp_folder),
        },
        artifact_paths=[str(exp_folder / "tune_model.py"), str(exp_folder)],
        priority="high",
    )
    try:
        store.append_card(run_card)
    except FileExistsError:
        pass

    score_str = f"CV={run_record.cv_mean:.4f}" if run_record.cv_mean is not None else "crashed"
    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"Tuned {exp_id} ({score_str}): {plan.cli_summary}",
    )
    return plan