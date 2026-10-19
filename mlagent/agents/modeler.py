"""Modeler agent: progressive complexity model training across validation folds.

Executes candidate model training matching designated complexity tiers (Tier 1 baseline,
Tier 2 ensemble/regularized, Tier 3 GBDT/Neural). Enforces strict fold isolation, saves
out-of-fold and test predictions, and extracts task-specific diagnostic reports.
"""

from __future__ import annotations

import json
from typing import Any

# Agent execution context and exceptions
from mlagent.agents.base import AgentContext, AgentFailedError

# Script execution sub-agent
from mlagent.agents.coder import run_coder

# Master graph state models
from mlagent.graph.state import (
    AnomalyReport,
    ClassificationReport,
    ClusteringReport,
    ErrorSlice,
    ExperimentRun,
    ForecastingReport,
    Ledger,
    ModelPlan,
    MultilabelReport,
    RankingReport,
    RegressionReport,
)

# Script rules text from contract
from mlagent.helper.deliverable_checks import find_sample_submission

# Event logging and notifications
from mlagent.helper.events import EventType
from mlagent.helper.script_contract import MODEL_SCRIPT_RULES_TEXT

# Prompt templates and role definitions
from mlagent.memory.assembler import MemoryAssembler
from mlagent.memory.cards import ExperimentRunCard
from mlagent.memory.store import MemoryStore
from mlagent.prompts.modeler import MODELER_ROLE

AGENT_KEY = "modeler"
AGENT_NAME = "Modeler"
RESULT_TAG = "MODEL_RESULT"

RESULT_KEYS = """\
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
- "model_path": "model.pkl" or "model.bin"
- "notes": str"""


def _parse_diagnostic_report(task_family: str, raw_report: dict[str, Any]) -> dict[str, Any]:
    # Parse raw diagnostic dictionary into appropriate task family report model
    out: dict[str, Any] = {}
    if "classification" in task_family:
        if "multilabel" in task_family:
            out["multilabel_report"] = MultilabelReport(
                hamming_loss=float(raw_report.get("hamming_loss", 0.0)),
                micro_f1=float(raw_report.get("micro_f1", 0.0)),
                macro_f1=float(raw_report.get("macro_f1", 0.0)),
            )
        else:
            out["classification_report"] = ClassificationReport(
                accuracy=float(raw_report.get("accuracy", 0.0)),
                balanced_accuracy=float(raw_report.get("balanced_accuracy", 0.0)),
                f1_macro=float(raw_report.get("f1_macro", 0.0)),
                roc_auc=raw_report.get("roc_auc"),
                log_loss=raw_report.get("log_loss"),
                confusion_matrix=raw_report.get("confusion_matrix", []),
                per_class_f1=raw_report.get("per_class_f1", {}),
            )
    elif "regression" in task_family:
        out["regression_report"] = RegressionReport(
            rmse=float(raw_report.get("rmse", 0.0)),
            mae=float(raw_report.get("mae", 0.0)),
            r2=float(raw_report.get("r2", 0.0)),
            median_absolute_error=float(raw_report.get("median_absolute_error", 0.0)),
            max_error=float(raw_report.get("max_error", 0.0)),
        )
    elif "clustering" in task_family:
        out["clustering_report"] = ClusteringReport(
            silhouette_score=float(raw_report.get("silhouette_score", 0.0)),
            calinski_harabasz=float(raw_report.get("calinski_harabasz", 0.0)),
            davies_bouldin=float(raw_report.get("davies_bouldin", 0.0)),
        )
    elif "ranking" in task_family:
        out["ranking_report"] = RankingReport(
            ndcg_at_5=float(raw_report.get("ndcg_at_5", 0.0)),
            ndcg_at_10=float(raw_report.get("ndcg_at_10", 0.0)),
            map_at_10=float(raw_report.get("map_at_10", 0.0)),
            mean_reciprocal_rank=float(raw_report.get("mean_reciprocal_rank", 0.0)),
        )
    elif "forecasting" in task_family or "time_series" in task_family:
        out["forecasting_report"] = ForecastingReport(
            horizon_mae=raw_report.get("horizon_mae", []),
            wape=float(raw_report.get("wape", 0.0)),
            smape=float(raw_report.get("smape", 0.0)),
            prediction_interval_coverage_80=raw_report.get("prediction_interval_coverage_80"),
        )
    elif "anomaly" in task_family:
        out["anomaly_report"] = AnomalyReport(
            precision_at_k=float(raw_report.get("precision_at_k", 0.0)),
            roc_auc=raw_report.get("roc_auc"),
            pr_auc=raw_report.get("pr_auc"),
            detected_anomaly_ratio=float(raw_report.get("detected_anomaly_ratio", 0.0)),
        )
    return out


def run_modeler(ledger: Ledger, agent_context: AgentContext) -> ModelPlan:
    # Plan and train model for current hypothesis
    event_bus = agent_context.event_bus
    exp_id = ledger.next_experiment_id()
    hyp = ledger.hypotheses[-1] if ledger.hypotheses else None

    target_family = hyp.target_model_family if hyp and hyp.target_model_family else "lightgbm"
    complexity_tier = hyp.model_complexity_tier if hyp else "tier_1_simple_baseline"
    feature_version = (
        ledger.latest_candidate_feature_version or ledger.active_base_feature_version()
    )

    event_bus.emit(
        EventType.AGENT_STARTED,
        AGENT_NAME,
        f"training {target_family} ({complexity_tier}) on features v{feature_version} for {exp_id}",
        ledger=ledger,
    )

    primary_metric = ledger.brief.primary_metric
    higher_is_better = ledger.brief.higher_is_better
    store = MemoryStore(agent_context.workspace.root)
    assembler = MemoryAssembler(store)

    task_text = (
        f"Train model experiment {exp_id} using model family '{target_family}' with complexity tier '{complexity_tier}' "
        f"on feature version v{feature_version}. Optimize for primary metric '{primary_metric}'."
    )
    prompt = assembler.build_prompt(
        agent_name="modeler",
        ledger=ledger,
        task=task_text,
        schema_hint=str(ModelPlan.model_json_schema()),
    )

    try:
        plan = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=ModelPlan,
            system_message=MODELER_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(f"Model planning failed for {exp_id}: {exc}") from exc

    # Set up experiment working directory
    exp_folder = agent_context.workspace.artifacts_folder / "experiments" / exp_id
    exp_folder.mkdir(parents=True, exist_ok=True)
    splits_file = agent_context.workspace.artifacts_folder / "splits.json"
    features_script = (
        agent_context.workspace.artifacts_folder / "features" / f"features_v{feature_version}.py"
    )
    project_folder = agent_context.workspace.project_folder

    target_col = ledger.brief.target_column or "target"
    id_cols = ledger.brief.id_columns
    id_col = id_cols[0] if id_cols else "Id"

    # Align with sample submission if present
    sample_path = find_sample_submission(ledger.brief, project_folder)
    if sample_path is not None:
        try:
            import pandas as pd

            sample_df = pd.read_csv(sample_path, nrows=2)
            if len(sample_df.columns) >= 2:
                id_col = str(sample_df.columns[0])
                target_col = str(sample_df.columns[1])
        except Exception:
            pass

    coder_task = (
        f"Train model experiment {exp_id} using {plan.model_family} on feature version v{feature_version}.\n\n"
        f"Working Directory: {exp_folder}\n"
        f"Splits JSON: {splits_file}\n"
        f"Features Module: {features_script}\n"
        f"Project Data: {project_folder}\n"
        f"Target Column: {target_col}\n"
        f"ID Column: {id_col}\n"
        f"Primary Metric: {primary_metric} (higher is better: {higher_is_better})\n"
        f"Hyperparameters: {json.dumps(plan.hyperparameters, indent=2)}\n"
        f"Post-processing: {plan.post_processing_strategy}\n\n"
        f"Rules:\n{MODEL_SCRIPT_RULES_TEXT}\n\n"
        "CRITICAL CODE PATTERNS (FOLLOW EXACTLY):\n"
        "1. DYNAMIC FEATURES IMPORT:\n"
        "   import sys, os\n"
        f"   features_dir = r'{features_script.parent}'\n"
        "   if features_dir not in sys.path:\n"
        "       sys.path.insert(0, features_dir)\n"
        f"   from {features_script.stem} import fit_preparation, apply_preparation\n\n"
        "2. LOAD SPLITS, PREPARE DATA & ENSURE NUMERIC FEATURES (PREVENTS STRING & MISMATCH ERRORS):\n"
        f"   with open(r'{splits_file}', 'r', encoding='utf-8') as f:\n"
        "       splits_data = json.load(f)\n"
        "   example_ids = np.array(splits_data['example_ids'])\n"
        "   fold_of_example = np.array(splits_data['fold_of_example'])\n"
        "   n_splits = int(splits_data['n_splits'])\n"
        f"   drop_cols = ['{target_col}', '{id_col}']\n"
        f"   is_linear = any(k in '{plan.model_family}'.lower() for k in ['ridge', 'linear', 'logistic', 'elastic', 'mlp', 'svm', 'knn'])\n"
        "   # Slicing folds:\n"
        "   for fold in range(n_splits):\n"
        "       val_idx = example_ids[fold_of_example == fold]\n"
        "       train_idx = example_ids[(fold_of_example != fold) & (fold_of_example != -1)]\n"
        "       train_fold = train_df.iloc[train_idx].reset_index(drop=True)\n"
        "       val_fold = train_df.iloc[val_idx].reset_index(drop=True)\n"
        f"       y_tr = train_fold['{target_col}']\n"
        f"       y_va = val_fold['{target_col}']\n"
        "       prep_state = fit_preparation(train_fold)\n"
        "       X_tr = apply_preparation(train_fold, prep_state)\n"
        "       X_va = apply_preparation(val_fold, prep_state)\n"
        "       # DROP TARGET & ID COLUMNS (PREVENTS TARGET LEAKAGE):\n"
        "       X_tr = X_tr.drop(columns=[c for c in drop_cols if c in X_tr.columns], errors='ignore')\n"
        "       X_va = X_va.drop(columns=[c for c in drop_cols if c in X_va.columns], errors='ignore')\n"
        "       # CONVERT CATEGORICAL STRINGS TO NUMERIC (PREVENTS 'could not convert string to float' ERRORS):\n"
        "       cat_cols = X_tr.select_dtypes(include=['object', 'category', 'string']).columns.tolist()\n"
        "       if cat_cols:\n"
        "           X_tr = pd.get_dummies(X_tr, columns=cat_cols, drop_first=True, dummy_na=False)\n"
        "           X_va = pd.get_dummies(X_va, columns=cat_cols, drop_first=True, dummy_na=False)\n"
        "       # UNCONDITIONALLY ALIGN VALIDATION FEATURES TO TRAINING COLUMNS (PREVENTS FEATURE NAME MISMATCH):\n"
        "       X_va = X_va.reindex(columns=X_tr.columns, fill_value=0.0)\n"
        "       # Clean NaNs and Infs safely on numeric columns:\n"
        "       X_tr = X_tr.fillna(0.0).replace([np.inf, -np.inf], 0.0)\n"
        "       X_va = X_va.fillna(0.0).replace([np.inf, -np.inf], 0.0)\n"
        "       if is_linear:\n"
        "           from sklearn.preprocessing import StandardScaler\n"
        "           scaler = StandardScaler()\n"
        "           X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)\n"
        "           X_va = pd.DataFrame(scaler.transform(X_va), columns=X_tr.columns)\n\n"
        "3. FINAL MODEL & TEST PREDICTIONS (APPLY IDENTICAL CLEANING):\n"
        "   full_train_idx = example_ids[fold_of_example != -1]\n"
        "   full_train_df = train_df.iloc[full_train_idx].reset_index(drop=True)\n"
        f"   y_full = full_train_df['{target_col}']\n"
        "   final_prep_state = fit_preparation(full_train_df)\n"
        "   X_full = apply_preparation(full_train_df, final_prep_state)\n"
        "   X_test = apply_preparation(test_df, final_prep_state)\n"
        "   X_full = X_full.drop(columns=[c for c in drop_cols if c in X_full.columns], errors='ignore')\n"
        "   X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns], errors='ignore')\n"
        "   cat_cols_full = X_full.select_dtypes(include=['object', 'category', 'string']).columns.tolist()\n"
        "   if cat_cols_full:\n"
        "       X_full = pd.get_dummies(X_full, columns=cat_cols_full, drop_first=True, dummy_na=False)\n"
        "       X_test = pd.get_dummies(X_test, columns=cat_cols_full, drop_first=True, dummy_na=False)\n"
        "   # UNCONDITIONALLY ALIGN TEST FEATURES TO FULL TRAIN COLUMNS:\n"
        "   X_test = X_test.reindex(columns=X_full.columns, fill_value=0.0)\n"
        "   X_full = X_full.fillna(0.0).replace([np.inf, -np.inf], 0.0)\n"
        "   X_test = X_test.fillna(0.0).replace([np.inf, -np.inf], 0.0)\n"
        "   if is_linear:\n"
        "       from sklearn.preprocessing import StandardScaler\n"
        "       scaler_full = StandardScaler()\n"
        "       X_full = pd.DataFrame(scaler_full.fit_transform(X_full), columns=X_full.columns)\n"
        "       X_test = pd.DataFrame(scaler_full.transform(X_test), columns=X_full.columns)\n\n"
        "4. HOLDOUT EVALUATION (MANDATORY IF HOLDOUT EXAMPLES EXIST):\n"
        "   holdout_idx = example_ids[fold_of_example == -1]\n"
        "   if len(holdout_idx) > 0:\n"
        "       holdout_fold = train_df.iloc[holdout_idx].reset_index(drop=True)\n"
        f"       y_holdout = holdout_fold['{target_col}']\n"
        "       X_holdout = apply_preparation(holdout_fold, final_prep_state)\n"
        "       X_holdout = X_holdout.drop(columns=[c for c in drop_cols if c in X_holdout.columns], errors='ignore')\n"
        "       if cat_cols_full:\n"
        "           X_holdout = pd.get_dummies(X_holdout, columns=cat_cols_full, drop_first=True, dummy_na=False)\n"
        "       X_holdout = X_holdout.reindex(columns=X_full.columns, fill_value=0.0)\n"
        "       X_holdout = X_holdout.fillna(0.0).replace([np.inf, -np.inf], 0.0)\n"
        "       if is_linear:\n"
        "           X_holdout = pd.DataFrame(scaler_full.transform(X_holdout), columns=X_full.columns)\n"
        "       holdout_preds = final_model.predict(X_holdout)\n"
        "       # calculate metric on holdout using same metric function as CV folds:\n"
        "       holdout_score = float(metric_fn(y_holdout, holdout_preds))\n"
        "   else:\n"
        "       holdout_score = None\n\n"
        "5. GBDT EARLY STOPPING & METRIC RULES:\n"
        "   - In CV folds: when using LightGBM/XGBoost/CatBoost with early stopping, you MUST pass eval_set=[(X_va, y_va)] to .fit().\n"
        "   - In final_model: do NOT use early stopping without an eval set; remove early stopping callbacks or pass eval_set=[(X_holdout, y_holdout)] so it does not crash with 'ValueError: Must have at least 1 validation dataset for early stopping'.\n"
        "   - Include 'holdout_score': holdout_score in <MODEL_RESULT> JSON dictionary.\n"
        "   - NEVER call np.isinf(df) or np.isnan(df) on DataFrames (raises TypeError: ufunc 'isinf' not supported)! Use df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], 0.0).fillna(0.0).\n\n"
        "6. REQUIRED FILES TO WRITE IN WORKING DIRECTORY:\n"
        f"   - val_predictions.csv (columns: ['{id_col}', 'fold', 'prediction'])\n"
        f"   - test_predictions.csv (columns: ['{id_col}', '{target_col}'] - MANDATORY: use '{target_col}', NOT 'prediction')!\n"
        f"   - For '{id_col}' in test_predictions: if '{id_col}' is in test_df.columns, use test_df['{id_col}']. If test_df has no ID column, check if sample submission exists to take its IDs; otherwise generate IDs from 1 to N (1-indexed: np.arange(1, len(test_df) + 1)), NEVER 0 to N-1!\n"
        "   - model.pkl (saved via joblib.dump)\n"
        "7. Output JSON tagged with <MODEL_RESULT>."
    )

    coder_result = run_coder(
        task_text=coder_task,
        result_tag=RESULT_TAG,
        result_keys_text=RESULT_KEYS,
        working_folder=exp_folder,
        script_name="train_model.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )

    # Initialize experiment run record
    run_record = ExperimentRun(
        experiment_id=exp_id,
        iteration=ledger.iteration,
        hypothesis_id=hyp.hypothesis_id if hyp else "",
        route_type=hyp.route if hyp else "fe_and_model",
        model_family=plan.model_family,
        complexity_tier=plan.complexity_tier,
        model_params=plan.hyperparameters,
        feature_version=feature_version,
        script_path=str(exp_folder / "train_model.py"),
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
            f"Experiment {exp_id} failed during execution: {coder_result.error_text[:120]}",
        )
        return plan

    # Extract metrics and scores
    data = coder_result.result_data
    run_record.train_score_mean = data.get("train_score_mean")
    run_record.cv_fold_scores = [float(s) for s in data.get("cv_fold_scores", [])]
    run_record.cv_mean = float(data.get("cv_mean")) if data.get("cv_mean") is not None else None
    run_record.cv_std = float(data.get("cv_std")) if data.get("cv_std") is not None else None
    run_record.holdout_score = (
        float(data.get("holdout_score")) if data.get("holdout_score") is not None else None
    )
    run_record.secondary_metrics = {
        k: float(v) for k, v in data.get("secondary_metrics", {}).items()
    }
    run_record.top_feature_importances = {
        k: float(v) for k, v in data.get("top_feature_importances", {}).items()
    }

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

    # Append ExperimentRunCard to MemoryStore
    cv_val = run_record.cv_mean if run_record.cv_mean is not None else 0.0
    cv_err = run_record.cv_std if run_record.cv_std is not None else 0.0
    holdout_val = run_record.holdout_score if run_record.holdout_score is not None else 0.0
    summary_text = (
        f"Run {exp_id}: {plan.model_family} (Tier {plan.complexity_tier}, FV {feature_version}). "
        f"CV={cv_val:.4f}±{cv_err:.4f}. Holdout={holdout_val:.4f}. Status={run_record.overfitting_status}."
    )
    run_card = ExperimentRunCard(
        tag="experiment_run",
        card_id=exp_id,
        owner="modeler",
        iteration=ledger.iteration,
        memory_summary=summary_text,
        payload={
            "experiment_id": exp_id,
            "model_family": plan.model_family,
            "tier": plan.complexity_tier,
            "feature_version": feature_version,
            "cv_mean": run_record.cv_mean,
            "cv_std": run_record.cv_std,
            "holdout_score": run_record.holdout_score,
            "train_score_mean": run_record.train_score_mean,
            "overfitting_status": run_record.overfitting_status,
            "feature_importances": dict(list(run_record.top_feature_importances.items())[:5]),
            "script_path": str(exp_folder / "train_model.py"),
            "artifact_dir": str(exp_folder),
        },
        artifact_paths=[str(exp_folder / "train_model.py"), str(exp_folder)],
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
        f"Experiment {exp_id} completed ({plan.model_family}, {score_str}): {plan.cli_summary}",
    )
    return plan
