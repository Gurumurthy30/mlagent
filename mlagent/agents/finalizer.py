"""Finalizer agent: deliverable ensembling, deployment verification, and drift monitoring.

Finalizes the end-to-end ML solution by packaging either the single winning Champion model
or an optimal ensemble of approved candidate models. Verifies the final deliverable against
the project contract, writes the technical summary report, and defines the production drift
monitoring specification with numeric alert thresholds.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Agent execution context and exceptions
from mlagent.agents.base import AgentContext, AgentFailedError

# Master graph state models
from mlagent.graph.state import (
    DeliveryPlan,
    FinalDeliverable,
    Ledger,
)

# Deliverable file verification
from mlagent.helper.deliverable_checks import check_predictions_file, find_sample_submission

# Event logging and notifications
from mlagent.helper.events import EventType

# Prompt templates and role definitions
from mlagent.memory.assembler import MemoryAssembler
from mlagent.memory.cards import FinalDeliverableCard
from mlagent.memory.store import MemoryStore
from mlagent.prompts.finalizer import FINALIZER_ROLE

AGENT_KEY = "finalizer"
AGENT_NAME = "Finalizer"


def _blend_test_predictions(
    selected_exp_ids: list[str],
    ledger: Ledger,
    output_path: Path,
) -> bool:
    # Blend test predictions from multiple approved experiments
    dataframes: list[pd.DataFrame] = []
    runs_by_id = {r.experiment_id: r for r in ledger.runs}

    for exp_id in selected_exp_ids:
        run = runs_by_id.get(exp_id)
        if not run:
            continue
        test_path = Path(run.artifact_dir) / "test_predictions.csv"
        if test_path.exists():
            try:
                df = pd.read_csv(test_path)
                dataframes.append(df)
            except Exception:
                pass

    if not dataframes:
        return False

    if len(dataframes) == 1:
        dataframes[0].to_csv(output_path, index=False)
        return True

    # Average numeric prediction columns across models while keeping ID columns
    base_df = dataframes[0].copy()
    numeric_cols = [c for c in base_df.columns if pd.api.types.is_numeric_dtype(base_df[c])]

    for col in numeric_cols:
        col_arrays = [df[col].to_numpy() for df in dataframes if col in df.columns]
        if col_arrays:
            base_df[col] = sum(col_arrays) / len(col_arrays)

    base_df.to_csv(output_path, index=False)
    return True


def _align_and_save_predictions(
    src_path: Path,
    dst_paths: list[Path],
    brief: Any,
    project_folder: Path,
) -> None:
    # Post-process predictions: align with sample submission or contract, enforce 1-to-N IDs
    if not src_path.exists():
        return
    try:
        df = pd.read_csv(src_path)
    except Exception:
        for dst in dst_paths:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
        return

    sample_path = find_sample_submission(brief, project_folder)
    if sample_path is not None:
        try:
            sample_df = pd.read_csv(sample_path)
            aligned_df = pd.DataFrame()
            id_col = sample_df.columns[0]
            if id_col in df.columns:
                aligned_df[id_col] = df[id_col]
            else:
                first_col = df.columns[0]
                if (
                    first_col.lower() in ["id", "imageid", "row_id", "sample_id", "index"]
                    or len(df.columns) > 1
                ):
                    aligned_df[id_col] = df[first_col]
                else:
                    aligned_df[id_col] = sample_df[id_col].values[: len(df)]

            for idx, target_col in enumerate(sample_df.columns[1:], start=1):
                if target_col in df.columns:
                    aligned_df[target_col] = df[target_col]
                elif "prediction" in df.columns:
                    aligned_df[target_col] = df["prediction"]
                elif len(df.columns) > idx:
                    aligned_df[target_col] = df.iloc[:, idx]
                elif len(df.columns) == 1:
                    aligned_df[target_col] = df.iloc[:, 0]
                else:
                    aligned_df[target_col] = sample_df[target_col].values[: len(df)]

            if len(aligned_df) == len(sample_df):
                aligned_df[sample_df.columns[0]] = sample_df.iloc[:, 0].values

            aligned_df = aligned_df.reindex(columns=sample_df.columns)
            df = aligned_df
        except Exception:
            pass
    else:
        target_col = brief.target_column or "target"
        id_cols = brief.id_columns
        id_col = id_cols[0] if id_cols else "Id"

        if len(df.columns) == 1:
            val_col = df.columns[0]
            ids = np.arange(1, len(df) + 1)
            df = pd.DataFrame({id_col: ids, target_col: df[val_col]})
        elif len(df.columns) >= 2:
            rename_dict = {}
            if "prediction" in df.columns and target_col not in df.columns:
                rename_dict["prediction"] = target_col

            first_col = df.columns[0]
            if first_col != id_col and (
                first_col.lower() in ["id", "imageid", "row_id", "index", "unnamed: 0"]
                or first_col == "0"
            ):
                rename_dict[first_col] = id_col

            if rename_dict:
                df = df.rename(columns=rename_dict)

            if target_col not in df.columns and len(df.columns) >= 2:
                cols = list(df.columns)
                cols[1] = target_col
                df.columns = cols

            if id_col not in df.columns:
                df.insert(0, id_col, np.arange(1, len(df) + 1))
            else:
                id_vals = df[id_col].to_numpy()
                if len(id_vals) > 0 and id_vals[0] == 0 and id_vals[-1] == len(df) - 1:
                    df[id_col] = np.arange(1, len(df) + 1)

            if id_col in df.columns and target_col in df.columns:
                df = df[[id_col, target_col]]

    for dst in dst_paths:
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False)


def run_finalizer(ledger: Ledger, agent_context: AgentContext) -> DeliveryPlan:
    # Finalize delivery package and write technical report via MemoryAssembler
    event_bus = agent_context.event_bus
    champion = ledger.champion_run()
    if not champion or champion.cv_mean is None:
        approved = [r for r in ledger.runs if r.approved]
        if approved:
            champion = approved[-1]
        else:
            raise AgentFailedError("Cannot finalize: no approved candidate model runs found.")

    event_bus.emit(
        EventType.AGENT_STARTED,
        AGENT_NAME,
        f"packaging final deliverable based on Champion {champion.experiment_id} ({champion.model_family})",
        ledger=ledger,
    )

    store = MemoryStore(agent_context.workspace.root)
    assembler = MemoryAssembler(store)

    task_text = (
        f"Package final deliverables based on Champion {champion.experiment_id} ({champion.model_family}, "
        f"CV={champion.cv_mean:.4f}). Select single champion or weighted ensemble, formulate production drift "
        "monitoring thresholds, and author executive summary."
    )
    prompt = assembler.build_prompt(
        agent_name="finalizer",
        ledger=ledger,
        task=task_text,
        schema_hint=str(DeliveryPlan.model_json_schema()),
    )

    try:
        plan = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=DeliveryPlan,
            system_message=FINALIZER_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(f"Finalizer planning failed: {exc}") from exc

    # Set up final deliverables directories (artifacts/final and report/)
    final_dir = agent_context.workspace.artifacts_folder / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    report_dir = agent_context.workspace.report_folder
    report_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = final_dir / "predictions.csv"
    report_predictions_file = report_dir / "predictions.csv"

    # Ensembling or champion copy
    is_blended = False
    if (
        plan.delivery_strategy in ["weighted_oof_ensemble", "stacking_ensemble"]
        and len(plan.selected_experiment_ids) > 1
    ):
        is_blended = _blend_test_predictions(plan.selected_experiment_ids, ledger, predictions_file)

    if not is_blended:
        champ_test = Path(champion.artifact_dir) / "test_predictions.csv"
        if champ_test.exists():
            shutil.copy2(champ_test, predictions_file)
        else:
            raise AgentFailedError(f"Champion predictions {champ_test} do not exist.")

    # Align predictions columns and IDs and save to both artifacts/final and report/
    _align_and_save_predictions(
        src_path=predictions_file,
        dst_paths=[predictions_file, report_predictions_file],
        brief=ledger.brief,
        project_folder=agent_context.workspace.project_folder,
    )

    # Copy champion model file to both final_dir and report_dir
    serialized_model_path = ""
    champ_model = Path(champion.model_path)
    if champ_model.exists():
        target_model = final_dir / champ_model.name
        shutil.copy2(champ_model, target_model)
        serialized_model_path = str(target_model)

        report_model = report_dir / champ_model.name
        shutil.copy2(champ_model, report_model)
        if champ_model.name != "model.pkl":
            try:
                shutil.copy2(champ_model, report_dir / "model.pkl")
            except Exception:
                pass

    # Deliverable format checks
    problems = check_predictions_file(
        predictions_path=predictions_file,
        brief=ledger.brief,
        project_folder=agent_context.workspace.project_folder,
    )

    # Compile executive technical report
    report_file = final_dir / "technical_report.md"
    report_content = [
        "# Executive Technical Report\n",
        f"## Project Objective\n{ledger.brief.enhanced_problem_statement}\n",
        f"## Final Model Architecture\n- Strategy: {plan.delivery_strategy}\n"
        f"- Winning Base Model: {champion.model_family}\n"
        f"- Primary Metric ({ledger.brief.primary_metric}): CV={champion.cv_mean:.4f}, Holdout={champion.holdout_score}\n",
        f"## Executive Summary\n{plan.executive_summary}\n",
        "## Known Limitations and Failure Modes\n",
    ]
    for lim in plan.known_limitations_and_failure_modes:
        report_content.append(f"- {lim}\n")

    report_content.append("\n## Production Drift Monitoring Plan\n")
    report_content.append(
        "| Feature / Target | Baseline Metric | Drift Test | Warning Threshold | Retrain Threshold | Action |\n"
    )
    report_content.append("|---|---|---|---|---|---|\n")
    for drift in plan.monitoring_and_maintenance_plan:
        report_content.append(
            f"| {drift.feature_or_target} | {drift.baseline_statistic} | {drift.drift_metric} | "
            f"{drift.warning_threshold} | {drift.retrain_threshold} | {drift.action} |\n"
        )

    full_report_text = "".join(report_content)
    report_file.write_text(full_report_text, encoding="utf-8")
    (report_dir / "report.md").write_text(full_report_text, encoding="utf-8")
    (report_dir / "technical_report.md").write_text(full_report_text, encoding="utf-8")

    # Update ledger final record
    ledger.final = FinalDeliverable(
        delivery_strategy=plan.delivery_strategy,
        champion_experiment_id=champion.experiment_id,
        ensemble_experiment_ids=plan.selected_experiment_ids,
        ensemble_weights={},
        final_cv_score=champion.cv_mean,
        final_holdout_score=champion.holdout_score,
        final_secondary_metrics=champion.secondary_metrics,
        predictions_path=str(report_predictions_file),
        serialized_pipeline_path=serialized_model_path,
        report_path=str(report_dir / "report.md"),
        deployment_checks_passed=(len(problems) == 0),
        check_messages=problems,
        monitoring_plan=plan.monitoring_and_maintenance_plan,
    )
    ledger.phase = "done"
    ledger.status = "done"

    # Append FinalDeliverableCard to MemoryStore
    cv_val = champion.cv_mean if champion.cv_mean is not None else 0.0
    ho_val = champion.holdout_score if champion.holdout_score is not None else 0.0
    summary_text = (
        f"Strategy: {plan.delivery_strategy}. Champion: {champion.experiment_id}. "
        f"CV: {cv_val:.4f}. Holdout: {ho_val:.4f}. Limits: {', '.join(problems)}."
    )
    final_card = FinalDeliverableCard(
        tag="final_deliverable",
        card_id="final_deliverable",
        owner="finalizer",
        iteration=ledger.iteration,
        memory_summary=summary_text,
        payload={
            "delivery_strategy": plan.delivery_strategy,
            "champion_experiment_id": champion.experiment_id,
            "ensemble_ids": plan.selected_experiment_ids,
            "final_cv_score": champion.cv_mean,
            "final_holdout_score": champion.holdout_score,
            "predictions_path": str(predictions_file),
            "pipeline_path": serialized_model_path,
            "monitoring_plan": [m.model_dump() for m in plan.monitoring_and_maintenance_plan],
            "known_limitations": problems,
        },
        artifact_paths=[str(predictions_file), serialized_model_path, str(report_file)],
        priority="core",
    )
    try:
        store.append_card(final_card)
    except FileExistsError:
        pass

    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"Delivery package ready ({plan.delivery_strategy}): {plan.cli_summary}",
    )
    return plan
