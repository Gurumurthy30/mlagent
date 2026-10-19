"""Synchronizes Ledger state into typed MemoryCards in MemoryStore.

Called automatically after every node run during save_ledger.
Ensures cards are append-only, champion pointer is updated atomically,
and older cards are archived.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlagent.memory.cards import (
    ChampionPointerCard,
    EdaFindingCard,
    EvaluationVerdictCard,
    ExperimentRunCard,
    FeatureVersionCard,
    FinalDeliverableCard,
    HypothesisCard,
    LessonLearnedCard,
    PrepSpecCard,
    ProjectBriefCard,
    ValidationPlanCard,
)
from mlagent.memory.store import MemoryStore

if TYPE_CHECKING:
    from mlagent.graph.state import Ledger
    from mlagent.helper.workspace import Workspace


def _safe_append(store: MemoryStore, card: Any) -> None:
    # Append card only if it does not already exist
    tag_dir = store.cards_dir / card.tag
    target_file = tag_dir / f"{card.card_id}.json"
    if not target_file.exists():
        store.append_card(card)
    store.archive_old_cards(card.tag, keep=50)


def sync_ledger_node_memory(
    workspace: Workspace,
    ledger: Ledger,
    node_name: str | None = None,
) -> None:
    # Map node name or ledger phase to memory cards
    store = MemoryStore(workspace.root)
    active_node = node_name or getattr(ledger, "phase", None)

    # 1. data_engineer -> ProjectBriefCard
    if active_node == "data_engineer" and ledger.brief and ledger.brief.enhanced_problem_statement:
        brief = ledger.brief
        summary_text = (
            f"Problem: {brief.enhanced_problem_statement}. Target: {brief.target_column or 'None'}. "
            f"Family: {brief.task_family}. Metric: {brief.primary_metric}."
        )
        artifact_paths: list[str] = []
        if brief.profile and getattr(brief.profile, "project_folder", None):
            artifact_paths.append(str(brief.profile.project_folder))

        card = ProjectBriefCard(
            tag="project_brief",
            card_id="project_brief",
            owner="data_engineer",
            iteration=0,
            memory_summary=summary_text,
            payload={
                "problem_statement": brief.enhanced_problem_statement,
                "task_family": brief.task_family,
                "target_column": brief.target_column,
                "primary_metric": brief.primary_metric,
                "secondary_metrics": brief.secondary_metrics,
                "risks": brief.data_sufficiency_and_risks,
            },
            artifact_paths=artifact_paths,
            priority="core",
        )
        _safe_append(store, card)

    # 2. eda_think -> EdaFindingCard
    elif active_node in ("eda_think", "eda_analyst") and ledger.eda and ledger.eda.findings:
        for finding in ledger.eda.findings:
            summary = f"{finding.statement}: {finding.modeling_implication}"
            card = EdaFindingCard(
                tag="eda_finding",
                card_id=finding.finding_id,
                owner="data_analyst",
                iteration=finding.eda_round,
                memory_summary=summary,
                payload={
                    "finding_id": finding.finding_id,
                    "statement": finding.statement,
                    "quantitative_evidence": finding.quantitative_evidence,
                    "modeling_implication": finding.modeling_implication,
                    "priority": finding.priority,
                },
                artifact_paths=[],
                priority="medium",
            )
            _safe_append(store, card)

    # 3. eda_blueprint -> PrepSpecCard
    elif active_node == "eda_blueprint" and ledger.eda and ledger.eda.concise_executive_summary:
        eda = ledger.eda
        summary = eda.concise_executive_summary or "Data preparation and cleaning plan locked."
        card = PrepSpecCard(
            tag="prep_spec",
            card_id=f"prep_spec_round_{ledger.iteration}",
            owner="data_analyst",
            iteration=ledger.iteration,
            memory_summary=summary,
            payload={
                "concise_executive_summary": eda.concise_executive_summary,
                "duplicate_policy": eda.duplicate_handling_policy,
                "validation_scheme": eda.recommended_validation_scheme,
                "redundant_columns": eda.redundant_or_collinear_columns_to_drop,
                "leakage_risks": eda.leakage_risks_to_guard,
            },
            priority="high",
        )
        _safe_append(store, card)

    # 4. ml_lead_round_0 -> ValidationPlanCard
    elif active_node == "ml_lead_round_0" and ledger.validation_plan:
        val = ledger.validation_plan
        summary = (
            f"Scheme: {val.validation_scheme}, Splits: {val.n_splits}, Holdout: {val.holdout_fraction:.2f}, "
            f"Metric: {ledger.brief.primary_metric}."
        )
        card = ValidationPlanCard(
            tag="validation_plan",
            card_id="validation_plan",
            owner="ml_lead",
            iteration=0,
            memory_summary=summary,
            payload={
                "scheme": val.validation_scheme,
                "n_splits": val.n_splits,
                "holdout_fraction": val.holdout_fraction,
                "split_column": val.split_column,
                "time_column": val.time_column,
                "model_progression": val.planned_model_progression,
            },
            priority="core",
        )
        _safe_append(store, card)

    # 5. feature_engineer -> FeatureVersionCard
    elif active_node == "feature_engineer" and ledger.feature_versions:
        fv = ledger.feature_versions[-1]
        summary = (
            f"Feature Version {fv.feature_version} (parent {fv.parent_feature_version}). "
            f"Added: {len(fv.added_columns)} cols. Pruned: {len(fv.pruned_columns)} cols. "
            f"Hypothesis: {fv.hypothesis_id or 'baseline'}."
        )
        card = FeatureVersionCard(
            tag="feature_version",
            card_id=f"fv_{fv.feature_version}",
            owner="feature_engineer",
            iteration=ledger.iteration,
            memory_summary=summary,
            payload={
                "version": fv.feature_version,
                "parent_version": fv.parent_feature_version,
                "hypothesis_id": fv.hypothesis_id,
                "added_columns": fv.added_columns,
                "pruned_columns": fv.pruned_columns,
                "script_path": fv.script_path,
            },
            artifact_paths=[fv.script_path] if fv.script_path else [],
            priority="high",
        )
        _safe_append(store, card)

    # 6. modeler / tuner -> ExperimentRunCard
    elif active_node in ("modeler", "tuner") and ledger.runs:
        run = ledger.runs[-1]
        cv_val = run.cv_mean if run.cv_mean is not None else 0.0
        cv_err = run.cv_std if run.cv_std is not None else 0.0
        holdout_val = run.holdout_score if run.holdout_score is not None else 0.0
        summary = (
            f"Run {run.experiment_id}: {run.model_family} (Tier {run.complexity_tier}, FV {run.feature_version}). "
            f"CV={cv_val:.4f}±{cv_err:.4f}. Holdout={holdout_val:.4f}. "
            f"Status={run.overfitting_status}."
        )
        card = ExperimentRunCard(
            tag="experiment_run",
            card_id=run.experiment_id,
            owner=active_node,
            iteration=run.iteration,
            memory_summary=summary,
            payload={
                "experiment_id": run.experiment_id,
                "model_family": run.model_family,
                "tier": run.complexity_tier,
                "feature_version": run.feature_version,
                "cv_mean": run.cv_mean,
                "cv_std": run.cv_std,
                "holdout_score": run.holdout_score,
                "train_score_mean": run.train_score_mean,
                "overfitting_status": run.overfitting_status,
                "feature_importances": dict(list(run.top_feature_importances.items())[:5]),
                "script_path": run.script_path,
                "artifact_dir": run.artifact_dir,
            },
            artifact_paths=[p for p in (run.script_path, run.artifact_dir) if p],
            priority="high",
        )
        _safe_append(store, card)

    # 7. evaluator -> EvaluationVerdictCard + (conditionally) ChampionPointerCard
    elif active_node == "evaluator" and ledger.runs:
        run = ledger.runs[-1]
        verdict_str = run.verdict or "audit_complete"
        beats_champ = run.promoted_to_champion
        gain_val = run.gain_vs_champion if run.gain_vs_champion is not None else 0.0
        summary = (
            f"Verdict {run.experiment_id}: {verdict_str}. Beats champ: {beats_champ} "
            f"(gain {gain_val:+.4f}). Status: {run.overfitting_status}."
        )
        card = EvaluationVerdictCard(
            tag="evaluation_verdict",
            card_id=f"verdict_{run.experiment_id}",
            owner="evaluator",
            iteration=ledger.iteration,
            memory_summary=summary,
            payload={
                "experiment_id": run.experiment_id,
                "verdict": verdict_str,
                "overfitting_status": run.overfitting_status,
                "train_vs_cv_gap": 0.0,
                "cv_vs_holdout_gap": None,
                "beats_champion": beats_champ,
                "gain_vs_champion": gain_val,
                "error_slices": [s.model_dump() for s in run.worst_error_slices[:3]],
                "feedback": run.evaluator_feedback,
            },
            priority="high",
        )
        _safe_append(store, card)

        # Update champion pointer if champion matches this experiment
        champ = ledger.champion_run()
        if champ and champ.experiment_id == run.experiment_id and beats_champ:
            champ_cv = champ.cv_mean if champ.cv_mean is not None else 0.0
            champ_summary = (
                f"Champion: {champ.experiment_id} (FV {champ.feature_version}) "
                f"CV={champ_cv:.4f} promoted."
            )
            champ_card = ChampionPointerCard(
                tag="champion_pointer",
                card_id=f"champion_iter_{ledger.iteration}_{champ.experiment_id}",
                owner="evaluator",
                iteration=ledger.iteration,
                memory_summary=champ_summary,
                payload={
                    "champion_experiment_id": champ.experiment_id,
                    "champion_feature_version": champ.feature_version,
                    "champion_cv_mean": champ_cv,
                    "promoted_at": f"iter_{ledger.iteration}",
                },
                priority="core",
            )
            store.update_champion_pointer(champ_card)

    # 8. ml_lead -> HypothesisCard + LessonLearnedCard
    elif active_node == "ml_lead":
        if ledger.hypotheses:
            hyp = ledger.hypotheses[-1]
            summary = (
                f"Hypothesis {hyp.hypothesis_id} ({hyp.route}): {hyp.hypothesis_statement} "
                f"Target: {hyp.target_model_family} (Tier {hyp.model_complexity_tier})."
            )
            h_card = HypothesisCard(
                tag="hypothesis",
                card_id=hyp.hypothesis_id,
                owner="ml_lead",
                iteration=hyp.iteration,
                memory_summary=summary,
                payload={
                    "hypothesis_id": hyp.hypothesis_id,
                    "iteration": hyp.iteration,
                    "route": hyp.route,
                    "base_feature_version": hyp.base_feature_version,
                    "statement": hyp.hypothesis_statement,
                    "selected_idea_ids": hyp.selected_feature_idea_ids,
                    "target_model_family": hyp.target_model_family,
                    "tier": hyp.model_complexity_tier,
                    "lesson_learned": hyp.lesson_learned,
                },
                priority="high",
            )
            _safe_append(store, h_card)

            if hyp.lesson_learned:
                l_summary = (
                    f"Lesson {hyp.hypothesis_id}: {hyp.lesson_learned} (Route: {hyp.route})."
                )
                l_card = LessonLearnedCard(
                    tag="lesson_learned",
                    card_id=f"lesson_{hyp.hypothesis_id}",
                    owner="ml_lead",
                    iteration=hyp.iteration,
                    memory_summary=l_summary,
                    payload={
                        "lesson_id": f"lesson_{hyp.hypothesis_id}",
                        "iteration": hyp.iteration,
                        "lesson_text": hyp.lesson_learned,
                        "applies_to_route": hyp.route,
                    },
                    priority="core",
                )
                _safe_append(store, l_card)

    # 9. finalizer -> FinalDeliverableCard
    elif active_node == "finalizer" and ledger.final and ledger.final.champion_experiment_id:
        fd = ledger.final
        cv_sc = fd.final_cv_score if fd.final_cv_score is not None else 0.0
        ho_sc = fd.final_holdout_score if fd.final_holdout_score is not None else 0.0
        summary = (
            f"Strategy: {fd.delivery_strategy}. Champion: {fd.champion_experiment_id}. "
            f"CV: {cv_sc:.4f}. Holdout: {ho_sc:.4f}."
        )
        card = FinalDeliverableCard(
            tag="final_deliverable",
            card_id="final_deliverable",
            owner="finalizer",
            iteration=ledger.iteration,
            memory_summary=summary,
            payload={
                "delivery_strategy": fd.delivery_strategy,
                "champion_experiment_id": fd.champion_experiment_id,
                "ensemble_ids": fd.ensemble_experiment_ids,
                "final_cv_score": fd.final_cv_score,
                "final_holdout_score": fd.final_holdout_score,
                "predictions_path": fd.predictions_path,
                "pipeline_path": fd.serialized_pipeline_path,
                "monitoring_plan": [m.model_dump() for m in fd.monitoring_plan],
                "known_limitations": fd.check_messages,
            },
            artifact_paths=[p for p in (fd.predictions_path, fd.serialized_pipeline_path) if p],
            priority="core",
        )
        _safe_append(store, card)

    # 10. summarizer -> archive maintenance
    elif active_node == "summarizer":
        store.archive_old_cards("final_deliverable", keep=50)
