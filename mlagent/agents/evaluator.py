"""Independent Evaluator agent: rigorous audit and Champion-vs-Challenger linear ratchet.

Audits candidate model runs against the current Champion. Verifies leak-free fold isolation,
inspects train-vs-CV and CV-vs-holdout generalization gaps, identifies elevated-error subgroups,
and executes the Champion ratchet update.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Agent execution context and exceptions
from mlagent.agents.base import AgentContext, AgentFailedError

# Master graph state models
from mlagent.graph.state import (
    EvaluationVerdict,
    Ledger,
)

# Event logging and notifications
from mlagent.helper.events import EventType

# Metric calculation registry
# Prompt templates and role definitions
from mlagent.memory.assembler import MemoryAssembler
from mlagent.memory.cards import (
    ChampionPointerCard,
    EvaluationVerdictCard,
)
from mlagent.memory.store import MemoryStore
from mlagent.prompts.evaluator import EVALUATOR_ROLE

AGENT_KEY = "evaluator"
AGENT_NAME = "Evaluator"


def _audit_predictions_file(pred_path: Path) -> tuple[bool, str, int]:
    # Check predictions file integrity: exists, non-empty, zero NaNs/Infs
    if not pred_path.exists():
        return False, f"File {pred_path.name} not found on disk.", 0
    try:
        df = pd.read_csv(pred_path)
    except Exception as exc:
        return False, f"Could not read {pred_path.name}: {exc}", 0

    if len(df) == 0:
        return False, f"File {pred_path.name} is empty.", 0

    if df.isna().any().any():
        return False, f"File {pred_path.name} contains NaN values.", len(df)

    return True, "Valid", len(df)


def run_evaluator(ledger: Ledger, agent_context: AgentContext) -> EvaluationVerdict:
    # Audit latest experiment run and update Champion status
    event_bus = agent_context.event_bus
    if not ledger.runs:
        raise AgentFailedError("Evaluator called but no experiment runs exist.")

    run = ledger.runs[-1]
    event_bus.emit(
        EventType.AGENT_STARTED,
        AGENT_NAME,
        f"auditing experiment {run.experiment_id} ({run.model_family})",
        ledger=ledger,
    )

    # 1. Handle execution crash
    if run.crashed or run.cv_mean is None:
        rejection_reasons = [f"Script crashed during training: {run.error_traceback[-200:]}"]
        verdict = EvaluationVerdict(
            thinking="Script crashed during execution. Hard rejection.",
            cli_summary=f"Experiment {run.experiment_id} crashed during training. Rejected.",
            experiment_id=run.experiment_id,
            verdict="reject",
            rejection_reasons=rejection_reasons,
            overfitting_status="healthy",
            train_vs_cv_gap=0.0,
            beats_current_champion=False,
            primary_metric_gain_vs_champion=0.0,
            actionable_feedback_for_ml_lead="Fix script syntax, API deprecation, or memory overflow.",
        )
        run.verdict = "reject"
        run.verdict_reasons = rejection_reasons
        run.evaluator_feedback = verdict.actionable_feedback_for_ml_lead

        if ledger.hypotheses:
            ledger.hypotheses[-1].outcome = "script_error"

        event_bus.emit(
            EventType.WARNING,
            AGENT_NAME,
            f"Run {run.experiment_id} rejected due to execution crash.",
        )
        return verdict

    # 2. Independent file and metric checks
    exp_dir = Path(run.artifact_dir)
    val_pred_path = exp_dir / "val_predictions.csv"
    test_pred_path = exp_dir / "test_predictions.csv"

    val_ok, val_msg, val_rows = _audit_predictions_file(val_pred_path)
    test_ok, test_msg, test_rows = _audit_predictions_file(test_pred_path)

    hard_failures: list[str] = []
    if not val_ok:
        hard_failures.append(f"Validation predictions check failed: {val_msg}")
    if not test_ok:
        hard_failures.append(f"Test predictions check failed: {test_msg}")

    # Generalization gaps
    train_score = run.train_score_mean if run.train_score_mean is not None else run.cv_mean
    train_cv_gap = abs(train_score - run.cv_mean)
    holdout_gap = abs(run.cv_mean - run.holdout_score) if run.holdout_score is not None else None

    # Current Champion benchmark
    champion = ledger.champion_run(exclude_experiment_id=run.experiment_id)
    is_baseline = champion is None or champion.cv_mean is None

    if not is_baseline:
        raw_gain = run.cv_mean - champion.cv_mean
        gain_vs_champ = raw_gain if ledger.brief.higher_is_better else -raw_gain
        beats_champ_metric = gain_vs_champ > 0.0
        champ_status_text = (
            f"Champion: {champion.experiment_id} (CV={champion.cv_mean:.4f}). "
            f"Primary metric: {ledger.brief.primary_metric} ({'higher' if ledger.brief.higher_is_better else 'lower'} is better). "
            f"Gain vs Champion: {gain_vs_champ:+.4f} (positive = improvement)."
        )
    else:
        gain_vs_champ = 0.0
        beats_champ_metric = True
        champ_status_text = (
            "Champion: None yet. This candidate is the initial BASELINE run. "
            "Do NOT reject for zero gain or missing champion comparison. "
            "Approve as the initial Champion if predictions are valid and generalization is healthy or mild_overfit."
        )

    audit_metrics_text = (
        f"Validation predictions: {val_msg} ({val_rows} rows)\n"
        f"Test predictions: {test_msg} ({test_rows} rows)\n"
        f"Train-vs-CV Gap: {train_cv_gap:.4f}\n"
        f"CV-vs-Holdout Gap: {f'{holdout_gap:.4f}' if holdout_gap is not None else 'N/A'}\n"
        f"{champ_status_text}\n"
        f"Hard failures detected: {'; '.join(hard_failures) if hard_failures else 'None'}"
    )

    store = MemoryStore(agent_context.workspace.root)
    assembler = MemoryAssembler(store)

    task_text = (
        f"Audit experiment {run.experiment_id} ({run.model_family}, Tier: {run.complexity_tier}) on features v{run.feature_version}. "
        f"Results: CV={run.cv_mean:.4f} (std: {run.cv_std or 0.0:.4f}), Train={run.train_score_mean}, Holdout={run.holdout_score}.\n"
        f"{audit_metrics_text}\n"
        "Assess generalization integrity, identify error slices, and render approval verdict."
    )
    prompt = assembler.build_prompt(
        agent_name="evaluator",
        ledger=ledger,
        task=task_text,
        schema_hint=str(EvaluationVerdict.model_json_schema()),
    )

    try:
        verdict = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=EvaluationVerdict,
            system_message=EVALUATOR_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(
            f"Evaluator structured audit failed for {run.experiment_id}: {exc}"
        ) from exc

    # 2.5 Authoritative Python Mathematical Decision
    if hard_failures:
        determined_verdict = "reject"
        verdict.rejection_reasons.extend(hard_failures)
    elif is_baseline:
        # Baseline establishes Champion if predictions valid and not severe overfit
        if verdict.overfitting_status == "severe_overfit":
            determined_verdict = "reject"
        else:
            determined_verdict = "approve"
            # Clear spurious rejection reasons from LLM regarding zero gain on baseline
            verdict.rejection_reasons = [
                r
                for r in verdict.rejection_reasons
                if not any(
                    k in r.lower()
                    for k in ["no gain", "gain <=", "zero gain", "missing holdout", "unverified"]
                )
            ]
    else:
        # Challenger must beat Champion CV score under primary metric and not severely overfit
        if beats_champ_metric and verdict.overfitting_status != "severe_overfit":
            determined_verdict = "approve"
            verdict.rejection_reasons = [
                r
                for r in verdict.rejection_reasons
                if not any(
                    k in r.lower() for k in ["no gain", "gain <=", "zero gain", "missing holdout"]
                )
            ]
        else:
            determined_verdict = "reject"
            if not beats_champ_metric:
                no_gain_msg = (
                    f"Candidate CV ({run.cv_mean:.4f}) did not beat Champion CV ({champion.cv_mean:.4f}). "
                    f"Gain: {gain_vs_champ:+.4f}."
                )
                if not any(
                    "gain" in r.lower() or "beat" in r.lower() for r in verdict.rejection_reasons
                ):
                    verdict.rejection_reasons.append(no_gain_msg)

    verdict.verdict = determined_verdict
    verdict.primary_metric_gain_vs_champion = gain_vs_champ

    # 3. Update ExperimentRun record
    run.verdict = verdict.verdict
    run.verdict_reasons = list(verdict.rejection_reasons)
    run.overfitting_status = verdict.overfitting_status
    run.evaluator_feedback = verdict.actionable_feedback_for_ml_lead
    run.gain_vs_champion = verdict.primary_metric_gain_vs_champion

    # 4. Champion linear ratchet update
    promoted = False
    if verdict.verdict == "approve":
        promoted = ledger.update_champion_after_evaluation(run)

    verdict.beats_current_champion = promoted

    # 5. Append EvaluationVerdictCard to MemoryStore
    verdict_summary = (
        f"Verdict {run.experiment_id}: {verdict.verdict}. Beats champ: {promoted} "
        f"(gain {verdict.primary_metric_gain_vs_champion:+.4f}). Status: {verdict.overfitting_status}."
    )
    verdict_card = EvaluationVerdictCard(
        tag="evaluation_verdict",
        card_id=f"verdict_{run.experiment_id}",
        owner="evaluator",
        iteration=ledger.iteration,
        memory_summary=verdict_summary,
        payload={
            "experiment_id": run.experiment_id,
            "verdict": verdict.verdict,
            "overfitting_status": verdict.overfitting_status,
            "train_vs_cv_gap": verdict.train_vs_cv_gap,
            "cv_vs_holdout_gap": verdict.cv_vs_holdout_gap,
            "beats_champion": promoted,
            "gain_vs_champion": verdict.primary_metric_gain_vs_champion,
            "error_slices": [s.model_dump() for s in verdict.worst_error_slices[:3]],
            "feedback": verdict.actionable_feedback_for_ml_lead,
        },
        priority="high",
    )
    try:
        store.append_card(verdict_card)
    except FileExistsError:
        pass

    # 6. Update mutable ChampionPointerCard if new champion was promoted
    if promoted:
        champ_cv = run.cv_mean if run.cv_mean is not None else 0.0
        champ_summary = (
            f"Champion: {run.experiment_id} (FV {run.feature_version}) CV={champ_cv:.4f} promoted."
        )
        champ_card = ChampionPointerCard(
            tag="champion_pointer",
            card_id=f"champion_iter_{ledger.iteration}_{run.experiment_id}",
            owner="evaluator",
            iteration=ledger.iteration,
            memory_summary=champ_summary,
            payload={
                "champion_experiment_id": run.experiment_id,
                "champion_feature_version": run.feature_version,
                "champion_cv_mean": champ_cv,
                "promoted_at": f"iter_{ledger.iteration}",
            },
            priority="core",
        )
        store.update_champion_pointer(champ_card)

    # 7. Update latest hypothesis record
    if ledger.hypotheses:
        latest_hyp = ledger.hypotheses[-1]
        if promoted:
            latest_hyp.outcome = "validated_new_champion"
        elif verdict.overfitting_status == "severe_overfit":
            latest_hyp.outcome = "overfit_rejected"
        else:
            latest_hyp.outcome = "refuted_regression"

    # Emit completion event
    status_label = "PROMOTED TO CHAMPION" if promoted else f"Verdict: {verdict.verdict}"
    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"Audit of {run.experiment_id} ({status_label}): {verdict.cli_summary}",
    )
    return verdict
