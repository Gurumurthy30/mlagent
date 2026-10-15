"""The Evaluator: an independent reviewer who verifies run honesty and checks for leakage.

The Coder writes an independent verification script to recompute the score and check leakage.
Code rules catch hard failures immediately. Otherwise the LLM assesses findings.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from mlagent.agents.base import AgentContext
from mlagent.agents.coder import run_coder
from mlagent.digest import build_digest
from mlagent.events import EventType
from mlagent.scoring import is_counted_gain
from mlagent.script_contract import SPLITS_FORMAT_TEXT, VALIDATION_PREDICTIONS_FORMAT_TEXT
from mlagent.state import ExperimentRun, Ledger, Verdict

AGENT_KEY = "evaluator"
AGENT_NAME = "Evaluator"
RESULT_TAG = "CHECKS"

# Tunable evaluator thresholds
SCORE_TOLERANCE_RELATIVE = 0.02
HOLDOUT_GAP_STD_MULTIPLIER = 3.0
HOLDOUT_GAP_RELATIVE_LIMIT = 0.05
LARGE_SCORE_JUMP_STD_MULTIPLIER = 3.0
TRAIN_VAL_OVERLAP_TOLERANCE = 0

CHECKS_RESULT_KEYS = """\
- "recomputed_score": number or null, mean across validation splits recomputed from predictions and true labels
- "recomputed_holdout_score": number or null, score on holdout examples (null if no holdout)
- "leakage_flags": list of {"description": text, "severity": "high"|"medium"|"low"}
- "train_validation_overlap": integer, number of validation examples in train split
- "notes": list of short observations"""

VERDICT_PROMPT = """\
You are the Evaluator of an ML team. You independently verify model runs.
You never see the Modeler's reasoning, only the verification checks.
You do not judge model architecture choice—only whether the score is honest, reproducible, and leak-free.

GOAL:
Determine if the model's performance is honest and usable, or flawed by data leakage or evaluation errors.

SITUATION:
{digest}

THE RUN:
{run_line}

INDEPENDENT CHECKS:
{checks_text}

HOW TO THINK:
- An honest modest score is approved; an amazing score achieved via leakage or target encoding in features must be rejected.
- High severity leakage or train-val overlap is a strict rejection.
- Medium flags require plausible domain explanations (e.g., legitimate pre-outcome features). Two unexplained medium flags warrant rejection.
- A score that is completely constant across splits or identical to a trivial baseline should be examined with scrutiny.

YOU MAY:
- Approve a run whose score is lower than previous runs, provided it is methodologically sound.
- Reject a run whose predictions are corrupted, constant, or misaligned with true labels.

OUTPUT:
Return JSON:
- thinking: 2-4 sentences reviewing the checks.
- reasons: list of short factual reasons citing numbers from the checks.
- verdict: "approve" or "reject"."""


class LeakageFlag(BaseModel):
    description: str
    severity: Literal["high", "medium", "low"] = "medium"


class EvaluationChecks(BaseModel):
    recomputed_score: float | None = None
    recomputed_holdout_score: float | None = None
    leakage_flags: list[LeakageFlag] = []
    train_validation_overlap: int = 0
    notes: list[str] = []


class VerdictAnswer(BaseModel):
    thinking: str = ""
    reasons: list[str] = []
    verdict: Verdict = "approve"


def hard_check_reasons(
    run: ExperimentRun,
    checks: EvaluationChecks,
    higher_is_better: bool,
    best_approved_run: ExperimentRun | None = None,
    config: Any = None,
) -> list[str]:
    """Rules that reject a run without calling the LLM."""
    reasons: list[str] = []
    mean_score = run.score_mean
    std_score = run.score_std or 0.0

    score_tol = getattr(config, "score_tolerance_relative", SCORE_TOLERANCE_RELATIVE) if config else SCORE_TOLERANCE_RELATIVE
    gap_std_mult = getattr(config, "holdout_gap_std_multiplier", HOLDOUT_GAP_STD_MULTIPLIER) if config else HOLDOUT_GAP_STD_MULTIPLIER
    gap_rel_lim = getattr(config, "holdout_gap_relative_limit", HOLDOUT_GAP_RELATIVE_LIMIT) if config else HOLDOUT_GAP_RELATIVE_LIMIT
    jump_std_mult = getattr(config, "large_score_jump_std_multiplier", LARGE_SCORE_JUMP_STD_MULTIPLIER) if config else LARGE_SCORE_JUMP_STD_MULTIPLIER

    # 1. Score mismatch
    if checks.recomputed_score is not None and mean_score is not None:
        tolerance = max(std_score, score_tol * abs(mean_score), 1e-9)
        if abs(checks.recomputed_score - mean_score) > tolerance:
            reasons.append(
                f"Recomputed score {checks.recomputed_score:.4f} differs from logged "
                f"score {mean_score:.4f} by more than {tolerance:.4f}."
            )

    # 2. Holdout gap
    holdout_score = checks.recomputed_holdout_score if checks.recomputed_holdout_score is not None else run.holdout_score
    if holdout_score is not None and mean_score is not None:
        shortfall = (mean_score - holdout_score) if higher_is_better else (holdout_score - mean_score)
        allowed_gap = max(gap_std_mult * std_score, gap_rel_lim * abs(mean_score))
        if shortfall > allowed_gap:
            reasons.append(
                f"Holdout score {holdout_score:.4f} is significantly worse than validation "
                f"score {mean_score:.4f} (shortfall {shortfall:.4f} > {allowed_gap:.4f})."
            )

    # 3. Leakage flags
    high_flags = [f for f in checks.leakage_flags if f.severity == "high"]
    if high_flags:
        reasons.extend(f"High-severity leakage: {f.description}" for f in high_flags)

    medium_flags = [f for f in checks.leakage_flags if f.severity == "medium"]
    if len(medium_flags) >= 2:
        reasons.append(f"Multiple medium-severity leakage flags detected ({len(medium_flags)}).")

    # 4. Train-validation overlap
    if checks.train_validation_overlap > TRAIN_VAL_OVERLAP_TOLERANCE:
        reasons.append(
            f"Train/validation overlap detected: {checks.train_validation_overlap} examples overlap."
        )

    # 5. Large jump with leakage flag
    if best_approved_run and best_approved_run.score_mean is not None and mean_score is not None:
        jump = (mean_score - best_approved_run.score_mean) if higher_is_better else (best_approved_run.score_mean - mean_score)
        if jump > jump_std_mult * max(std_score, 1e-6) and checks.leakage_flags:
            reasons.append(
                f"Score jump ({jump:+.4f}) is abnormally large (> {jump_std_mult}x std) while leakage flags are present."
            )

    return reasons


def run_evaluator(
    ledger: Ledger, agent_context: AgentContext, run: ExperimentRun
) -> ExperimentRun:
    """Evaluate one run and return it with verdict, reasons, gain_vs_best, counted_gain."""
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, f"checking {run.experiment_id}")

    # Best approved run prior to this run
    prior_approved = [
        r for r in ledger.runs
        if r.approved and r.experiment_id != run.experiment_id
    ]
    higher_is_better = ledger.brief.higher_is_better
    best_prior = (
        (max(prior_approved, key=lambda r: r.score_mean) if higher_is_better else min(prior_approved, key=lambda r: r.score_mean))
        if prior_approved
        else None
    )

    workspace = agent_context.workspace
    direction = "higher" if higher_is_better else "lower"
    task_text = (
        f"Independently check the run {run.experiment_id}.\n"
        f"Run: {run.headline}\n"
        f"Run script: {run.script_path}\n"
        f"Saved validation predictions: {run.validation_predictions_path}\n"
        f"Saved test predictions: {run.test_predictions_path}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Primary metric: {ledger.brief.primary_metric} ({direction} is better)\n"
        f"Project folder: {workspace.project_folder}\n\n"
        "Instructions:\n"
        "1. Recompute the primary metric on each validation split using `from mlagent_tools import load_splits, load_val_preds_with_labels`.\n"
        "2. Recompute the score on fold -1 holdout examples if present.\n"
        "3. Check for target leakage or train-test overlap.\n\n"
        f"{VALIDATION_PREDICTIONS_FORMAT_TEXT}\n\n"
        f"{SPLITS_FORMAT_TEXT}"
    )

    coder_res = run_coder(
        task_text=task_text,
        result_tag=RESULT_TAG,
        result_keys_text=CHECKS_RESULT_KEYS,
        working_folder=workspace.evaluation_folder(run.experiment_id),
        script_name="check_run.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )

    checks: EvaluationChecks | None = None
    if coder_res.succeeded:
        try:
            checks = EvaluationChecks.model_validate(coder_res.result_data)
        except ValidationError:
            checks = None

    if checks is None:
        event_bus.emit(EventType.WARNING, AGENT_NAME, f"{run.experiment_id} approved without independent checks")
        run.verdict = "approve"
        run.verdict_reasons = ["Independent checks script failed; approved unverified."]
        run.checks = {"unverified": True}
    else:
        run.checks = checks.model_dump()
        eval_cfg = getattr(agent_context.config, "evaluation", None)
        hard_reasons = hard_check_reasons(run, checks, higher_is_better, best_prior, config=eval_cfg)
        if hard_reasons:
            run.verdict = "reject"
            run.verdict_reasons = hard_reasons
            event_bus.emit(EventType.DECISION, AGENT_NAME, f"{run.experiment_id} rejected by hard rules")
        else:
            prompt = VERDICT_PROMPT.format(
                digest=build_digest(AGENT_KEY, ledger),
                run_line=run.headline,
                checks_text=json.dumps(checks.model_dump(), indent=1),
            )
            answer = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, VerdictAnswer)
            run.verdict = answer.verdict
            run.verdict_reasons = answer.reasons

    # Compute gain vs best and counted_gain
    if run.verdict == "approve" and run.score_mean is not None:
        if best_prior and best_prior.score_mean is not None:
            gain = (run.score_mean - best_prior.score_mean) if higher_is_better else (best_prior.score_mean - run.score_mean)
            run.gain_vs_best = round(gain, 6)
            run.counted_gain = is_counted_gain(run.validation_scores, best_prior.validation_scores, higher_is_better)
        else:
            run.gain_vs_best = 0.0
            run.counted_gain = True
    else:
        run.counted_gain = False

    event_bus.emit(EventType.DECISION, AGENT_NAME, f"{run.experiment_id} {run.status}")
    return run
