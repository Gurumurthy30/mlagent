"""The Evaluator: an independent reviewer who verifies run honesty and checks for leakage.

The Coder writes an independent verification script to recompute the score and check leakage.
Code rules catch hard failures immediately. Otherwise the LLM assesses findings.
"""

from __future__ import annotations

import json
from pathlib import Path
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
SCORE_TOLERANCE_RELATIVE = 0.05
HOLDOUT_GAP_STD_MULTIPLIER = 4.0
HOLDOUT_GAP_RELATIVE_LIMIT = 0.08
LARGE_SCORE_JUMP_STD_MULTIPLIER = 4.0
TRAIN_VAL_OVERLAP_TOLERANCE = 0

CHECKS_RESULT_KEYS = """\
- "recomputed_score": number or null, mean across validation splits recomputed from predictions and true labels
- "recomputed_holdout_score": number or null, score on holdout examples (null if no holdout)
- "leakage_flags": list of {"description": text, "severity": "high"|"medium"|"low"}
- "train_validation_overlap": integer, number of validation examples in train split
- "notes": list of short observations"""

VERDICT_PROMPT = """\
You are the Evaluator of an autonomous Kaggle Grandmaster ML team. Your purpose is independent verification: \
auditing model runs for data leakage, split integrity, fold stability, and submission validity. \
You do not judge model architecture taste; you judge whether the validation score and test predictions are honest, leak-free, and usable.

DECISION RULES (in priority order):
1. HARD REJECTION CRITERIA (reject ONLY when backed by concrete evidence in INDEPENDENT CHECKS):
   - TRUE DATA LEAKAGE: Any "high" severity leakage flag demonstrating target leakage (e.g., target column used in features, \
fitting target encoders/scalers on full data before splitting, or `train_validation_overlap > 0`) -> REJECT.
   - CORRUPTED OR CONSTANT PREDICTIONS: Validation or test predictions contain `NaN`/`Inf`, have zero variance \
(constant predictions across rows), or `test_predictions.csv` fails deliverable row-count/format checks -> REJECT.
   - SEVERE HOLDOUT COLLAPSE: Holdout score degrades relative to CV mean well beyond cross-validation noise, proving severe validation overfitting -> REJECT.
   - UNEXPLAINED SCORE MISMATCH: `recomputed_score` differs materially from the run's reported score beyond rounding/threshold tolerance -> REJECT.
2. DISCERN TRUE LEAKAGE FROM LEGITIMATE DOMAIN SIGNAL:
   - Do NOT reject a run merely because a legitimate domain feature (like `CryoSleep`, `total_spent`, or decomposed `Cabin` deck) \
has strong predictive power, provided `fit_preparation` was fit strictly on training folds and holdout performance agrees with CV.
   - A single "medium" or "low" flag with a sound domain explanation and consistent holdout score should be APPROVED.
3. APPROVE HONEST REGRESSIONS:
   - If a run is leak-free, reproducible, and produces valid predictions, APPROVE it even if its CV score is slightly lower \
than the current best run. Honest runs from new model families (e.g. CatBoost, XGBoost, Ridge) are essential for final ensemble blending and Reviewer credit assignment!

WORKED EXAMPLE:
- Run: "experiment_003 | lightgbm | 0.9998"
- Checks: "leakage_flags: [{{'description': 'column target_copy has AUC 1.000 and is identical to target', 'severity': 'high'}}], train_validation_overlap: 0"
- Correct Output:
  * thinking: "The run achieves near-perfect score 0.9998 because target_copy has AUC 1.000. This is a direct target leak."
  * reasons: ["High leakage flag: column target_copy has AUC 1.000, leaking the ground-truth label."]
  * verdict: "reject"

SITUATION:
{digest}

THE RUN:
{run_line}

INDEPENDENT CHECKS:
{checks_text}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 2-4 sentences reviewing the independent checks, comparing recomputed CV with holdout, verifying test prediction integrity, and evaluating any leakage flags.
- reasons: list of 1 to 3 short factual sentences citing exact numbers from the checks (required for both approve and reject).
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
    filtered_flags = []
    for f in checks.leakage_flags:
        desc_lower = f.description.lower()
        if ("val_predictions" in desc_lower or "validation predictions" in desc_lower) and ("missing" in desc_lower or "not found" in desc_lower):
            if run.validation_predictions_path and Path(run.validation_predictions_path).is_file():
                continue
        if ("test_predictions" in desc_lower or "test predictions" in desc_lower) and ("missing" in desc_lower or "not found" in desc_lower):
            if run.test_predictions_path and Path(run.test_predictions_path).is_file():
                continue
        filtered_flags.append(f)

    high_flags = [f for f in filtered_flags if f.severity == "high"]
    if high_flags:
        reasons.extend(f"High-severity leakage: {f.description}" for f in high_flags)

    medium_flags = [f for f in filtered_flags if f.severity == "medium"]
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
        if jump > jump_std_mult * max(std_score, 1e-6) and filtered_flags:
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
    event_bus.emit(
        EventType.STATUS,
        AGENT_NAME,
        f"validating {run.experiment_id} ({run.model_family})",
        {"reason": "running leakage checks and cross-validation verification"},
    )

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
    eval_folder = workspace.evaluation_folder(run.experiment_id)
    task_text = (
        f"Independently check the run {run.experiment_id}.\n"
        f"Run: {run.headline}\n"
        f"Post-processing applied by run: {run.post_processing or 'none'}\n"
        f"Run script: {run.script_path}\n"
        f"Saved validation predictions (ABSOLUTE PATH): {run.validation_predictions_path}\n"
        f"Saved test predictions (ABSOLUTE PATH): {run.test_predictions_path}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Primary metric: {ledger.brief.primary_metric} ({direction} is better)\n"
        f"Deliverable format: {ledger.brief.deliverable_format}\n"
        f"Project folder: {workspace.project_folder}\n\n"
        "CRITICAL FILE PATH WARNING:\n"
        f"Your check script executes inside '{eval_folder}'. Do NOT look for 'val_predictions.csv' or 'test_predictions.csv' "
        f"in the current working directory! Always use the exact absolute paths {run.validation_predictions_path} "
        f"and {run.test_predictions_path} when loading or checking predictions.\n\n"
        "VERIFICATION INSTRUCTIONS:\n"
        "1. RECOMPUTE CV & HOLDOUT SCORES ACCURATELY:\n"
        f"   - Load validation predictions from '{run.validation_predictions_path}' and join with ground truth (`from mlagent_tools import load_splits, load_val_preds_with_labels`).\n"
        "   - For probability/ranking metrics (`roc_auc`, `pr_auc`, `log_loss`, `brier_score`), compute the metric on each split (`fold >= 0`) using the `proba_*` column(s) (or `prediction` if continuous probabilities are stored there).\n"
        "   - For discrete classification metrics (`accuracy`, `f1`, `f1_macro`, `f1_weighted`, `mcc`, `qwk`) and regression metrics (`rmse`, `rmsle`, `mae`, `r2`), compute the metric on each split (`fold >= 0`) using the `prediction` column.\n"
        "   - Recompute `recomputed_score` (mean across CV folds `fold >= 0`) and `recomputed_holdout_score` (on `fold == -1` if present, else `null`).\n"
        "2. CHECK TEST PREDICTIONS & SUBMISSION INTEGRITY:\n"
        f"   - Inspect test predictions at '{run.test_predictions_path}': verify the file exists on disk, has zero `NaN`/`Inf` values, has non-zero variance (not constant predictions), and matches the test set row count.\n"
        f"   - Only if '{run.test_predictions_path}' does not exist on disk, has `NaN`s, is constant, or has the wrong row count, append a `{{'description': ..., 'severity': 'high'}}` entry to `leakage_flags`.\n"
        "3. AUDIT FOR TRUE DATA LEAKAGE:\n"
        "   - Inspect `run.script_path` and predictions to verify that `fit_preparation` / encoders / target encodings were fit inside each training fold (`train_validation_overlap == 0`) and the target column was not included as an input feature.\n"
        "   - Only flag `severity: 'high'` for genuine leakage, train/val overlap, or corrupted predictions—not for legitimate domain features.\n\n"
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
    verdict_label = "approved" if run.approved else f"rejected ({run.verdict_reasons[0] if run.verdict_reasons else 'failed validation'})"
    score_display = f"{run.score_mean:.4f}" if run.score_mean is not None else "no score"
    event_bus.emit(
        EventType.RESULT,
        AGENT_NAME,
        f"{run.experiment_id}: {verdict_label} — score {score_display}",
    )
    return run