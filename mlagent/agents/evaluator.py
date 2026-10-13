"""The Evaluator: an independent reviewer who does not trust the Modeler's numbers.

The Coder writes its own checking script: recompute the score from the saved validation
predictions, score the holdout again, look for leakage. Hard rules in code come first and can
reject a run without any LLM call. Otherwise the LLM reads the checks and gives the verdict.
"""

import json
from typing import Literal

from pydantic import BaseModel, ValidationError

from mlagent.agents.base import (
    AgentContext,
    AgentFailedError,
    build_common_prompt_values,
    display_name,
    render_prompt,
)
from mlagent.agents.coder import run_coder
from mlagent.digest import describe_brief, describe_run
from mlagent.events import EventType
from mlagent.settings import HOLDOUT_GAP_RELATIVE, SCORE_MISMATCH_RELATIVE_TOLERANCE
from mlagent.state import Evaluation, ExperimentRun, Ledger, Verdict

AGENT_KEY = "evaluator"
AGENT_NAME = display_name(AGENT_KEY)
RESULT_TAG = "CHECKS"
HOLDOUT_GAP_STD_COUNT = 3

CHECKS_RESULT_KEYS_TEXT = """\
- "recomputed_score": number or null, the mean over the validation splits of the primary \
metric, recomputed by your own code from the saved validation predictions and the true labels. \
Use null only when it truly cannot be computed
- "recomputed_holdout_score": number or null, the same for the holdout examples
- "leakage_flags": list of objects {"description": text with the numbers, "severity": "high", \
"medium" or "low"}. Use high for a single input that predicts the target almost perfectly, \
for the target or an id that encodes it, and for validation examples that also sit in the \
training part. Use medium for suspicious but explainable signals, and low for minor ones. An \
empty list means you checked and found nothing
- "train_validation_overlap": integer, how many validation examples also appear in the \
training part of the same split (same row, file, text, speaker, group or near-duplicate)
- "notes": list of short texts about anything else you noticed, for example predictions that \
are constant, outside the valid range or contain missing values, a score that is much better \
than a trivial baseline, a very uneven spread of scores between splits, or a check that could \
not run and why"""

VERDICT_PROMPT = """\
You are the Evaluator of an ML team. You are an independent reviewer. You do not trust the \
Modeler's numbers, and you never see the Modeler's reasoning. You only see what the checks \
found. Your job is to say whether the score is honest and usable, not whether it is high. A \
modest score that is honest is approved. A great score that is leaked is rejected.

{run_rules}

WHAT THE TEAM KNOWS:
{digest}

THE RUN:
{run_line}

THE INDEPENDENT CHECKS (already passed the automatic hard rules):
{checks_text}

Think it through:
1. Does the recomputed score match the logged score, and the holdout score agree with the \
validation scores? A small gap is normal noise. A large one means overfitting to the splits or \
a leak.
2. Is there any high leakage flag, or any overlap between training and validation? Real \
overlap, or an input that predicts the target almost perfectly for a reason the task does not \
explain, means the score is not honest. Reject.
3. Do medium and low flags have an innocent explanation in the task or the EDA (for example \
a column that really is a strong signal available at prediction time)? If the explanation is \
not in the information above, treat the flag as real. Several medium flags together count as \
a serious concern.
4. Is the score implausible for this task, for example near perfect when the team's earlier \
runs were far lower, or the same score on every split?
5. Are the predictions usable: right form for the metric, no missing values, not constant?
If a check could not run, say so. Never invent a result for it.

Decide. Approve when the result looks honest and usable. Reject when it looks leaked, \
overfitted or broken. Reject for dishonesty or breakage only. A weak result is for the \
Reviewer to judge, not you. When unsure, prefer reject over approve for leakage, because an \
approved leak misleads every later round.
Give 1 to 3 short reasons, also when you approve, each with the number or check it relies on.
Return JSON: verdict (approve or reject), reasons (list of texts)."""


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
    verdict: Verdict
    reasons: list[str]


def hard_check_reasons(
    run: ExperimentRun, checks: EvaluationChecks, higher_is_better: bool
) -> list[str]:
    """Rules that reject a run without asking the LLM. An empty list means the run passes."""
    reasons = []
    mean_score = run.score_mean
    std_score = run.score_std or 0.0

    if checks.recomputed_score is not None and mean_score is not None:
        tolerance = max(std_score, SCORE_MISMATCH_RELATIVE_TOLERANCE * abs(mean_score), 1e-9)
        if abs(checks.recomputed_score - mean_score) > tolerance:
            reasons.append(
                f"The recomputed score {checks.recomputed_score:.4f} differs from the logged "
                f"score {mean_score:.4f} by more than {tolerance:.4f}."
            )

    holdout_score = checks.recomputed_holdout_score
    if holdout_score is None:
        holdout_score = run.holdout_score
    if holdout_score is not None and mean_score is not None:
        shortfall = mean_score - holdout_score if higher_is_better else holdout_score - mean_score
        allowed_gap = max(HOLDOUT_GAP_STD_COUNT * std_score, HOLDOUT_GAP_RELATIVE * abs(mean_score))
        if shortfall > allowed_gap:
            reasons.append(
                f"The holdout score {holdout_score:.4f} is much worse than the validation "
                f"score {mean_score:.4f}."
            )

    for flag in checks.leakage_flags:
        if flag.severity == "high":
            reasons.append(f"High-severity leakage flag: {flag.description}")
    return reasons


def build_checks_task(ledger: Ledger, run: ExperimentRun, agent_context: AgentContext) -> str:
    workspace = agent_context.workspace
    direction = "higher" if ledger.plan.higher_is_better else "lower"
    return (
        f"Independently check the run {run.experiment_id}. Write your own scoring code. You may "
        "read the run's script to see which files it used, but do not import from it.\n"
        f"Run: {describe_run(ledger, run)}\n"
        f"Run script: {run.script_path}\n"
        f"Saved validation predictions: {run.validation_predictions_path or 'none'}\n"
        f"Saved test predictions: {run.test_predictions_path}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Features version used: {run.feature_version}\n"
        f"Primary metric: {ledger.plan.primary_metric} ({direction} is better)\n"
        f"Project folder (read only): {workspace.project_folder}\n\n"
        "Do these checks:\n"
        "1. Recompute the primary metric on each validation split from the saved validation "
        "predictions and the true labels, using the splits file. Report the mean.\n"
        "2. Recompute the score on the holdout examples (fold -1), if any and if predictions "
        "exist for them.\n"
        "3. Look for leakage. Tables: does any single column predict the target almost "
        "perfectly on its own? Other data: do validation examples also appear in the training "
        "part (same file, text, speaker, near-duplicate)? Does the model use an id, a file "
        "name or any field that gives the answer away?\n\n"
        f"Rules: write only inside {workspace.evaluation_folder(run.experiment_id)}. "
        "Never change files in the experiment folder.\n\n"
        f"{describe_brief(ledger)}"
    )


def build_unverified_evaluation(run: ExperimentRun, problem: str) -> Evaluation:
    """When the checks cannot run, approve with a visible warning instead of stalling the loop."""
    return Evaluation(
        experiment_id=run.experiment_id,
        verdict="approve",
        reasons=[f"The independent checks could not run ({problem}). Approved without checks."],
        checks={"unverified": True},
    )


def run_evaluator(ledger: Ledger, agent_context: AgentContext, experiment_id: str) -> Evaluation:
    event_bus = agent_context.event_bus
    run = ledger.find_run(experiment_id)
    if run is None or run.crashed:
        raise AgentFailedError(
            f"The Evaluator was asked to check {experiment_id}, which has no result."
        )
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, f"checking {experiment_id}")

    coder_result = run_coder(
        task_text=build_checks_task(ledger, run, agent_context),
        result_tag=RESULT_TAG,
        result_keys_text=CHECKS_RESULT_KEYS_TEXT,
        working_folder=agent_context.workspace.evaluation_folder(experiment_id),
        script_name="check_run.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )
    checks = None
    problem = coder_result.error_text or "no result"
    if coder_result.succeeded:
        try:
            checks = EvaluationChecks.model_validate(coder_result.result_data)
        except ValidationError as error:
            problem = f"the checks did not follow the contract: {error}"
    if checks is None:
        event_bus.emit(EventType.WARNING, AGENT_NAME, f"{experiment_id} approved WITHOUT checks")
        return build_unverified_evaluation(run, problem)

    hard_reasons = hard_check_reasons(run, checks, ledger.plan.higher_is_better)
    if hard_reasons:
        event_bus.emit(EventType.DECISION, AGENT_NAME, f"{experiment_id} rejected by a hard rule")
        return Evaluation(
            experiment_id=experiment_id,
            verdict="reject",
            reasons=hard_reasons,
            checks=checks.model_dump(),
        )

    common_values = build_common_prompt_values(AGENT_KEY, ledger, agent_context)
    prompt = render_prompt(
        VERDICT_PROMPT,
        **common_values,
        run_line=describe_run(ledger, run),
        checks_text=json.dumps(checks.model_dump(), indent=1),
    )
    answer = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, VerdictAnswer)
    verdict_word = "approved" if answer.verdict == "approve" else "rejected"
    event_bus.emit(EventType.DECISION, AGENT_NAME, f"{experiment_id} {verdict_word}")
    return Evaluation(
        experiment_id=experiment_id,
        verdict=answer.verdict,
        reasons=answer.reasons,
        checks=checks.model_dump(),
    )
