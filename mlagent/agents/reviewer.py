"""The Reviewer: senior scientist who steers the loop and writes the final report.

Per-round: reads diagnosis of the best run and team memory, then chooses next focus.
Final: writes the comprehensive final report from the team's entire trajectory.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from mlagent.agents.base import AgentContext, facts_to_text
from mlagent.agents.coder import run_coder
from mlagent.digest import build_digest, build_report_facts, shorten_text
from mlagent.events import EventType
from mlagent.script_contract import SPLITS_FORMAT_TEXT, VALIDATION_PREDICTIONS_FORMAT_TEXT
from mlagent.settings import MAX_REVIEW_INSTRUCTIONS
from mlagent.state import ExperimentRun, Ledger, Review, ReviewFocus
from mlagent.stop_rules import apply_reviewer_guards

AGENT_KEY = "reviewer"
AGENT_NAME = "Reviewer"
RESULT_TAG = "DIAG"

DIAG_RESULT_KEYS = """\
- "weak_spots": list of at most 5 texts detailing worst-performing segments with counts and metric scores
- "comparison": text comparing current best run with previous best
- "notes": short observations covering split spread, holdout gap, and prediction properties"""

REVIEW_PROMPT = """\
You are the Reviewer of an autonomous Kaggle Grandmaster ML team, the senior engineer steering the iterative development \
loop. The next agent receives ONLY your instructions or question—make them concrete, actionable, and \
completely self-contained.

DECISION RULES (in priority order):
1. TREE-SEARCH ROUTING & PORTFOLIO EXPLORATION:
   - Check the `MODEL FAMILY LEADERBOARD` and `planned_model_families` in SITUATION: if distinct model families \
from the plan have not yet been trained, prioritize `model` exploration before tuning or adding more features.
   - "features": If feature version is stable and model performance is bottlenecked by specific subgroups, error segments, \
or unexploited EDA feature ideas. Instruct Feature Engineer with exact column names and mathematical transforms.
   - "model": If feature version was upgraded or current model family is not improving, switch to an untried model family \
(`lightgbm` -> `catboost` -> `xgboost` -> `ridge`/`mlp`) to build ensemble diversity.
   - "tune": If a top-performing model family is established, features are stable, and hyperparameters have never been searched.
   - "data": If cross-validation behavior is erratic, holdout gap is large, or performance has stalled across 2 rounds without \
counted gain, reopen EDA to inspect worst out-of-fold errors in `val_predictions.csv`.
   - "finish": Only when improvements have saturated cleanly across multiple model families and minimum rounds are completed.
2. HYPOTHESIS LEDGER & LESSON RECORDING (`hypothesis_verdict`):
   - Explicitly classify the latest round's experiment as `validated` (kept), `refuted` (clean run, no CV gain—do not repeat), \
`bug` (script crashed or failed verification—repaired), or `baseline`.
   - Record one crisp sentence in `lesson` stating what empirical fact this round established.
3. STANDALONE, ACTIONABLE INSTRUCTIONS:
   - Instructions must name exact column operations, formulas, or model hyperparameters. Never give generic advice.

WORKED EXAMPLE (Focus: Model):
- assessment: "Feature v2 added group aggregations and improved LightGBM CV to 0.8120 (+0.0078 gain). However, CatBoost and XGBoost have not been explored yet, limiting ensemble diversity."
- bottleneck: "Single model family reliance on LightGBM; need diverse gradient boosting tree structures."
- hypothesis_verdict: "validated"
- lesson: "Group-level luxury spending aggregations successfully resolved the multi-passenger error segment."
- reason: "Explore untried model family CatBoost on features v2 to establish cross-family ensemble diversity."
- focus: "model"
- instructions: ["Train a CatBoostClassifier with depth=6, iterations=1500, learning_rate=0.04 using features v2.", \
"Apply out-of-fold threshold tuning to maximize primary_metric."]
- question: ""

SITUATION:
{digest}

DIAGNOSIS OF THE BEST RUN:
{diagnosis_text}

OUTPUT:
Return JSON with exactly these fields in order:
- assessment: 3-5 sentences diagnosing current performance, trajectory, and progress against compute budget.
- bottleneck: the specific error segment, fold instability, or pipeline limitation identified.
- hypothesis_verdict: one of "validated", "refuted", "bug", "baseline".
- lesson: exactly one sentence stating what this round taught the team.
- reason: one clear sentence explaining why this focus is the highest-leverage next step.
- focus: one of "features", "model", "tune", "data", "finish".
- instructions: list of up to 5 self-contained, concrete instructions for the next agent (empty for "finish" or "data").
- question: specific data question (required ONLY if focus is "data", otherwise "")."""


class ReviewDraft(BaseModel):
    assessment: str = ""
    bottleneck: str = ""
    hypothesis_verdict: Literal["validated", "refuted", "bug", "baseline"] = "baseline"
    lesson: str = ""
    reason: str = ""
    focus: ReviewFocus = "features"
    instructions: list[str] = []
    question: str = ""


REPORT_PROMPT = """\
You are the Reviewer writing the official Final Technical Report for the machine learning project.
Write plain Markdown without surrounding JSON or fences.
Lead with results, use ONLY the facts given, copy numbers exactly, and write "Not recorded." for any missing fact.
Do not use hype. Keep total length under {max_words} words.

Trace every claim about a cause to a finding ID or lesson in the facts; otherwise write "the team's notes suggest".

SECTIONS (use these exact headings in this order):
# Technical Report: [Task and Outcome]

## In brief
At most 3 sentences: what was built, final score compared with first run, whether to rely on it, and predictions path.

## Key numbers
3 to 5 bullet points with key numbers strictly from the facts.

## What the data showed
The EDA summary and key findings that mattered. For each, state what the team did and what happened. Note any findings never acted upon.

## What the team did
Input preparation, model exploration, and tuning in chronological order. State what helped, what failed, and Evaluator rejections with reasons.

## The final predictions
File path, format, row count, winning model, and deployment instructions.

## How far to trust it
Cross-validation spread, holdout agreement, checks flagged, and practical meaning of the metric.

## Caveats and next steps
Up to 5 actionable bullet points tied to facts (unanswered EDA questions or untried ideas). The very first step goes last.

FACTS:
{facts}"""


def run_diagnosis(
    ledger: Ledger, best_run: ExperimentRun, reviewed_round: int, agent_context: AgentContext
) -> str:
    """Run diagnosis script on the best run's predictions."""
    workspace = agent_context.workspace
    direction = "higher" if ledger.brief.higher_is_better else "lower"
    task_text = (
        f"Diagnose where best run {best_run.experiment_id} fails, using saved validation predictions.\n"
        f"Validation predictions: {best_run.validation_predictions_path}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Primary metric: {ledger.brief.primary_metric} ({direction} is better)\n"
        f"Project folder: {workspace.project_folder}\n\n"
        "Instructions:\n"
        "- Use `from mlagent_tools import load_splits, load_val_preds_with_labels`.\n"
        "- Identify worst-performing segments (subsets with >= 20 examples where error is high).\n"
        "- Look at confidence, calibration, and residuals.\n\n"
        f"{VALIDATION_PREDICTIONS_FORMAT_TEXT}\n\n"
        f"{SPLITS_FORMAT_TEXT}"
    )

    coder_res = run_coder(
        task_text=task_text,
        result_tag=RESULT_TAG,
        result_keys_text=DIAG_RESULT_KEYS,
        working_folder=workspace.artifacts_folder / AGENT_KEY,
        script_name=f"diagnose_round_{reviewed_round}.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )
    if not coder_res.succeeded or not isinstance(coder_res.result_data, dict):
        return "Diagnosis script did not produce results."
    return facts_to_text(coder_res.result_data)


def run_reviewer(ledger: Ledger, agent_context: AgentContext) -> Review:
    """Review the finished round and choose the next focus."""
    event_bus = agent_context.event_bus
    reviewed_round = ledger.iteration
    next_iteration = reviewed_round + 1
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, f"reviewing round {reviewed_round}")

    best_run = ledger.best_approved_run()
    if best_run is not None and best_run.validation_predictions_path:
        diagnosis_text = run_diagnosis(ledger, best_run, reviewed_round, agent_context)
    else:
        diagnosis_text = "No approved run with validation predictions available yet."

    digest = build_digest(AGENT_KEY, ledger)
    prompt = REVIEW_PROMPT.format(digest=digest, diagnosis_text=diagnosis_text)
    draft = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, ReviewDraft)

    review = Review(
        iteration=next_iteration,
        best_experiment_id=best_run.experiment_id if best_run else "",
        assessment=draft.assessment,
        bottleneck=draft.bottleneck,
        hypothesis_verdict=draft.hypothesis_verdict,
        focus=draft.focus,
        reason=draft.reason,
        instructions=draft.instructions[:MAX_REVIEW_INSTRUCTIONS],
        lesson=draft.lesson.strip(),
        question=draft.question if draft.focus == "data" else "",
        reviewed_focus=ledger.focus,
    )

    review = apply_reviewer_guards(ledger, review)
    short_reason = shorten_text(review.reason, 120)
    event_bus.emit(EventType.DECISION, AGENT_NAME, f"next = {review.focus}. Why: {short_reason}")
    if review.guard_note:
        event_bus.emit(EventType.WARNING, AGENT_NAME, review.guard_note)
    event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, f"round {reviewed_round} reviewed")
    return review


def run_report(ledger: Ledger, agent_context: AgentContext) -> tuple[str, str]:
    """Generate final report from facts. Returns (report_markdown, headline)."""
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "writing final report")

    max_words = agent_context.config.report.max_words
    facts = build_report_facts(ledger, char_limit=agent_context.config.llm.context)
    prompt = REPORT_PROMPT.format(max_words=max_words, facts=facts)
    report_markdown = agent_context.llm_client.ask_text(AGENT_NAME, prompt)

    # Extract 'In brief' section for headline
    headline = ""
    match = re.search(r"## In brief\s*\n(.*?)(?=\n## |\Z)", report_markdown, re.DOTALL)
    if match:
        brief_lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
        headline = " ".join(brief_lines)

    if not headline:
        best = ledger.best_approved_run()
        score_val = f"{best.score_mean:.4f}" if best and best.score_mean is not None else "none"
        headline = f"Delivered model {best.model_family if best else 'unknown'} with score {score_val}."

    # Append runs table if not included
    if "| Run |" not in report_markdown and "## Experiment Runs" not in report_markdown:
        table_lines = [
            "",
            "## Experiment Runs",
            "",
            "| Run | Model | Features | Score | Gain vs Best | Status | Summary |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in ledger.runs:
            s = f"{r.score_mean:.4f}" if r.score_mean is not None else "n/a"
            g = f"{r.gain_vs_best:+.4f}" if r.gain_vs_best is not None else "-"
            table_lines.append(
                f"| {r.experiment_id} | {r.model_family} | v{r.feature_version} | {s} | {g} | {r.status} | {r.summary} |"
            )
        report_markdown += "\n" + "\n".join(table_lines) + "\n"

    event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, "final report generated")
    return report_markdown, headline