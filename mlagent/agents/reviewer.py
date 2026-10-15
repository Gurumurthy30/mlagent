"""The Reviewer: senior scientist who steers the loop and writes the final report.

Per-round: reads diagnosis of the best run and team memory, then chooses next focus.
Final: writes the comprehensive final report from the team's entire trajectory.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from mlagent.agents.base import AgentContext, facts_to_text
from mlagent.agents.coder import run_coder
from mlagent.digest import build_digest, build_report_facts
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
You are the Reviewer of an autonomous ML team, the senior engineer deciding what the team does next.
The next agent only sees your instructions or question. Include exact numbers, segment names, and actions.

GOAL:
Determine the highest-leverage focus for the next round (features, model, tune, data, or finish)
and summarize the lesson learned this round.

SITUATION:
{digest}

DIAGNOSIS OF THE BEST RUN:
{diagnosis_text}

HOW TO THINK:
- Before choosing a focus, check if a high-priority EDA finding or untried feature idea remains unacted upon; it can yield higher gain than another model change.
- If error is concentrated in specific classes, groups, or ranges: focus on "features" targeting those segments.
- If features gave no gain repeatedly, or model is under-fitting globally: focus on "model" with a distinct family.
- If a strong model family has stable features and parameters were never searched: focus on "tune".
- If validation behavior is anomalous or reveals an uninspected data issue: focus on "data" with a concrete question.
- If improvements have plateaued and minimum rounds are completed: focus on "finish".

YOU MAY:
- Reopen data analysis ("data") if you have a specific testable question.
- Recommend finishing early if performance has saturated cleanly.

OUTPUT:
Return JSON:
- assessment: 3-5 sentences diagnosing current bottlenecks and trajectory.
- lesson: one sentence stating what this round taught the team.
- focus: one of "features", "model", "tune", "data", "finish".
- reason: one sentence explaining why this focus is chosen.
- instructions: list of up to 5 concrete action points for the next agent (empty for finish or data).
- question: specific data question (only when focus is "data")."""


class ReviewDraft(BaseModel):
    assessment: str = ""
    lesson: str = ""
    focus: ReviewFocus = "features"
    reason: str = ""
    instructions: list[str] = []
    question: str = ""


REPORT_PROMPT = """\
You are the Reviewer writing the official Final Technical Report for the machine learning project.
Write plain Markdown without surrounding JSON or fences.
Lead with results, use ONLY the facts given, copy numbers exactly, and write "Not recorded." for any missing fact.
Do not use hype. Keep total length under 900 words.

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
        focus=draft.focus,
        reason=draft.reason,
        instructions=draft.instructions[:MAX_REVIEW_INSTRUCTIONS],
        lesson=draft.lesson.strip(),
        question=draft.question if draft.focus == "data" else "",
        reviewed_focus=ledger.focus,
    )

    review = apply_reviewer_guards(ledger, review)
    event_bus.emit(EventType.DECISION, AGENT_NAME, f"next focus: {review.focus}. {review.reason}")
    if review.guard_note:
        event_bus.emit(EventType.WARNING, AGENT_NAME, review.guard_note)
    event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, f"round {reviewed_round} reviewed")
    return review


def run_report(ledger: Ledger, agent_context: AgentContext) -> tuple[str, str]:
    """Generate final report from facts. Returns (report_markdown, headline)."""
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "writing final report")

    facts = build_report_facts(ledger)
    prompt = REPORT_PROMPT.format(facts=facts)
    report_markdown = agent_context.llm_client.ask_text(AGENT_NAME, prompt)

    # Extract 'In brief' section for headline
    headline = ""
    match = re.search(r"## In brief\s*\n(.*?)(?=\n## |\Z)", report_markdown, re.DOTALL)
    if match:
        brief_lines = [l.strip() for l in match.group(1).splitlines() if l.strip()]
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
