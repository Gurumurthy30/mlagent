"""Builds the short text each agent receives from the ledger. Plain code, no LLM involved.

Every agent gets only the sections it needs, most important first, cut to its character limit
from settings.DIGEST_CHAR_LIMITS.
"""

from collections.abc import Callable

from mlagent.settings import DIGEST_CHAR_LIMITS, MAX_ERROR_TEXT_CHARS
from mlagent.state import ExperimentRun, Ledger

BEST_RUNS_SHOWN = 5
LATEST_RUNS_SHOWN = 10
FINDINGS_SHOWN = 12
FEATURES_SHOWN = 20
CRASHES_SHOWN = 3
REVIEWS_SHOWN = 3
REJECTIONS_SHOWN = 5
FILES_SHOWN = 10
PLAYBOOK_CHARS = 1200
DESCRIPTION_CHARS = 100

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def trim_text(text: str, character_limit: int) -> str:
    if len(text) <= character_limit:
        return text
    marker = "\n...(trimmed)"
    return text[: character_limit - len(marker)] + marker


def format_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.4f}"


def describe_direction(higher_is_better: bool) -> str:
    return "higher is better" if higher_is_better else "lower is better"


def join_items(title: str, items: list[str]) -> str:
    if not items:
        return ""
    return f"{title}: " + "; ".join(items)


def describe_task(ledger: Ledger) -> str:
    metrics_text = ledger.user_metrics_text or "not given"
    return f"USER TASK: {ledger.user_task}\nUSER METRICS: {metrics_text}"


def describe_brief(ledger: Ledger) -> str:
    brief = ledger.brief
    if not brief.summary:
        return ""
    metric_text = f"{brief.primary_metric} ({describe_direction(brief.higher_is_better)})"
    lines = [
        "PROJECT BRIEF:",
        f"- summary: {brief.summary}",
        f"- modality: {brief.modality}, task family: {brief.task_family}",
        f"- target: {brief.target or 'none'}, id column: {brief.id_column or 'none'}",
        f"- primary metric: {metric_text}",
        f"- extra metrics: {', '.join(brief.extra_metrics) or 'none'}",
        f"- deliverable: {brief.deliverable_format}",
    ]
    for file_info in brief.files[:FILES_SHOWN]:
        lines.append(f"- file: {file_info.path} ({file_info.kind}, {file_info.role})")
    if len(brief.files) > FILES_SHOWN:
        lines.append(f"- ... and {len(brief.files) - FILES_SHOWN} more files")
    lines.append(join_items("- open questions", brief.open_questions))
    return "\n".join(line for line in lines if line)


def sorted_findings(ledger: Ledger) -> list:
    findings = sorted(ledger.eda.findings, key=lambda finding: PRIORITY_ORDER[finding.priority])
    return findings[:FINDINGS_SHOWN]


def describe_eda(ledger: Ledger) -> str:
    eda = ledger.eda
    if not eda.findings and not eda.feature_ideas:
        return ""
    lines = ["EDA FINDINGS:"]
    for finding in sorted_findings(ledger):
        lines.append(
            f"- {finding.finding_id} [{finding.priority}] {finding.statement}"
            f" -> {finding.implication}"
        )
    lines.append(join_items("DATA ISSUES", eda.issues))
    lines.append(join_items("LEAKAGE RISKS", eda.leakage_risks))
    lines.append(join_items("VALIDATION HINTS", eda.validation_hints))
    lines.append(join_items("MODEL HINTS", eda.model_hints))
    idea_texts = [f"{idea.description} ({idea.finding_id})" for idea in eda.feature_ideas]
    lines.append(join_items("FEATURE IDEAS", idea_texts))
    return "\n".join(line for line in lines if line)


def describe_eda_progress(ledger: Ledger) -> str:
    eda = ledger.eda
    open_items = [item for item, status in eda.checklist.items() if status == "open"]
    covered_count = sum(1 for status in eda.checklist.values() if status == "covered")
    lines = [
        f"EDA ROUNDS USED: {eda.eda_loops_used}",
        f"CHECKLIST: {covered_count} covered, {len(open_items)} open",
        join_items("OPEN CHECKLIST ITEMS", open_items),
    ]
    if eda.findings:
        lines.append("FINDINGS SO FAR:")
        for finding in sorted_findings(ledger):
            lines.append(f"- {finding.finding_id} [{finding.priority}] {finding.statement}")
    return "\n".join(line for line in lines if line)


def describe_leakage_risks(ledger: Ledger) -> str:
    return join_items("KNOWN LEAKAGE RISKS", ledger.eda.leakage_risks)


def describe_plan(ledger: Ledger) -> str:
    plan = ledger.plan
    if not plan.primary_metric:
        return ""
    metric_text = f"{plan.primary_metric} ({describe_direction(plan.higher_is_better)})"
    validation_text = plan.validation_scheme
    if plan.validation_reason:
        validation_text += f" ({plan.validation_reason})"
    target_text = plan.target_score if plan.target_score is not None else "none"
    lines = [
        "PLAN:",
        f"- primary metric: {metric_text}",
        f"- extra metrics: {', '.join(plan.extra_metrics) or 'none'}",
        f"- validation: {validation_text}",
        f"- splits file: {plan.splits_path}" if plan.splits_path else "",
        f"- baseline: {plan.baseline_description}" if plan.baseline_description else "",
        f"- target score: {target_text}",
        f"- use GPU: {'yes' if plan.use_gpu else 'no'}",
        join_items("- risks", plan.risks),
        f"- playbook: {trim_text(plan.playbook, PLAYBOOK_CHARS)}" if plan.playbook else "",
    ]
    return "\n".join(line for line in lines if line)


def describe_features(ledger: Ledger) -> str:
    if not ledger.features:
        return ""
    lines = [f"FEATURES TRIED (current features file is version {ledger.current_feature_version}):"]
    for feature in ledger.features[-FEATURES_SHOWN:]:
        gain_text = "gain n/a" if feature.score_gain is None else f"gain {feature.score_gain:+.4f}"
        kept_text = "kept" if feature.kept else "dropped"
        lines.append(f"- {feature.feature_id} {feature.name}: {kept_text}, {gain_text}")
    return "\n".join(lines)


def describe_run(ledger: Ledger, run: ExperimentRun) -> str:
    evaluation = ledger.find_evaluation(run.experiment_id)
    if run.crashed:
        status = "crashed"
    elif evaluation is None:
        status = "not evaluated"
    else:
        status = "approved" if evaluation.verdict == "approve" else "rejected"
    spread_text = "" if run.score_std is None else f" +/- {run.score_std:.4f}"
    description = trim_text(run.description, DESCRIPTION_CHARS)
    return (
        f"- {run.experiment_id} | {run.focus} | {run.model_family} | "
        f"score {format_score(run.score_mean)}{spread_text} | {status} | {description}"
    )


def describe_runs(ledger: Ledger) -> str:
    if not ledger.runs:
        return ""
    best_runs = sorted(
        ledger.approved_runs(),
        key=lambda run: run.score_mean,
        reverse=ledger.plan.higher_is_better,
    )[:BEST_RUNS_SHOWN]
    latest_runs = ledger.runs[-LATEST_RUNS_SHOWN:]
    shown_ids = {run.experiment_id for run in best_runs + latest_runs}
    shown_runs = [run for run in ledger.runs if run.experiment_id in shown_ids]

    best_run = ledger.best_approved_run()
    best_text = best_run.experiment_id if best_run else "none yet"
    lines = [f"EXPERIMENT RUNS (showing {len(shown_runs)} of {len(ledger.runs)}):"]
    lines.append(f"BEST APPROVED RUN: {best_text}")
    lines.extend(describe_run(ledger, run) for run in shown_runs)
    return "\n".join(lines)


def describe_crashes(ledger: Ledger) -> str:
    crashes = ledger.recent_crashes(CRASHES_SHOWN)
    if not crashes:
        return ""
    lines = ["RECENT CRASHES (do not repeat them):"]
    for run in crashes:
        lines.append(
            f"- {run.experiment_id} {run.model_family}: {run.error[-MAX_ERROR_TEXT_CHARS:]}"
        )
    return "\n".join(lines)


def describe_evaluations(ledger: Ledger) -> str:
    rejections = [e for e in ledger.evaluations if e.verdict == "reject"][-REJECTIONS_SHOWN:]
    if not rejections:
        return ""
    lines = ["RECENT REJECTIONS BY THE EVALUATOR:"]
    for evaluation in rejections:
        lines.append(f"- {evaluation.experiment_id}: {'; '.join(evaluation.reasons)}")
    return "\n".join(lines)


def describe_reviews(ledger: Ledger) -> str:
    if not ledger.reviews:
        return ""
    lines = ["LATEST REVIEWS:"]
    for review in ledger.reviews[-REVIEWS_SHOWN:]:
        gain_text = "yes" if review.counted_gain else "no"
        lines.append(
            f"- after round {review.iteration}: ran {review.reviewed_focus} (counted gain: "
            f"{gain_text}), then chose {review.focus}. {review.reason}"
        )
        for instruction in review.instructions:
            lines.append(f"  instruction: {instruction}")
    return "\n".join(lines)


SectionBuilder = Callable[[Ledger], str]

SECTION_BUILDERS_BY_AGENT: dict[str, list[SectionBuilder]] = {
    "data_engineer": [describe_task],
    "data_analyst": [describe_task, describe_brief, describe_eda_progress],
    "ml_lead": [describe_task, describe_brief, describe_eda],
    "feature_engineer": [
        describe_task,
        describe_plan,
        describe_brief,
        describe_reviews,
        describe_features,
        describe_eda,
        describe_runs,
        describe_crashes,
    ],
    "modeler": [
        describe_task,
        describe_plan,
        describe_brief,
        describe_reviews,
        describe_runs,
        describe_crashes,
        describe_features,
        describe_eda,
    ],
    "evaluator": [describe_plan, describe_leakage_risks, describe_runs, describe_evaluations],
    "tuner": [describe_plan, describe_runs, describe_evaluations, describe_crashes],
    "reviewer": [
        describe_task,
        describe_plan,
        describe_brief,
        describe_runs,
        describe_evaluations,
        describe_reviews,
        describe_features,
        describe_crashes,
        describe_eda,
    ],
    "submitter": [
        describe_task,
        describe_brief,
        describe_plan,
        describe_runs,
        describe_evaluations,
    ],
    "coder": [describe_plan, describe_crashes],
}


def build_digest(agent_name: str, ledger: Ledger) -> str:
    section_builders = SECTION_BUILDERS_BY_AGENT[agent_name]
    section_texts = [build_section(ledger) for build_section in section_builders]
    digest = "\n\n".join(text for text in section_texts if text)
    return trim_text(digest, DIGEST_CHAR_LIMITS[agent_name])


def build_progress_text(ledger: Ledger) -> str:
    """A short readable summary for the user, saved as memory/progress.md."""
    best_run = ledger.best_approved_run()
    if best_run:
        best_score = format_score(best_run.score_mean)
        best_text = f"{best_run.experiment_id} ({best_run.model_family}), score {best_score}"
    else:
        best_text = "none yet"
    last_review = ledger.reviews[-1] if ledger.reviews else None
    lines = [
        "# mlagent progress",
        "",
        f"- status: {ledger.status} (phase: {ledger.phase}, focus: {ledger.focus})",
        f"- experiment rounds finished: {ledger.iteration}",
        f"- runs: {len(ledger.runs)}, approved: {len(ledger.approved_runs())}",
        f"- best approved run: {best_text}",
        f"- stop reason: {ledger.stop_reason or 'not stopped yet'}",
        "",
        "## Task",
        ledger.user_task,
        "",
        f"Metrics asked for: {ledger.user_metrics_text or 'not given'}",
    ]
    if last_review:
        lines += ["", "## Latest review", f"Focus {last_review.focus}: {last_review.reason}"]
    return "\n".join(lines) + "\n"
