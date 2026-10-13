"""The Submitter: packages the result.

It tries an ensemble of the best approved runs and keeps it only if it beats the best single run
by the counted-gain rule. It copies the final predictions into the report folder, checks them,
and writes report.md. Without an approved run it writes only the report.
"""

import shutil
from pathlib import Path

from pydantic import BaseModel, ValidationError

from mlagent.agents.base import (
    AgentContext,
    build_common_prompt_values,
    display_name,
    render_prompt,
)
from mlagent.agents.coder import run_coder
from mlagent.deliverable_checks import check_predictions_file
from mlagent.digest import describe_brief, describe_run, format_score
from mlagent.events import EventType
from mlagent.llm_client import LlmUnavailableError
from mlagent.scoring import all_finite, is_counted_gain
from mlagent.settings import ENSEMBLE_MAX_RUNS, PREDICTIONS_FILE_STEM
from mlagent.state import ExperimentRun, FinalResult, Ledger

AGENT_KEY = "submitter"
AGENT_NAME = display_name(AGENT_KEY)
RESULT_TAG = "BLEND"
MAX_FINDINGS_IN_REPORT = 5

BLEND_RESULT_KEYS_TEXT = """\
- "validation_scores": list of numbers, the primary metric of the blend on each validation \
split, or null when these predictions cannot be blended sensibly
- "test_predictions_path": path of the blended predictions for the data to predict, in the \
deliverable format ("" when they cannot be blended)
- "method": one short text on how the runs were combined, or why they cannot be"""

REPORT_PROMPT = """\
You are the Submitter of an ML team. Write the final report for the user, in Markdown.

{run_rules}

THE FACTS (the only source you may use):
{facts_text}

Write these sections: Summary (3 sentences), What the team did, Results, The final \
predictions, Caveats and next steps. Use only the facts above. Never invent a number. Say \
plainly what is uncertain. Keep it under 600 words. Do not repeat the run table, it is added \
below your text by the program."""


class BlendResult(BaseModel):
    validation_scores: list[float] | None = None
    test_predictions_path: str = ""
    method: str = ""


def pick_ensemble_runs(ledger: Ledger) -> list[ExperimentRun]:
    candidates = [run for run in ledger.approved_runs() if run.validation_predictions_path]
    ranked_runs = sorted(
        candidates, key=lambda run: run.score_mean, reverse=ledger.plan.higher_is_better
    )
    return ranked_runs[:ENSEMBLE_MAX_RUNS]


def build_blend_task(ledger: Ledger, runs: list[ExperimentRun], agent_context: AgentContext) -> str:
    workspace = agent_context.workspace
    direction = "higher" if ledger.plan.higher_is_better else "lower"
    run_lines = "\n".join(
        f"- {describe_run(ledger, run)}\n"
        f"  validation predictions: {run.validation_predictions_path}\n"
        f"  test predictions: {run.test_predictions_path}"
        for run in runs
    )
    return (
        "Blend the predictions of these runs and score the blend on the same validation "
        "splits.\n"
        f"{run_lines}\n\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Primary metric: {ledger.plan.primary_metric} ({direction} is better)\n"
        f"Deliverable format of the predictions: {ledger.brief.deliverable_format}\n"
        f"Project folder (read only): {workspace.project_folder}\n"
        f"Write the blended test predictions inside {workspace.artifacts_folder / AGENT_KEY}.\n"
        "Use a simple blend (the average of probabilities or values, or a vote for labels). "
        "Score the blend on every validation split like the single runs were scored. If these "
        "predictions cannot be blended sensibly (for example generated text or masks), print "
        "validation_scores as null and explain in method.\n\n"
        f"{describe_brief(ledger)}"
    )


def try_ensemble(
    ledger: Ledger, best_run: ExperimentRun, agent_context: AgentContext
) -> tuple[Path | None, str]:
    """Return the blended predictions path and a description, or (None, '') when not kept."""
    ensemble_runs = pick_ensemble_runs(ledger)
    if len(ensemble_runs) < 2:
        return None, ""

    coder_result = run_coder(
        task_text=build_blend_task(ledger, ensemble_runs, agent_context),
        result_tag=RESULT_TAG,
        result_keys_text=BLEND_RESULT_KEYS_TEXT,
        working_folder=agent_context.workspace.artifacts_folder / AGENT_KEY,
        script_name="blend_runs.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )
    if not coder_result.succeeded:
        return None, ""
    try:
        blend = BlendResult.model_validate(coder_result.result_data)
    except ValidationError:
        return None, ""

    blend_predictions_path = (
        Path(blend.test_predictions_path) if blend.test_predictions_path else None
    )
    if (
        blend.validation_scores is None
        or not all_finite(blend.validation_scores)
        or blend_predictions_path is None
        or not blend_predictions_path.exists()
    ):
        return None, ""
    if not is_counted_gain(
        blend.validation_scores, best_run.validation_scores, ledger.plan.higher_is_better
    ):
        return None, ""

    run_ids = ", ".join(run.experiment_id for run in ensemble_runs)
    return blend_predictions_path, f"blend of {run_ids}: {blend.method}"


def copy_predictions(source_path: Path, agent_context: AgentContext) -> Path:
    report_folder = agent_context.workspace.report_folder
    report_folder.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        target_path = report_folder / PREDICTIONS_FILE_STEM
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    else:
        target_path = report_folder / f"{PREDICTIONS_FILE_STEM}{source_path.suffix}"
        shutil.copyfile(source_path, target_path)
    return target_path


def build_runs_table(ledger: Ledger) -> str:
    lines = [
        "| Run | Round | Model | Score | Holdout | Status |",
        "|---|---|---|---|---|---|",
    ]
    approved_ids = {run.experiment_id for run in ledger.approved_runs()}
    for run in ledger.runs:
        score_text = format_score(run.score_mean)
        if run.score_std is not None:
            score_text += f" +/- {run.score_std:.4f}"
        if run.crashed:
            status = "crashed"
        elif run.experiment_id in approved_ids:
            status = "approved"
        else:
            status = "rejected or not checked"
        lines.append(
            f"| {run.experiment_id} | {run.focus} | {run.model_family} | {score_text} | "
            f"{format_score(run.holdout_score)} | {status} |"
        )
    return "\n".join(lines)


def build_report_facts(ledger: Ledger, final_result: FinalResult) -> str:
    brief = ledger.brief
    plan = ledger.plan
    best_run = ledger.best_approved_run()
    direction = "higher" if plan.higher_is_better else "lower"
    unverified_ids = [
        evaluation.experiment_id
        for evaluation in ledger.evaluations
        if evaluation.checks.get("unverified")
    ]
    lines = [
        f"Task (the user's words): {ledger.user_task}",
        f"Metrics (the user's words): {ledger.user_metrics_text or 'not given'}",
        f"Data: {brief.summary} Modality {brief.modality}, task {brief.task_family}, "
        f"target {brief.target or 'none'}.",
        f"Validation: {plan.validation_scheme} ({plan.validation_reason}).",
        f"Primary metric: {plan.primary_metric} ({direction} is better). "
        f"Extra metrics: {', '.join(plan.extra_metrics) or 'none'}.",
        f"Experiment rounds finished: {ledger.iteration}. Runs: {len(ledger.runs)}, approved: "
        f"{len(ledger.approved_runs())}. Stop reason: {ledger.stop_reason or 'none'}.",
    ]
    if best_run is not None:
        lines.append(
            f"Best approved run: {best_run.experiment_id}, {best_run.model_family} "
            f"{best_run.model_params}, score {format_score(best_run.score_mean)}, "
            f"extra scores {best_run.extra_scores}."
        )
    else:
        lines.append("There is no approved run, so no predictions were produced.")
    kept_features = [feature.name for feature in ledger.features if feature.kept]
    lines.append(f"Input preparation steps kept: {', '.join(kept_features) or 'none'}.")
    for finding in ledger.eda.findings[:MAX_FINDINGS_IN_REPORT]:
        lines.append(f"EDA finding: {finding.statement}")
    for issue in ledger.eda.issues[:MAX_FINDINGS_IN_REPORT]:
        lines.append(f"Data issue: {issue}")
    lines.append(f"Final predictions: {final_result.predictions_path or 'none'}.")
    lines.append(f"Ensemble: {final_result.ensemble_description or 'not used'}.")
    lines.append("Delivery checks: " + " ".join(final_result.check_messages))
    if unverified_ids:
        lines.append(f"Approved WITHOUT independent checks: {', '.join(unverified_ids)}.")
    if ledger.error_message:
        lines.append(f"The run ended early because: {ledger.error_message}")
    return "\n".join(lines)


def write_report(ledger: Ledger, final_result: FinalResult, agent_context: AgentContext) -> Path:
    facts_text = build_report_facts(ledger, final_result)
    common_values = build_common_prompt_values(AGENT_KEY, ledger, agent_context)
    prompt = render_prompt(
        REPORT_PROMPT, run_rules=common_values["run_rules"], facts_text=facts_text
    )
    try:
        narrative = agent_context.llm_client.ask_text(AGENT_NAME, prompt).strip()
    except LlmUnavailableError:
        narrative = (
            "The language model could not be reached, so this report lists the facts only.\n\n"
            + facts_text
        )

    report_text = f"# mlagent report\n\n{narrative}\n\n## All runs\n\n{build_runs_table(ledger)}\n"
    report_path = agent_context.workspace.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def run_submitter(ledger: Ledger, agent_context: AgentContext) -> FinalResult:
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "packaging the result")
    final_result = FinalResult()

    best_run = ledger.best_approved_run()
    if best_run is None:
        final_result.check_messages = ["No approved run, so no predictions file was written."]
    else:
        final_result.best_experiment_id = best_run.experiment_id
        source_path = Path(best_run.test_predictions_path)
        blend_path, blend_description = try_ensemble(ledger, best_run, agent_context)
        if blend_path is not None:
            source_path = blend_path
            final_result.ensemble_description = blend_description
            event_bus.emit(EventType.DECISION, AGENT_NAME, f"using the {blend_description}")
        else:
            event_bus.emit(
                EventType.INFO, AGENT_NAME, f"using the best single run {best_run.experiment_id}"
            )

        predictions_path = copy_predictions(source_path, agent_context)
        final_result.predictions_path = str(predictions_path)
        problems = check_predictions_file(
            predictions_path, ledger.brief, agent_context.workspace.project_folder
        )
        final_result.check_messages = problems or ["All delivery checks passed."]
        for problem in problems:
            event_bus.emit(EventType.WARNING, AGENT_NAME, problem)

    final_result.report_path = str(write_report(ledger, final_result, agent_context))
    event_bus.emit(
        EventType.AGENT_FINISHED, AGENT_NAME, f"report written to {final_result.report_path}"
    )
    return final_result
