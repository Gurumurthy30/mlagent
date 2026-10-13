"""The Tuner: tunes the best models with a bounded search. It runs only in tune rounds.

For each of the best approved model families, the Coder turns the model's script into a
search with a fixed number of trials and a time cap, following the same script contract.
The Evaluator checks every tuned run like any other run.
"""

import json
from pathlib import Path

from mlagent.agents.base import AgentContext, display_name
from mlagent.agents.coder import run_coder
from mlagent.agents.modeler import fill_run_from_result
from mlagent.digest import describe_brief, describe_run
from mlagent.events import EventType
from mlagent.script_contract import (
    MODEL_RESULT_KEYS_TEXT,
    MODEL_SCRIPT_RULES_TEXT,
    SPLITS_FORMAT_TEXT,
)
from mlagent.settings import (
    RANDOM_SEED,
    TUNE_TIME_FRACTION,
    TUNE_TOP_MODEL_COUNT,
    TUNE_TRIAL_COUNT,
)
from mlagent.state import ExperimentRun, Ledger

AGENT_KEY = "tuner"
AGENT_NAME = display_name(AGENT_KEY)
RESULT_TAG = "RESULT"

TUNE_RESULT_KEYS_TEXT = (
    MODEL_RESULT_KEYS_TEXT
    + """
- "model_params": object, the best parameters the search found
- "trials_run": integer, how many trials finished"""
)


def pick_tune_targets(ledger: Ledger) -> list[ExperimentRun]:
    """The best approved run of each model family, best first, at most the top few."""
    best_run_by_family: dict[str, ExperimentRun] = {}
    for run in ledger.approved_runs():
        family = run.model_family.strip().lower()
        current_best = best_run_by_family.get(family)
        if current_best is None:
            best_run_by_family[family] = run
            continue
        is_better = (
            run.score_mean > current_best.score_mean
            if ledger.plan.higher_is_better
            else run.score_mean < current_best.score_mean
        )
        if is_better:
            best_run_by_family[family] = run

    ranked_runs = sorted(
        best_run_by_family.values(),
        key=lambda run: run.score_mean,
        reverse=ledger.plan.higher_is_better,
    )
    return ranked_runs[:TUNE_TOP_MODEL_COUNT]


def build_tuning_task(
    ledger: Ledger, source_run: ExperimentRun, experiment_folder: Path, agent_context: AgentContext
) -> str:
    workspace = agent_context.workspace
    plan = ledger.plan
    features_path = workspace.features_file(source_run.feature_version)
    search_minutes = int(agent_context.config.limits.code_timeout_minutes * TUNE_TIME_FRACTION)
    search_percent = int(TUNE_TIME_FRACTION * 100)
    direction = "higher" if plan.higher_is_better else "lower"
    instructions = ledger.reviews[-1].instructions if ledger.reviews else []
    instruction_text = "\n".join(f"- {instruction}" for instruction in instructions) or "none"
    return (
        "Tune this model with a bounded search, as ONE script.\n"
        f"Source run: {describe_run(ledger, source_run)}\n"
        f"Source script (read it and follow its pattern): {source_run.script_path}\n"
        f"Current parameters: {json.dumps(source_run.model_params)}\n\n"
        f"Search: at most {TUNE_TRIAL_COUNT} trials with Optuna (or the library's own search), "
        "on the same splits and the same features file. The search must stop by itself after "
        f"{search_percent} percent of the time limit ({search_minutes} minutes), whichever "
        "comes first. Then refit with the best parameters and write the "
        "predictions and the model.\n"
        "Report the best trial: its validation scores per split, holdout score, extra scores, "
        'and its parameters in "model_params".\n\n'
        f"Primary metric: {plan.primary_metric} ({direction} is better)\n"
        f"Extra metrics: {', '.join(plan.extra_metrics) or 'none'}\n"
        f"Seed: {RANDOM_SEED}\n"
        f"Use the GPU: {'yes' if plan.use_gpu else 'no'}\n"
        f"Project folder (read only): {workspace.project_folder}\n"
        f"Experiment folder (save everything here): {experiment_folder}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Features file: {features_path} (add {workspace.features_folder} to sys.path, then "
        f"import {features_path.stem})\n"
        f"Deliverable format of the predictions: {ledger.brief.deliverable_format}\n"
        f"Instructions from the Reviewer:\n{instruction_text}\n\n"
        f"SCRIPT RULES:\n{MODEL_SCRIPT_RULES_TEXT}\n\n"
        f"{SPLITS_FORMAT_TEXT}\n\n"
        f"{describe_brief(ledger)}"
    )


def run_tuner(ledger: Ledger, agent_context: AgentContext) -> list[ExperimentRun]:
    """Tune up to the top few model families. A crash is returned as a run with an error."""
    event_bus = agent_context.event_bus
    targets = pick_tune_targets(ledger)
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, f"tuning {len(targets)} model(s)")

    working_ledger = ledger.model_copy(deep=True)
    tuned_runs: list[ExperimentRun] = []
    for source_run in targets:
        experiment_id = working_ledger.next_experiment_id()
        experiment_folder = agent_context.workspace.experiment_folder(experiment_id)
        run = ExperimentRun(
            experiment_id=experiment_id,
            focus="tune",
            model_family=source_run.model_family,
            model_params=source_run.model_params,
            description=f"tuned {source_run.model_family}, from {source_run.experiment_id}",
            feature_version=source_run.feature_version,
            iteration=ledger.iteration,
            artifact_dir=str(experiment_folder),
        )
        event_bus.emit(EventType.DECISION, AGENT_NAME, f"{experiment_id}: {run.description}")

        coder_result = run_coder(
            task_text=build_tuning_task(ledger, source_run, experiment_folder, agent_context),
            result_tag=RESULT_TAG,
            result_keys_text=TUNE_RESULT_KEYS_TEXT,
            working_folder=experiment_folder,
            script_name="tune_model.py",
            caller_key=AGENT_KEY,
            ledger=ledger,
            agent_context=agent_context,
        )
        run.script_path = str(coder_result.script_path)
        if not coder_result.succeeded:
            run.error = coder_result.error_text or "The Coder could not produce a working script."
        else:
            run.error = fill_run_from_result(run, coder_result.result_data, experiment_folder)
            best_params = coder_result.result_data.get("model_params")
            if not run.error and isinstance(best_params, dict):
                run.model_params = best_params
                run.description = f"tuned {run.model_family} {json.dumps(best_params)}"

        working_ledger.runs.append(run)
        tuned_runs.append(run)
        if run.crashed:
            event_bus.emit(EventType.WARNING, AGENT_NAME, f"{experiment_id} crashed")
        else:
            event_bus.emit(
                EventType.INFO, AGENT_NAME, f"{experiment_id} score {run.score_mean:.4f}"
            )

    event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, f"{len(tuned_runs)} tuned run(s)")
    return tuned_runs
