"""The Submitter: packages the final deliverable mechanically.

Tries an ensemble of the best approved runs (keeping it only if it beats the best single run
by counted gain), copies the final predictions, and runs delivery checks. No LLM calls.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from pydantic import BaseModel, ValidationError

from mlagent.agents.base import AgentContext
from mlagent.agents.coder import run_coder
from mlagent.deliverable_checks import check_predictions_file
from mlagent.events import EventType
from mlagent.scoring import all_finite, is_counted_gain
from mlagent.script_contract import SPLITS_FORMAT_TEXT, VALIDATION_PREDICTIONS_FORMAT_TEXT
from mlagent.settings import ENSEMBLE_MAX_RUNS, PREDICTIONS_FILE_STEM
from mlagent.state import ExperimentRun, FinalResult, Ledger

AGENT_KEY = "submitter"
AGENT_NAME = "Submitter"
RESULT_TAG = "BLEND"

BLEND_RESULT_KEYS = """\
- "validation_scores": list of numbers or null, primary metric of blend on each validation split
- "test_predictions_path": string, path of blended test predictions
- "method": string describing runs combined and blending rule"""


class BlendResult(BaseModel):
    validation_scores: list[float] | None = None
    test_predictions_path: str = ""
    method: str = ""


def pick_ensemble_runs(ledger: Ledger) -> list[ExperimentRun]:
    candidates = [run for run in ledger.approved_runs() if run.validation_predictions_path]
    higher_is_better = ledger.brief.higher_is_better
    ranked = sorted(
        candidates,
        key=lambda r: r.score_mean if r.score_mean is not None else (float("-inf") if higher_is_better else float("inf")),
        reverse=higher_is_better,
    )
    return ranked[:ENSEMBLE_MAX_RUNS]


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


def try_ensemble(
    ledger: Ledger, best_run: ExperimentRun, agent_context: AgentContext
) -> tuple[Path | None, str]:
    """Return blended predictions path and description if ensemble beats best single run."""
    ensemble_runs = pick_ensemble_runs(ledger)
    if len(ensemble_runs) < 2:
        return None, ""

    workspace = agent_context.workspace
    direction = "higher" if ledger.brief.higher_is_better else "lower"
    run_lines = "\n".join(
        f"- {r.experiment_id}: val={r.validation_predictions_path}, test={r.test_predictions_path}"
        for r in ensemble_runs
    )

    task_text = (
        f"Blend the predictions of these approved runs and score on validation splits:\n"
        f"{run_lines}\n\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Primary metric: {ledger.brief.primary_metric} ({direction} is better)\n"
        f"Deliverable format: {ledger.brief.deliverable_format}\n"
        f"Project folder: {workspace.project_folder}\n"
        f"Write blended test predictions inside {workspace.artifacts_folder / AGENT_KEY}.\n\n"
        "Instructions:\n"
        "- Use `from mlagent_tools import load_splits, load_val_preds_with_labels`.\n"
        "- Average probabilities or values across runs.\n"
        "- Return JSON BLEND with validation_scores, test_predictions_path, method.\n\n"
        f"{VALIDATION_PREDICTIONS_FORMAT_TEXT}\n\n"
        f"{SPLITS_FORMAT_TEXT}"
    )

    coder_res = run_coder(
        task_text=task_text,
        result_tag=RESULT_TAG,
        result_keys_text=BLEND_RESULT_KEYS,
        working_folder=workspace.artifacts_folder / AGENT_KEY,
        script_name="blend_runs.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )

    if not coder_res.succeeded or not isinstance(coder_res.result_data, dict):
        return None, ""

    try:
        blend = BlendResult.model_validate(coder_res.result_data)
    except ValidationError:
        return None, ""

    blend_path = Path(blend.test_predictions_path) if blend.test_predictions_path else None
    if (
        blend.validation_scores is None
        or not all_finite(blend.validation_scores)
        or blend_path is None
        or not blend_path.exists()
    ):
        return None, ""

    if not is_counted_gain(
        blend.validation_scores, best_run.validation_scores, ledger.brief.higher_is_better
    ):
        return None, ""

    run_ids = ", ".join(r.experiment_id for r in ensemble_runs)
    return blend_path, f"blend of {run_ids}: {blend.method}"


def run_submitter(ledger: Ledger, agent_context: AgentContext) -> FinalResult:
    """Prepare final predictions and run deliverable checks mechanically."""
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "packaging deliverable")

    final = FinalResult()
    best_run = ledger.best_approved_run()

    if best_run is None:
        final.check_messages = ["No approved run available to package."]
        event_bus.emit(EventType.WARNING, AGENT_NAME, "no approved run")
        return final

    final.best_experiment_id = best_run.experiment_id
    source_pred_path = Path(best_run.test_predictions_path)

    # Attempt ensembling
    blend_path, blend_desc = try_ensemble(ledger, best_run, agent_context)
    if blend_path is not None:
        source_pred_path = blend_path
        final.ensemble_description = blend_desc
        event_bus.emit(EventType.DECISION, AGENT_NAME, f"using {blend_desc}")
    else:
        event_bus.emit(EventType.INFO, AGENT_NAME, f"using best single run {best_run.experiment_id}")

    saved_path = copy_predictions(source_pred_path, agent_context)
    final.predictions_path = str(saved_path)

    problems = check_predictions_file(
        saved_path, ledger.brief, agent_context.workspace.project_folder
    )
    final.check_messages = problems or ["All deliverable checks passed."]
    for p in problems:
        event_bus.emit(EventType.WARNING, AGENT_NAME, f"deliverable check: {p}")

    event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, f"deliverable ready at {saved_path.name}")
    return final
