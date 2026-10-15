"""The Modeler: trains models.

Round 0 trains the baseline. A features round retrains the best setup on the new features.
A model round tries a new model family or setup. Code checks for duplicate setups and
ensures scripts adhere to the result contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from mlagent.agents.base import AgentContext
from mlagent.agents.coder import run_coder
from mlagent.digest import build_digest
from mlagent.events import EventType
from mlagent.scoring import fill_run_from_result, make_setup_signature
from mlagent.script_contract import (
    MODEL_RESULT_KEYS_TEXT,
    MODEL_SCRIPT_RULES_TEXT,
    SPLITS_FORMAT_TEXT,
)
from mlagent.settings import MAX_FEATURE_SOURCE_CHARS, RANDOM_SEED
from mlagent.state import ExperimentRun, Ledger

AGENT_KEY = "modeler"
AGENT_NAME = "Modeler"
RESULT_TAG = "RESULT"

CHOOSE_PROMPT = """\
You are the Modeler of an autonomous ML team. You decide what model and parameters to train next.
A Coder writes and executes the script.

GOAL:
Select the model setup with the highest probability of genuine score improvement for its compute cost.

SITUATION:
{digest}

THIS ROUND: {round_goal}

HOW TO THINK:
- A setup is defined by family + parameters + features version. The same family and parameters are only allowed when the features version is new. Never repeat a setup.
- Round 0 is the baseline: train a fast, reliable, hard-to-break model.
- Prefer architectures that match the data modality (tree ensembles for tabular, embeddings/nets for text/image/audio).
- One clear change at a time so the team understands what caused the score difference.

YOU MAY:
- Select faster, lighter models if training time is tight.
- Refuse to add hyperparameter tweaks when a completely different family is needed.

OUTPUT:
Return JSON:
- thinking: 3-5 sentences analyzing candidate families and expected gains.
- model_family: snake_case identifier (e.g., lightgbm, random_forest, ridge).
- model_params: flat dict of hyperparameters.
- reason: one plain sentence stating what this run tests and why (becomes the run summary)."""


class ModelChoice(BaseModel):
    thinking: str = ""
    model_family: str
    model_params: dict = {}
    reason: str = ""


def run_modeler(ledger: Ledger, agent_context: AgentContext) -> ExperimentRun:
    """Train one model. Crashes are returned as runs with errors, never raised."""
    event_bus = agent_context.event_bus
    experiment_id = ledger.next_experiment_id()
    experiment_folder = agent_context.workspace.experiment_folder(experiment_id)

    # 1. Choose setup (inline duplicate check)
    tried_signatures = {
        make_setup_signature(r.model_family, r.model_params, r.feature_version)
        for r in ledger.runs
        if not r.crashed
    }

    best_run = ledger.best_approved_run()
    choice: ModelChoice | None = None

    if ledger.focus == "features" and best_run is not None:
        candidate_sig = make_setup_signature(
            best_run.model_family, best_run.model_params, ledger.current_feature_version
        )
        if candidate_sig not in tried_signatures:
            choice = ModelChoice(
                model_family=best_run.model_family,
                model_params=best_run.model_params,
                reason=f"retrain best setup ({best_run.model_family}) on features v{ledger.current_feature_version}",
            )

    if choice is None:
        round_goal = (
            f"Train the baseline from the plan: {ledger.plan.baseline_description}"
            if ledger.focus == "start"
            else "Choose a NEW model family or distinct setup likely to beat the best run."
        )
        digest = build_digest(AGENT_KEY, ledger)
        for _ in range(3):
            prompt = CHOOSE_PROMPT.format(digest=digest, round_goal=round_goal)
            candidate = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, ModelChoice)
            sig = make_setup_signature(
                candidate.model_family, candidate.model_params, ledger.current_feature_version
            )
            if sig not in tried_signatures:
                choice = candidate
                break
        if choice is None:
            # Fallback choice if duplicates kept recurring
            choice = candidate or ModelChoice(
                model_family="baseline_model",
                model_params={},
                reason="fallback model after duplicate choices",
            )

    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, f"{experiment_id}: {choice.reason}")

    # 2. Ask Coder to train and score
    workspace = agent_context.workspace
    features_path = workspace.features_file(ledger.current_feature_version)
    features_source = ""
    if features_path.is_file():
        features_source = features_path.read_text(encoding="utf-8")[:MAX_FEATURE_SOURCE_CHARS]

    direction = "higher" if ledger.brief.higher_is_better else "lower"
    instructions = ledger.reviews[-1].instructions if ledger.reviews else []
    instructions_text = "\n".join(f"- {i}" for i in instructions) or "none"

    task_text = (
        f"Train and score this model with ONE script.\n"
        f"Model: {choice.model_family}\n"
        f"Parameters: {json.dumps(choice.model_params)}\n"
        f"Reason: {choice.reason}\n\n"
        f"Primary metric: {ledger.brief.primary_metric} ({direction} is better)\n"
        f"Seed: {RANDOM_SEED}\n"
        f"Use GPU: {'yes' if ledger.plan.use_gpu else 'no'}\n"
        f"Project folder (read-only): {workspace.project_folder}\n"
        f"Experiment folder (outputs): {experiment_folder}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Features file: {features_path} (import {features_path.stem} after adding {workspace.features_folder} to sys.path)\n"
        f"Deliverable format: {ledger.brief.deliverable_format}\n"
        f"Reviewer instructions:\n{instructions_text}\n\n"
        "Requirements:\n"
        "- Save test_predictions.csv and val_predictions.csv inside the experiment folder.\n"
        "- Return JSON RESULT with validation_scores, test_predictions_path, model_path, etc.\n\n"
        f"SCRIPT RULES:\n{MODEL_SCRIPT_RULES_TEXT}\n\n"
        f"{SPLITS_FORMAT_TEXT}\n\n"
        f"Features source excerpt:\n{features_source}"
    )

    coder_res = run_coder(
        task_text=task_text,
        result_tag=RESULT_TAG,
        result_keys_text=MODEL_RESULT_KEYS_TEXT,
        working_folder=experiment_folder,
        script_name="train_model.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )

    # 3. Assemble ExperimentRun
    run = ExperimentRun(
        experiment_id=experiment_id,
        focus=ledger.focus,
        model_family=choice.model_family,
        model_params=choice.model_params,
        summary=choice.reason,
        feature_version=ledger.current_feature_version,
        script_path=str(coder_res.script_path),
        artifact_dir=str(experiment_folder),
        iteration=ledger.iteration,
    )

    if not coder_res.succeeded:
        run.error = coder_res.error_text or "Training script failed."
    else:
        run.error = fill_run_from_result(run, coder_res.result_data, experiment_folder)

    if run.crashed:
        event_bus.emit(EventType.WARNING, AGENT_NAME, f"{experiment_id} crashed: {run.error[:80]}")
    else:
        event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, f"{experiment_id} trained, score {run.score_mean:.4f}")

    return run
