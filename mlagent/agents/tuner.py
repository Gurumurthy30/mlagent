"""The Tuner: tunes the best models with a bounded search without LLM calls.

For each of the best approved model families, the Coder runs a bounded parameter search
(e.g., using Optuna or random search) with a time cap, adhering to the script contract.
"""

from __future__ import annotations

import json

from mlagent.agents.base import AgentContext
from mlagent.agents.coder import run_coder
from mlagent.digest import shorten_text
from mlagent.events import EventType
from mlagent.scoring import fill_run_from_result
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
AGENT_NAME = "Tuner"
RESULT_TAG = "RESULT"

TUNE_RESULT_KEYS = (
    MODEL_RESULT_KEYS_TEXT
    + """
- "model_params": flat object, the parameters of the best trial
- "trials_run": integer, how many trials finished"""
)


def pick_tune_targets(ledger: Ledger) -> list[ExperimentRun]:
    """Pick the best approved run from each model family."""
    best_by_family: dict[str, ExperimentRun] = {}
    higher_is_better = ledger.brief.higher_is_better

    for run in ledger.approved_runs():
        family = run.model_family.strip().lower()
        curr = best_by_family.get(family)
        if curr is None:
            best_by_family[family] = run
            continue
        if run.score_mean is not None and curr.score_mean is not None:
            is_better = run.score_mean > curr.score_mean if higher_is_better else run.score_mean < curr.score_mean
            if is_better:
                best_by_family[family] = run

    ranked = sorted(
        best_by_family.values(),
        key=lambda r: r.score_mean if r.score_mean is not None else (float("-inf") if higher_is_better else float("inf")),
        reverse=higher_is_better,
    )
    return ranked[:TUNE_TOP_MODEL_COUNT]


def run_tuner(ledger: Ledger, agent_context: AgentContext) -> list[ExperimentRun]:
    """Tune top approved model families."""
    event_bus = agent_context.event_bus
    targets = pick_tune_targets(ledger)
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, f"tuning {len(targets)} model(s)")

    tuned_runs: list[ExperimentRun] = []
    workspace = agent_context.workspace
    higher_is_better = ledger.brief.higher_is_better
    direction = "higher" if higher_is_better else "lower"
    search_minutes = max(int(agent_context.config.limits.code_timeout_minutes * TUNE_TIME_FRACTION), 1)

    for source_run in targets:
        experiment_id = ledger.next_experiment_id()
        experiment_folder = workspace.experiment_folder(experiment_id)
        features_path = workspace.features_file(source_run.feature_version)

        instructions = ledger.reviews[-1].instructions if ledger.reviews else []
        instructions_text = "\n".join(f"- {i}" for i in instructions) or "none"
        post_proc = source_run.post_processing or ledger.plan.metric_post_processing

        task_text = (
            f"Tune model {source_run.model_family} using a bounded hyperparameter search (Optuna or structured search) in ONE script.\n"
            f"Source script: {source_run.script_path}\n"
            f"Initial parameters (Trial 0 baseline): {json.dumps(source_run.model_params)}\n"
            f"Parent CV score to beat: {source_run.score_mean}\n"
            f"Post-processing to preserve: {post_proc or 'none'}\n"
            f"Search up to {TUNE_TRIAL_COUNT} trials or {search_minutes} minutes maximum.\n"
            f"Primary metric: {ledger.brief.primary_metric} ({direction} is better)\n"
            f"Target column: {ledger.brief.target or 'none'}, ID column: {ledger.brief.id_column or 'none'}\n"
            f"Deliverable format: {ledger.brief.deliverable_format}\n"
            f"Random seed: {RANDOM_SEED}\n"
            f"Use GPU: {'yes' if ledger.plan.use_gpu else 'no'}\n"
            f"Project folder (read-only): {workspace.project_folder}\n"
            f"Experiment folder (outputs): {experiment_folder}\n"
            f"Splits file: {workspace.splits_path}\n"
            f"Features file: {features_path} (import {features_path.stem} after adding {workspace.features_folder} to sys.path)\n"
            f"Reviewer instructions:\n{instructions_text}\n\n"
            "GRANDMASTER TUNING PROTOCOL (in priority order):\n"
            "1. PRE-CACHE PREPARED FOLDS ONCE: Before starting the trial loop, run `fit_preparation(train_fold)` and `apply_preparation` "
            "for each CV fold (`fold >= 0`) ONCE and store the prepared fold matrices in a list in memory so trials run 10x faster.\n"
            "2. TRIAL 0 WARM-START GUARANTEE: Trial 0 MUST evaluate the exact `Initial parameters` without changes (e.g. `study.enqueue_trial(initial_params)`). "
            "If no later trial beats Trial 0, keep Trial 0's parameters and predictions so tuning NEVER regresses below the parent run.\n"
            "3. HIGH-LEVERAGE SEARCH SPACE:\n"
            "   - For GBDTs (LightGBM / XGBoost / CatBoost): tune regularization and tree capacity (`colsample_bytree`/`feature_fraction` in [0.5, 1.0], "
            "`subsample`/`bagging_fraction` in [0.5, 1.0], `min_child_samples`/`min_data_in_leaf` in [5, 100], `max_depth` in [3, 10] or `num_leaves` in [15, 127], "
            "`reg_alpha` in [1e-3, 10.0] log-scale, `reg_lambda` in [1e-3, 10.0] log-scale, `learning_rate` in [0.015, 0.12]) with early stopping.\n"
            "   - For Linear / Ridge / Logistic / MLP: tune regularization (`alpha` or `C` log-scale [1e-4, 1e3], `l1_ratio`, hidden dimensions, `dropout`, `weight_decay`).\n"
            "4. STRICT HOLDOUT ISOLATION: Tune strictly on CV folds (`fold >= 0`). NEVER evaluate or select trials using `fold == -1` (holdout).\n"
            "5. FINAL PREDICTIONS & POST-PROCESSING: Using the best trial parameters, compute out-of-fold validation predictions (`fold >= 0`), "
            "score holdout (`fold == -1`) once if present, average test set predictions across all CV fold models, apply the `Post-processing to preserve` (e.g. OOF threshold tuning), "
            "and write `val_predictions.csv` (including `proba_<class>` for classification) and `test_predictions.csv` (matching `Deliverable format`).\n"
            "6. RESULT LINE: Output JSON RESULT with best trial `validation_scores`, `holdout_score`, `test_predictions_path`, `model_path`, `model_params`, and `trials_run`.\n\n"
            f"SCRIPT RULES:\n{MODEL_SCRIPT_RULES_TEXT}\n\n"
            f"{SPLITS_FORMAT_TEXT}"
        )

        event_bus.emit(
            EventType.STATUS,
            AGENT_NAME,
            f"tuning {source_run.model_family} parameters",
            {"reason": f"optimizing hyperparams of {source_run.experiment_id} across up to {TUNE_TRIAL_COUNT} trials"},
        )

        run = ExperimentRun(
            experiment_id=experiment_id,
            focus="tune",
            model_family=source_run.model_family,
            model_params=source_run.model_params,
            summary=f"tuned {source_run.model_family} from {source_run.experiment_id}",
            hypothesis=f"Tuning regularization and capacity hyperparameters of {source_run.model_family} ({source_run.experiment_id}) will improve CV score.",
            post_processing=post_proc,
            feature_version=source_run.feature_version,
            iteration=ledger.iteration,
            artifact_dir=str(experiment_folder),
        )

        coder_res = run_coder(
            task_text=task_text,
            result_tag=RESULT_TAG,
            result_keys_text=TUNE_RESULT_KEYS,
            working_folder=experiment_folder,
            script_name="tune_model.py",
            caller_key=AGENT_KEY,
            ledger=ledger,
            agent_context=agent_context,
        )

        run.script_path = str(coder_res.script_path)
        if not coder_res.succeeded:
            run.error = coder_res.error_text or "Tuning script failed to run."
        else:
            run.error = fill_run_from_result(run, coder_res.result_data, experiment_folder)
            best_params = coder_res.result_data.get("model_params") if isinstance(coder_res.result_data, dict) else None
            if not run.error and isinstance(best_params, dict):
                run.model_params = best_params
                param_str = shorten_text(json.dumps(best_params), 80)
                run.summary = f"tuned {run.model_family}: best params {param_str}"

        tuned_runs.append(run)
        if run.crashed:
            event_bus.emit(EventType.WARNING, AGENT_NAME, f"{experiment_id} crashed: {run.error[:80]}")
        else:
            score_str = f"{run.score_mean:.4f}" if run.score_mean is not None else "n/a"
            event_bus.emit(EventType.INFO, AGENT_NAME, f"{experiment_id} score {score_str}")
            event_bus.emit(
                EventType.RESULT,
                AGENT_NAME,
                f"{experiment_id} (tuned {source_run.model_family}): score {score_str}",
            )

    event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, f"{len(tuned_runs)} model(s) tuned")
    return tuned_runs