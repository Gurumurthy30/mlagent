"""The Modeler: trains models.

Round 0 trains the baseline. A features round retrains the best setup on the new features.
A model round tries a new model family or setup. Code checks for duplicate setups and
ensures scripts adhere to the result contract.
"""

from __future__ import annotations

import json

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
You are the Modeler of an autonomous Kaggle Grandmaster ML team. Your purpose is to choose the next \
model family, concrete hyperparameters, and metric-aligned post-processing to maximize cross-validation \
score and build genuine model diversity for final ensembling. A Coder implements and runs your script.

DECISION RULES (in priority order):
1. PORTFOLIO DIVERSITY FIRST (CHECK MODEL FAMILY LEADERBOARD):
   - Round 0 (`focus == start`): Train the fast, reliable baseline from the ML Lead's plan (e.g., default `lightgbm` on tabular; `tfidf_ridge` on text; lightweight pretrained backbone on vision/audio).
   - Subsequent Model Rounds: Consult `MODEL FAMILY LEADERBOARD` and `planned_model_families` in SITUATION. Prioritize UNTRIED model families before making minor hyperparameter tweaks to an already-tried family!
   - Modality-matched families:
     * Tabular: Explore the diverse GBDT trio (`lightgbm` leaf-wise -> `catboost` oblivious trees -> `xgboost` depth-wise), then add a non-tree family (`ridge` / `logistic_regression` / `mlp` / `extra_trees` / `random_forest`) so Submitter can blend uncorrelated predictions.
     * Text / NLP: `tfidf_ridge` / `tfidf_sgd`, `tfidf_svd_lightgbm`, `pretrained_transformer` (e.g. DeBERTa/RoBERTa).
     * Vision / Audio: Pretrained `timm` CNN/ViT backbones (`resnet34`, `convnext_tiny`, `efficientnet_b0`) or GBDT/Ridge on pooled frozen embeddings + metadata.
     * Time Series: `lightgbm`, `catboost`, `xgboost` with lag/rolling features, and `ridge` trend models.
2. METRIC-ALIGNED LOSS & POST-PROCESSING (`post_processing`):
   - Always align the training objective and `post_processing` with `primary_metric` and the ML Lead's `metric_post_processing`:
     * Threshold metrics (`f1`, `f1_macro`, `accuracy`, `balanced_accuracy`, `mcc`, `qwk`): Train with probability output (`binary_logloss` / `CrossEntropy`) and tune the classification threshold(s) on out-of-fold (OOF) probabilities (`scipy.optimize`) before producing discrete labels.
     * Ranking / probability metrics (`roc_auc`, `pr_auc`): Output raw probabilities (`predict_proba`) in both `val_predictions.csv` and `test_predictions.csv`—NEVER threshold to 0/1!
     * LogLoss / Brier (`log_loss`): Clip probabilities to `[1e-6, 1 - 1e-6]` and check OOF calibration.
     * Skewed regression / RMSLE (`rmsle`, `rmse`, `mae`): If target is non-negative right-skewed or metric is `rmsle`, train on `np.log1p(y)` and invert predictions with `np.expm1(pred)` clipped to valid target bounds.
3. HARD DUPLICATE AVOIDANCE & SINGLE-CHANGE CREDIT ASSIGNMENT:
   - Never repeat a `(model_family, model_params, feature_version)` setup already listed in RUNS.
   - Avoid banned or crashed configurations listed in DEAD ENDS.
4. WELL-REGULARIZED HYPERPARAMETERS & BOUNDED RUNTIME:
   - For GBDTs: use `n_estimators` (800 to 2500) paired with `early_stopping_rounds=50`, `learning_rate` (0.02 to 0.08), `subsample`/`colsample_bytree` (0.7 to 0.9), and sensible depth/leaves so training finishes quickly without overfitting.

WORKED EXAMPLE (Tabular Classification):
- hypothesis: "CatBoost symmetric (oblivious) trees with depth 6 and OOF threshold optimization will capture categorical interactions differently than leaf-wise LightGBM and improve both single-model CV and ensemble diversity."
- model_family: "catboost"
- model_params: {{"iterations": 1500, "learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 3.0, "eval_metric": "Logloss"}}
- post_processing: "Optimize classification probability threshold on out-of-fold predictions (grid/Golden-section search 0.30 to 0.70) to maximize primary_metric; average raw test probabilities across CV folds before applying the optimal threshold."
- reason: "Train CatBoost with depth=6 and OOF threshold tuning to add oblivious-tree diversity alongside LightGBM."

SITUATION:
{digest}

THIS ROUND: {round_goal}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 3-5 sentences analyzing the MODEL FAMILY LEADERBOARD, recent run bottlenecks, untried model families, and how this model + post-processing fits the primary metric.
- hypothesis: 1-2 sentences stating the falsifiable hypothesis this model setup tests.
- model_family: snake_case identifier (e.g., "lightgbm", "catboost", "xgboost", "ridge", "logistic_regression", "mlp", "random_forest", "extra_trees").
- model_params: flat object of concrete hyperparameter values.
- post_processing: exact post-processing rule to apply to OOF and test predictions (threshold tuning, probability clipping, log1p/expm1 target transform, or none).
- reason: one clear sentence stating what this run tests and why (becomes the run summary)."""


class ModelChoice(BaseModel):
    thinking: str = ""
    hypothesis: str = ""
    model_family: str
    model_params: dict = {}
    post_processing: str = ""
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
                hypothesis=(
                    f"Retraining best model family ({best_run.model_family}) on upgraded "
                    f"features v{ledger.current_feature_version} will improve cross-validation score."
                ),
                model_family=best_run.model_family,
                model_params=best_run.model_params,
                post_processing=best_run.post_processing or ledger.plan.metric_post_processing,
                reason=f"retrain best setup ({best_run.model_family}) on features v{ledger.current_feature_version}",
            )

    if choice is None:
        round_goal = (
            f"Train the Round 0 baseline from the plan: {ledger.plan.baseline_description} "
            f"(metric post-processing: {ledger.plan.metric_post_processing or 'standard'})"
            if ledger.focus == "start"
            else (
                "Choose an UNTRIED model family from the plan (or a structurally distinct architecture) "
                "with metric-aligned post-processing to beat the best run and increase ensemble diversity."
            )
        )
        digest = build_digest(AGENT_KEY, ledger)
        candidate: ModelChoice | None = None
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

    if not choice.post_processing and ledger.plan.metric_post_processing:
        choice.post_processing = ledger.plan.metric_post_processing

    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, f"{experiment_id}: {choice.reason}")
    event_bus.emit(
        EventType.STATUS,
        AGENT_NAME,
        f"training {choice.model_family} on features v{ledger.current_feature_version}",
        {"reason": choice.reason},
    )

    # 2. Ask Coder to train and score
    workspace = agent_context.workspace
    features_path = workspace.features_file(ledger.current_feature_version)
    features_source = ""
    if features_path.is_file():
        features_source = features_path.read_text(encoding="utf-8")[:MAX_FEATURE_SOURCE_CHARS]

    direction = "higher" if ledger.brief.higher_is_better else "lower"
    instructions = ledger.reviews[-1].instructions if ledger.reviews else []
    instructions_text = "\n".join(f"- {i}" for i in instructions) or "none"
    post_proc_text = f"Post-processing strategy: {choice.post_processing}\n" if choice.post_processing else ""

    task_text = (
        f"Train and score this model with ONE complete, leak-free Python script.\n"
        f"Model family: {choice.model_family}\n"
        f"Parameters: {json.dumps(choice.model_params)}\n"
        f"Reason: {choice.reason}\n"
        f"Hypothesis: {choice.hypothesis}\n"
        f"{post_proc_text}"
        f"Primary metric: {ledger.brief.primary_metric} ({direction} is better)\n"
        f"Extra metrics: {ledger.brief.extra_metrics}\n"
        f"Target column: {ledger.brief.target or 'none'}, ID column: {ledger.brief.id_column or 'none'}\n"
        f"Seed: {RANDOM_SEED}\n"
        f"Use GPU: {'yes' if ledger.plan.use_gpu else 'no'}\n"
        f"Project folder (read-only): {workspace.project_folder}\n"
        f"Experiment folder (outputs): {experiment_folder}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Features file: {features_path} (import {features_path.stem} after adding {workspace.features_folder} to sys.path)\n"
        f"Deliverable format: {ledger.brief.deliverable_format}\n"
        f"Reviewer instructions:\n{instructions_text}\n\n"
        "TRAINING & PREDICTION PROTOCOL:\n"
        "1. FOLD-ISOLATED FEATURE PREPARATION: For each CV split `k` (`0 .. n_splits - 1`), call `state = fit_preparation(train_fold_df, seed=Seed)` "
        "strictly on `train_fold_df`, then transform `train_fold_df`, `val_fold_df`, `holdout_df` (if `fold == -1` exists), and `test_df` using `apply_preparation(df, state)`.\n"
        "2. FOLD-AVERAGED TEST PREDICTIONS: Accumulate raw test predictions (or predicted probabilities for classification) across all `n_splits` fold models (`test_preds_avg = sum(fold_test_preds) / n_splits`).\n"
        "3. METRIC POST-PROCESSING: If `Post-processing strategy` specifies out-of-fold threshold tuning (for F1/accuracy/MCC/QWK), probability clipping, or target `log1p`/`expm1`, "
        "fit the threshold/transform on the out-of-fold validation predictions (`fold >= 0`), apply it to `val_predictions.csv` (`prediction` column), holdout (`fold == -1`), and `test_predictions.csv`.\n"
        "4. SAVE REQUIRED FILES IN EXPERIMENT FOLDER:\n"
        "   - `val_predictions.csv`: must include `example_id`, `fold` (for all CV folds `0..n_splits-1` AND holdout `fold == -1`), `prediction` (formatted for the primary metric), AND `proba_<class>` columns for classification.\n"
        "   - `test_predictions.csv`: must strictly match `Deliverable format` (exact column names, exact test row count, ID order, and value type: probabilities vs discrete class labels/booleans).\n"
        "   - Save the trained model artifact as `model.<ext>`.\n"
        "5. Return JSON RESULT with validation_scores, holdout_score, test_predictions_path, model_path, etc.\n\n"
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
        hypothesis=choice.hypothesis,
        post_processing=choice.post_processing,
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
        score_str = f"{run.score_mean:.4f}" if run.score_mean is not None else "n/a"
        event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, f"{experiment_id} trained, score {score_str}")
        event_bus.emit(
            EventType.RESULT,
            AGENT_NAME,
            f"{experiment_id} ({choice.model_family}): score {score_str} on features v{ledger.current_feature_version}",
        )

    return run