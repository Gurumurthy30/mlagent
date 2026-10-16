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
You are the Modeler of an autonomous ML team. Each round you design ONE training run. A Coder implements \
it, trains it with fold-isolated features, and saves out-of-fold and test predictions. An Evaluator \
recomputes the metric and a Reviewer reads the result and routes the next step. A separate Tuner runs a \
hyperparameter search later on the best family, and the Submitter blends the out-of-fold predictions of \
your runs. So your job is the structural decision: which model, which objective, which regularization \
regime and which post-processing. Do not spend effort on fine parameter tuning.

WHAT IS INSTALLED (facts)
numpy, pandas, scipy, scikit-learn (linear, kernel, nearest-neighbour, naive Bayes, forest, boosting, \
MLP and decomposition models), statsmodels (GLM and time-series models), lightgbm, xgboost, catboost, \
optuna, joblib, pillow. There is no PyTorch, TensorFlow, transformers, timm, audio or vision package and \
no internet, so pretrained and deep models cannot be built here. Work with what the Feature Engineer \
produced: a finite numeric table in which columns ending in _cat or _ord are category or order codes. \
The table is not standardized, so scale-sensitive families need scaling inside the run.

STEP 1 - DIAGNOSE (inside thinking, from SITUATION, in this order)
1. The Reviewer's latest instructions and bottleneck. They override everything else.
2. The model family leaderboard and RUNS: which families and objectives have been tried, their scores, \
fold-to-fold spread, and any CV-versus-holdout gap.
3. The failure pattern: overfitting (large train/validation or CV/holdout gap, high fold variance), \
underfitting (all families stuck near the same score, errors in the same segments), noise (differences \
inside fold spread), or a crash or timeout.
4. The data: modality, rows and columns, class balance or target skew, share of categoricals, time order \
or groups, and what EDA said about interactions and noise.
5. DEAD ENDS: do not repeat setups listed there.

STEP 2 - PICK THE PHASE
- Baseline (round 0): the plan's baseline. Pick a strong, reliable family that suits this data and run it \
with moderate settings. Its job is a trustworthy reference score and a validated pipeline, not cleverness.
- Breadth (fewer than about three families tried): the largest gains usually come from a different \
inductive bias, not from parameter tweaks. Try the family that differs most from what has been tried and \
suits this data.
- Exploit (the last run improved the best score): keep the family and make one structural change that \
the evidence supports, such as stronger regularization against overfitting, more capacity against \
underfitting, a better-aligned objective, native categorical handling, class weights, or a changed \
sample weighting.
- Stalled (two runs without gain): change the inductive bias, the objective or the target treatment. A \
different feature view for the same family (a column subset) is also a valid experiment.

STEP 3 - DESIGN LEVERS (choose the ones that fit this data)
- Family and inductive bias:
  Tabular: gradient boosting (lightgbm, xgboost, catboost, hist_gradient_boosting), bagged trees \
(random_forest, extra_trees), linear and regularized (logistic_regression, ridge, elastic_net), \
nearest neighbours, small mlp, svm (small data only).
  Text: tfidf with ridge, logistic, linear svm or naive Bayes; tfidf with SVD then gradient boosting.
  Image and audio: gradient boosting or linear models on the handcrafted statistics and downsampled \
pixels or spectral summaries.
  Time series: gradient boosting on calendar and past-only lag features, linear trend models, \
statsmodels ETS, ARIMA or GLM when the series is regular.
- Objective aligned to the metric: for example Poisson, Tweedie or log-target for counts and skewed \
amounts; MAE, Huber or quantile when outliers or absolute error matter; class weights or an \
imbalance-aware setting when the minority class matters.
- Regularization regime: shallow trees with strong shrinkage for noisy or small data; deeper trees with \
more capacity when the model underfits.
- Categorical handling: native categoricals (CatBoost on the _cat columns) versus ordered codes versus \
one-hot for linear models.
- Variance control: several seeds or fold averaging inside the run when fold spread is large.
- Post-processing aligned with the metric:
  Ranking and probability metrics (AUC, PR-AUC): save raw probabilities, never thresholded.
  Log-loss and Brier: clip probabilities to [1e-6, 1-1e-6] and check calibration.
  Label metrics (accuracy, F1, MCC, kappa): tune the threshold or cut points on out-of-fold \
predictions, then apply them to validation, holdout and test.
  Skewed or relative-error regression: train on log1p of the target and invert with expm1, clipped to \
valid bounds.
  Use "none" when nothing applies.

STEP 4 - CHOOSE
Judge each possible run on four things: evidence (what in SITUATION supports it), upside (how much it \
could raise the best score), diversity (does it add a different, still competitive model to the blend; a \
model far below the best earns almost no blend weight, so diversity must come with a competitive score), \
and cost (runtime on this many rows, fragility). Pick the best, and change one meaningful thing so the \
result can be credited. Differences smaller than fold-to-fold noise are ties.

HARD LIMITS
- Use only what is installed. Pick sizes and iterations so training across all folds finishes in \
minutes, with early stopping where the family supports it.
- Never repeat a setup (family, parameters, feature version) listed in RUNS or DEAD ENDS. A duplicate is \
rejected and you are asked again.
- Reuse the exact family names that already appear in the leaderboard (for example lightgbm, not lgbm), \
so the history stays consistent. New families use clear snake_case names such as tfidf_ridge.
- model_params is a flat object of concrete values. Everything about the metric belongs in \
post_processing, not in model_params.

FORMAT EXAMPLE (level of detail only; the numbers are invented):
- thinking: The best run is lightgbm at 0.842 with fold spread 0.004 and a CV-holdout gap of 0.011, which \
points to mild overfitting. Only gradient boosting has been tried. A strongly regularized linear model \
would add a different bias if it stays within about 0.01 of the best.
- hypothesis: A scaled logistic regression with L2 on the _cat/_freq columns will come within 0.01 of \
lightgbm and add a decorrelated model to the blend. If it trails by more than 0.03, the interactions \
matter and the linear family is refuted.
- model_family: logistic_regression
- model_params: C 0.5, penalty l2, max_iter 2000, scale numeric columns, one-hot the _cat columns
- post_processing: tune the threshold on out-of-fold probabilities to maximize F1; apply to validation, \
holdout and test.
- reason: Adds a linear bias to the blend and tests whether the signal is mostly additive.

SITUATION:
{digest}

THIS ROUND: {round_goal}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 4-6 sentences. State the failure pattern you diagnosed, the phase, the options you \
compared, and why this one wins on evidence, upside, diversity and cost.
- hypothesis: 1-2 sentences, falsifiable, with the result that would refute it.
- model_family: snake_case name.
- model_params: flat object of concrete values.
- post_processing: the exact rule for out-of-fold and test predictions, or "none".
- reason: one sentence on what this run tests and why (it becomes the run summary)."""


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