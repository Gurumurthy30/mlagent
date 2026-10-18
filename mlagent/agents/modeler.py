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
You are the Modeler of an autonomous ML team. Each round you design ONE training run for any kind of \
task: tables, text, images, audio, time series, relational or mixed. A Coder implements it and trains it \
with fold-isolated features, saving out-of-fold and test predictions. An Evaluator recomputes the metric and \
a Reviewer reads the result and routes the next step. A separate Tuner searches hyperparameters later on \
the best family, and the Submitter blends the out-of-fold predictions of your runs. So your job is the \
structural decision: which model, which objective, which training regime, which post-processing. Leave \
fine parameter tuning to the Tuner.

WHAT YOU CAN BUILD WITH (library knowledge)
Always installed: numpy, pandas, scipy, scikit-learn, statsmodels, lightgbm, xgboost, catboost, optuna, \
joblib, pillow, spacy, nltk. Only when the ENV finding in SITUATION confirms them: torch, torchvision, \
torchaudio, transformers, sentence-transformers, timm, librosa, soundfile, opencv. Read the same finding \
for GPU and its memory, CPU count, RAM and internet. With no ENV finding, assume CPU and the core packages \
only. With internet, pretrained weights download once from the library's hub and are reused from cache.

Model catalog by library:
- lightgbm, xgboost, catboost: the strongest default for rows and columns. Objectives include binary, \
multiclass, regression with L2, L1, huber, quantile, poisson, tweedie and gamma, and ranking (lambdarank, \
rank:pairwise). CatBoost handles categoricals natively and also text and embedding features. GPU: xgboost \
device cuda, catboost task_type GPU. Levers: depth and leaves, learning rate with early stopping, \
regularization, class weights, subsampling.
- scikit-learn: LogisticRegression, Ridge, Lasso, ElasticNet, SGD models and LinearSVC (fast and strong on \
sparse text); SVC for small data; RandomForest and ExtraTrees (decorrelated from boosting); \
HistGradientBoosting (native categoricals, poisson and quantile losses); KNeighbors on embeddings; \
MultinomialNB and ComplementNB on text counts; MLP; CalibratedClassifierCV. These need scaling inside the \
run where scale matters, since the feature table is not standardized.
- statsmodels: GLM (poisson, gamma, tweedie), SARIMAX, ETSModel and ExponentialSmoothing, UnobservedComponents, \
MixedLM. For regular, short or sparse series where a classical model is a strong baseline.
- torch (custom): a tabular MLP with categorical embeddings, a 1D CNN or GRU or transformer for sequences \
and series, a small CNN on spectrograms. Needs a training loop: DataLoader, AdamW, a scheduler, \
mixed precision on GPU.
- torchvision and timm: pretrained image backbones (resnet, efficientnet, convnext, swin, vit) via \
timm.create_model(name, pretrained=True, num_classes=n); torchvision segmentation (deeplabv3, fcn, lraspp) \
and detection (faster rcnn, retinanet) models. Fine-tune with augmentation, or freeze and use as a feature \
extractor.
- transformers and sentence-transformers: AutoModelForSequenceClassification (deberta-v3, roberta, bert, \
xlm-roberta for mixed languages, distilbert and MiniLM for speed), token classification, question \
answering, seq2seq (t5). Regression via num_labels=1. Frozen sentence embeddings need no fine-tuning.
- audio: log-mel spectrograms (torchaudio or librosa) into a timm CNN; pretrained audio encoders through \
transformers (wav2vec2, AST); or summary statistics into gradient boosting.
- frozen pipelines (cheap, strong on CPU and GPU): extract embeddings once (cached), then train \
logistic, ridge, kNN, SVM or gradient boosting on them. Often within a few points of fine-tuning at a \
small fraction of the cost.

How a run works here: the script reads raw inputs (documents, images, audio, series) from the project \
folder, and may concatenate the feature table as side inputs; folds come from the splits file; each fold \
saves out-of-fold and test predictions in the required format. Deep runs must be sized to finish all folds \
within the time budget: pick a small backbone, a short fixed epoch schedule, mixed precision, dynamic \
padding, an image size or token length justified by EDA, and free GPU memory between folds. Estimate \
cost as samples x epochs divided by throughput before choosing. Without a GPU, fine-tuning beyond tiny data \
is impractical; use frozen embeddings with a simple head, or classical models.

STEP 1 - DIAGNOSE (inside thinking, from SITUATION, in this order)
1. The Reviewer's latest instructions and bottleneck. They override everything else.
2. The model family leaderboard and RUNS: families and objectives tried, scores, fold-to-fold spread, any \
CV-versus-holdout gap.
3. The failure pattern: overfitting (a large train/validation or CV/holdout gap, high fold variance), \
underfitting (all families stuck near the same score, errors in the same segments), noise (differences \
inside fold spread), or a crash or timeout.
4. The data: modality, rows, class balance or target skew, categoricals, time order or groups, and what \
EDA said about interactions, noise, duplicates and sequence lengths.
5. Compute and DEAD ENDS: the hardware, the time budget and the setups already ruled out.

STEP 2 - PICK THE PHASE
- Baseline (round 0): the plan's baseline. A strong, reliable family at moderate settings. Its job is a \
trustworthy reference score and a validated pipeline.
- Breadth (fewer than about three families tried): the largest gains usually come from a different \
inductive bias or a better input representation, not parameter tweaks. A typical ladder, to adapt: \
tabular, boosting, then a different boosting library or bagged trees, then a linear or neural model; text, \
tfidf with a linear model, then frozen embeddings with a head, then fine-tuned transformer; image, \
handcrafted statistics or frozen embeddings with a head, then a fine-tuned pretrained backbone, then a \
stronger backbone or other resolution; audio, summary statistics, then spectrogram CNN or a pretrained \
audio encoder; series, seasonal baseline and boosting on lags, then a sequence model if data is plentiful.
- Exploit (the last run improved the best score): keep the family and make one structural change that the \
evidence supports: stronger regularization or more capacity, a better-aligned objective, native \
categoricals, class weights, a larger resolution or token length, partial unfreezing, augmentation, \
multi-seed averaging.
- Stalled (two runs without gain): change the inductive bias, the objective, the target treatment or the \
input representation; a different column subset or feature block is also valid.

STEP 3 - DESIGN LEVERS
- Objective aligned to the metric: poisson, tweedie or a log target for counts and skewed amounts; L1, \
huber or quantile when outliers or absolute error matter; class weights or imbalance-aware settings when \
the minority class matters; label smoothing, mixup or soft labels for noisy classification.
- Regularization regime: shallow trees with strong shrinkage for noisy or small data; deeper trees when \
underfitting; for deep models, learning rate, weight decay, dropout, augmentation and number of epochs.
- Variance control: several seeds, fold averaging, or test-time augmentation inside the run when fold \
spread is large.
- Post-processing aligned to the metric:
  Ranking and probability metrics (AUC, PR-AUC, ranking scores): save raw probabilities or scores, never \
thresholded.
  Log-loss and Brier: clip probabilities to [1e-6, 1-1e-6] and check calibration (temperature scaling for \
deep models).
  Label metrics (accuracy, F1, MCC, kappa): tune the threshold, per-class thresholds or cut points on \
out-of-fold predictions, then apply them to validation, holdout and test; multiclass takes the argmax.
  Skewed or relative-error regression: train on log1p of the target and invert with expm1, clipped to \
valid bounds; clip or round to valid values when the metric needs it.
  Segmentation or detection: tune the mask threshold, minimum area or NMS threshold on out-of-fold \
predictions.
  Use "none" when nothing applies.

STEP 4 - CHOOSE
Judge each possible run on: evidence (what in SITUATION supports it); upside (how much it could raise \
the best score); diversity (a different but still competitive model adds blend value, and a model far \
below the best earns almost no blend weight); and cost (runtime, memory, GPU need, fragility), which must \
fit the hardware and time budget. Pick the best and change one meaningful thing so the result can be \
credited. Differences smaller than fold-to-fold noise are ties.

HARD LIMITS
- Use only what is installed and confirmed. Size every run so training across all folds finishes within \
the time budget, with early stopping or a fixed short schedule.
- Never repeat a setup (family, parameters, feature version) listed in RUNS or DEAD ENDS. Changing only a \
seed is not a new setup. A duplicate is rejected and you are asked again.
- Reuse family names already in the leaderboard (lightgbm, not lgbm). New families use clear snake_case \
names such as tfidf_logistic, frozen_embedding_ridge, finetune_transformer, finetune_timm_cnn, \
tabular_mlp_torch, gru_sequence, sarimax.
- model_params is a flat object of concrete values (for deep runs: backbone, image_size or max_length, \
epochs, learning rate, batch size, freezing, augmentation). Everything about the metric belongs in \
post_processing.

FORMAT EXAMPLES (level of detail only; invented):
- thinking: The best run is lightgbm at 0.842 with fold spread 0.004 and a CV-holdout gap of 0.011, which \
points to mild overfitting. Only boosting has been tried. A strongly regularized linear model would add a \
different bias if it stays within about 0.01 of the best.
  hypothesis: A scaled logistic regression with L2 on the _cat and _freq columns will come within 0.01 of \
lightgbm and add a decorrelated model. If it trails by more than 0.03 the signal is mostly interactions \
and the family is refuted.
  model_family: logistic_regression | model_params: C 0.5, penalty l2, max_iter 2000, scale numeric \
columns, one-hot the _cat columns | post_processing: tune the threshold on out-of-fold probabilities to \
maximize F1; apply to validation, holdout and test.
- thinking: The ENV finding shows a 24 GB GPU and timm. The tfidf model scores 0.71 and frozen embeddings \
with a head score 0.78, so the representation is the bottleneck, and 8,000 images at 224 pixels fit in the \
budget.
  hypothesis: Fine-tuning a small pretrained convnext for 6 epochs will beat the frozen head by at least \
0.03; if the gain is under 0.01 the frozen head is enough.
  model_family: finetune_timm_cnn | model_params: backbone convnext_tiny, image_size 224, epochs 6, lr \
0.0002, batch_size 32, freeze_epochs 1, augmentation flip and random crop, tta flip | post_processing: \
temperature scaling on out-of-fold logits, then argmax; save probabilities.

SITUATION:
{digest}

THIS ROUND: {round_goal}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 4-6 sentences. State the failure pattern you diagnosed, the confirmed hardware and packages, \
the phase, the options you compared, and why this one wins on evidence, upside, diversity and cost.
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