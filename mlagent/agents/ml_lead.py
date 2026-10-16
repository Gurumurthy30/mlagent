"""The ML Lead: decides validation scheme, baseline description, risks, and playbook.

The LLM writes the plan. The Coder then writes splits.json and code checks it.
Every later agent follows this plan.
"""

from typing import Literal

from pydantic import BaseModel

from mlagent.agents.base import AgentContext, AgentFailedError
from mlagent.agents.coder import last_error_line, run_coder
from mlagent.digest import build_digest, shorten_text
from mlagent.events import EventType
from mlagent.script_contract import SPLITS_FORMAT_TEXT, check_splits_file
from mlagent.settings import (
    BIG_FILE_MB,
    DEFAULT_SPLIT_COUNT,
    HOLDOUT_FRACTION,
    MIN_EXAMPLES_FOR_HOLDOUT,
    RANDOM_SEED,
)
from mlagent.state import ExperimentPlan, Ledger

AGENT_KEY = "ml_lead"
AGENT_NAME = "ML Lead"
RESULT_TAG = "SPLITS"

ValidationScheme = Literal[
    "kfold", "stratified_kfold", "group_kfold", "time_split", "single_split", "repeated_seeds"
]

PLAN_PROMPT = """\
You are the ML Lead of an autonomous Kaggle Grandmaster ML team. Your purpose is to lock in a \
leak-free validation split scheme that mirrors the train-to-test split mechanism, define metric-specific \
post-processing, prescribe a diverse multi-family model portfolio for ensembling, and write the team playbook. \
Every downstream agent follows your plan.

DECISION RULES (in priority order):
1. VALIDATION SCHEME ("TRUST YOUR CV" — MIRROR THE TEST SPLIT):
   - Inspect EDA `train_test_shift`, `validation_hints`, `group_columns`, and `time_columns`:
   - IF test contains unseen entities/groups (`|train_groups ∩ test_groups| == 0`) or rows share cluster/family/group \
identifiers (including composite IDs like `PassengerId` prefix `gggg_pp`, `user_id`, `patient_id`) -> MUST choose `"group_kfold"` \
and specify the column (or derived prefix expression) in `group_or_time_column`.
   - ELSE IF test is strictly future in time relative to train (`time_columns`) -> MUST choose `"time_split"` \
and set `group_or_time_column` to the timestamp/date column.
   - ELSE IF task is binary/multiclass classification (or regression with skewed/multimodal target where binned stratification helps) -> `"stratified_kfold"`.
   - ELSE IF standard continuous regression -> `"kfold"`.
   - ELSE IF deep learning on huge media dataset makes 5-fold CV exceed compute budget -> `"single_split"`.
   - ELSE IF unsupervised / no target labels exist -> `"repeated_seeds"`.
2. DIVERSE MODEL PORTFOLIO (`planned_model_families`):
   - Winning medals requires ensembling genuinely diverse inductive biases (never just 4 runs of LightGBM).
   - List 3-4 distinct model families in priority order matched to the modality:
     * Tabular: `["lightgbm", "catboost", "xgboost", "ridge_or_mlp"]` (or `["logistic_regression", "random_forest", ...]` on tiny data).
     * Text / NLP: `["tfidf_ridge", "tfidf_svd_lightgbm", "pretrained_transformer"]`.
     * Vision / Audio: `["pretrained_cnn_backbone", "vision_transformer_or_effnet", "gbdt_on_pooled_embeddings"]`.
     * Time Series: `["lightgbm_lag_features", "catboost_lag_features", "ridge_trend_model"]`.
3. METRIC-SPECIFIC POST-PROCESSING (`metric_post_processing`):
   - Specify exact mathematical post-processing aligned with `primary_metric`:
     * Threshold metrics (`f1`, `f1_macro`, `accuracy`, `balanced_accuracy`, `mcc`, `qwk`): "Optimize decision threshold(s) on out-of-fold probability predictions via scipy.optimize to maximize primary_metric before converting to class labels."
     * Ranking / probability metrics (`roc_auc`, `pr_auc`, `map_at_k`): "Output raw uncalibrated probabilities (never hard 0/1 labels); use rank-averaging or log-odds blending in ensemble."
     * Calibration loss (`log_loss`, `brier_score`): "Clip predicted probabilities to [1e-6, 1 - 1e-6] and check temperature/isotonic calibration on OOF."
     * Skewed regression / RMSLE (`rmsle`, `rmse`, `mae`, `mape`): "Train on log1p(target) if target is non-negative right-skewed or metric is rmsle, invert predictions with np.clip(np.expm1(pred), min_y, max_y)."
4. TECHNICAL PLAYBOOK (6-8 actionable lines):
   - Line 1-2: Must-have Round 1 foundation features (string/ID decompositions, encodings, row-wise sums, missingness flags).
   - Line 3-4: Round 2+ iterative feature priorities (group-relative diffs/ratios, fold-safe smoothed target encoding, key ratios).
   - Line 5-6: Model exploration order across `planned_model_families` and metric loss/post-processing enforcement.
   - Line 7-8: Guardrails against top leakage risks identified in EDA.
5. GPU UTILIZATION: Set `use_gpu=true` only if deep learning (vision, audio, transformers) or large GPU GBDT training is required and available.

WORKED EXAMPLE (Spaceship Titanic):
- validation_scheme: "group_kfold"
- group_or_time_column: "PassengerId (group prefix before '_')"
- planned_model_families: ["lightgbm", "catboost", "xgboost", "mlp_or_logistic"]
- metric_post_processing: "Optimize binary classification probability threshold on out-of-fold predictions (search 0.35 to 0.65) to maximize accuracy before casting to boolean True/False."

SITUATION:
{digest}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 3-5 sentences analyzing the train-to-test split mechanism, leakage risks, metric geometry, and multi-family ensemble strategy.
- validation_reason: one clear sentence explaining why this split scheme mirrors the test set and prevents leakage.
- validation_scheme: one of "kfold", "stratified_kfold", "group_kfold", "time_split", "single_split", "repeated_seeds".
- group_or_time_column: exact column name (or derived prefix rule like "PassengerId prefix before '_'") used for group_kfold or time_split, else empty string.
- baseline_description: fast, reliable Round 1 baseline model family and objective function.
- planned_model_families: ordered list of 3-4 distinct model families to explore for ensemble diversity.
- metric_post_processing: exact post-processing rule (threshold tuning, probability clipping, target log1p/expm1, or rank averaging) for the primary metric.
- target_score: numeric goal if requested by user, else null.
- use_gpu: boolean, true only if deep learning is needed and GPU is available.
- risks: list of up to 5 project-specific leakage, overfitting, or submission risks and their mitigations.
- playbook: 6-8 concise, high-impact guidance lines covering Round 1 foundation features, Round 2+ interactions, model family sequence, and metric handling."""


class PlanDraft(BaseModel):
    thinking: str = ""
    validation_reason: str = ""
    validation_scheme: ValidationScheme = "kfold"
    group_or_time_column: str = ""
    baseline_description: str = ""
    planned_model_families: list[str] = []
    metric_post_processing: str = ""
    target_score: float | None = None
    use_gpu: bool = False
    risks: list[str] = []
    playbook: str = ""


SPLITS_RESULT_KEYS = """\
- "splits_path": the path you saved the file to
- "n_splits": integer
- "n_examples": number of entries in example_ids (0 when you wrote seeds)
- "n_holdout": number of examples with fold -1"""


def run_ml_lead(ledger: Ledger, agent_context: AgentContext) -> ExperimentPlan:
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "writing the plan")

    # Step 1: Draft the plan using LLM
    digest = build_digest(AGENT_KEY, ledger)
    prompt = PLAN_PROMPT.format(digest=digest)
    draft = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, PlanDraft)

    # Resolve GPU setting with system capability
    can_gpu = agent_context.environment.can_train_on_gpu
    use_gpu_config = agent_context.config.compute.use_gpu
    use_gpu = False if use_gpu_config == "never" or not can_gpu else (True if use_gpu_config == "always" else draft.use_gpu)

    plan = ExperimentPlan(
        validation_scheme=draft.validation_scheme,
        validation_reason=draft.validation_reason.strip(),
        baseline_description=draft.baseline_description.strip(),
        target_score=draft.target_score,
        use_gpu=use_gpu,
        risks=draft.risks[:5],
        playbook=shorten_text(draft.playbook.strip(), 1200),
        group_or_time_column=draft.group_or_time_column.strip(),
        metric_post_processing=draft.metric_post_processing.strip(),
        planned_model_families=draft.planned_model_families,
    )

    # Step 2: Ask Coder to create splits.json
    workspace = agent_context.workspace
    splits_file = workspace.splits_path
    hint_text = "; ".join(ledger.eda.validation_hints) or "none"
    task_text = (
        f"Create the validation splits for this project and save as JSON to {splits_file}.\n"
        f"Scheme: {plan.validation_scheme}. Reason: {plan.validation_reason}\n"
        f"Group or Time column/rule: {plan.group_or_time_column or 'none'}\n"
        f"Target column: {ledger.brief.target or 'none'}, ID column: {ledger.brief.id_column or 'none'}\n"
        f"Brief group_columns: {ledger.brief.group_columns}, time_columns: {ledger.brief.time_columns}\n"
        f"Structure from EDA: {hint_text}\n"
        f"Random seed: {RANDOM_SEED}. Default splits: {DEFAULT_SPLIT_COUNT}.\n"
        f"Project folder: {workspace.project_folder}\n\n"
        "CANONICAL SPLIT IMPLEMENTATION RECIPE:\n"
        "1. Load training data (e.g. `df = pd.read_csv(...)`). Determine `n = len(df)`.\n"
        "2. Initialize: `folds = np.full(n, -1, dtype=int)`.\n"
        "3. Holdout selection (~10% of examples or groups, 0% if n < 500):\n"
        "   - If `group_kfold`:\n"
        "     * Extract group key. If group rule is prefix before '_' (e.g. PassengerId 'gggg_pp'), do `df['group_key'] = df[id_col].astype(str).str.split('_').str[0]`.\n"
        "     * Unique groups: `u_groups = np.array(sorted(df['group_key'].unique()))`.\n"
        "     * Shuffle unique groups: `np.random.RandomState(seed).shuffle(u_groups)`.\n"
        "     * Holdout groups: take first ~10%: `n_hold = max(1, int(len(u_groups) * 0.10))`.\n"
        "     * `holdout_groups = set(u_groups[:n_hold])`.\n"
        "     * Rows in holdout_groups STAY `folds = -1`.\n"
        "     * CV row positions: `cv_idx = np.where(~df['group_key'].isin(holdout_groups))[0]` (0-indexed integer array!).\n"
        "     * Split `cv_idx` with `StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)` if classification, else `GroupKFold(n_splits=n_splits)`:\n"
        "       ```python\n"
        "       for fold_num, (_, val_pos) in enumerate(splitter.split(cv_idx, y.iloc[cv_idx], groups.iloc[cv_idx])):\n"
        "           folds[cv_idx[val_pos]] = fold_num\n"
        "       ```\n"
        "   - If `stratified_kfold`:\n"
        "     * Hold out ~10% with `StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)`.\n"
        "       CAUTION: `train_idx, holdout_idx = next(sss.split(np.zeros(n), y))` (train_idx is 90% for CV, holdout_idx is 10%!).\n"
        "     * Split the 90% `train_idx` with `StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)`:\n"
        "       `for fold_num, (_, val_pos) in enumerate(skf.split(train_idx, y.iloc[train_idx])): folds[train_idx[val_pos]] = fold_num`.\n"
        "   - If `kfold`:\n"
        "     * `perm = np.random.RandomState(seed).permutation(n)`.\n"
        "     * `n_hold = int(n * 0.10)`; `cv_idx = perm[n_hold:]` (holdout `perm[:n_hold]` stays `folds = -1`).\n"
        "     * Split `cv_idx` with `KFold(n_splits=n_splits, shuffle=True, random_state=seed)`.\n"
        "   - If `time_split`:\n"
        "     * Sort rows by time column: `order = df[time_col].argsort().values`.\n"
        "     * Latest ~10% stay `folds = -1`. Split earlier ~90% into `n_splits` chronological windows.\n"
        "4. STRICT INTEGRITY ASSERTIONS (Must pass before saving):\n"
        "   - `assert len(folds) == n, 'Folds length must match dataset rows'`\n"
        "   - `assert not np.isnan(folds).any()`\n"
        "   - `assert set(folds).issubset(set(range(-1, n_splits)))`\n"
        "   - `for f in range(n_splits): assert (folds == f).sum() > 0, f'Fold {f} has no examples'`\n"
        "   - `n_holdout = int((folds == -1).sum())`\n"
        "   - `assert n_holdout <= int(n * 0.25), f'Holdout too large: {n_holdout}/{n}'`\n"
        "5. SAVE SPLITS.JSON TO DISK:\n"
        "   - Save full JSON to `splits_file` with keys: `scheme`, `seed`, `n_splits`, `id_description`: 'row position (0-indexed) in train.csv', `example_ids`: list(range(n)), `fold_of_example`: [int(x) for x in folds.tolist()].\n"
        "6. STDOUT RESULT TAG CONTRACT:\n"
        "   - The very last line printed to stdout MUST be `<SPLITS>{summary_json}</SPLITS>`.\n"
        "   - CRITICAL: In `<SPLITS>`, ONLY include summary metadata: `{'splits_path': str(splits_file), 'n_splits': n_splits, 'n_examples': n, 'n_holdout': n_holdout}`.\n"
        "   - NEVER print the huge `example_ids` or `fold_of_example` arrays to stdout (they are already saved in splits.json). Printing huge arrays will cause stdout buffer truncation!\n"
    )

    last_err = ""
    for try_num in range(1, 3):
        coder_task = task_text if not last_err else f"{task_text}\n\nPrevious attempt failed with:\n{last_err}"
        coder_res = run_coder(
            task_text=coder_task,
            result_tag=RESULT_TAG,
            result_keys_text=SPLITS_RESULT_KEYS,
            working_folder=workspace.artifacts_folder / AGENT_KEY,
            script_name=f"make_splits_try_{try_num}.py",
            caller_key=AGENT_KEY,
            ledger=ledger,
            agent_context=agent_context,
        )
        problems = check_splits_file(splits_file)
        if not problems:
            last_err = ""
            break
        if coder_res.succeeded:
            last_err = "Splits file failed validation: " + " ".join(problems)
        else:
            last_err = last_error_line(coder_res.error_text)
            if problems and "does not exist" not in problems[0]:
                last_err = f"Splits file issues ({' '.join(problems)}) | Script error: {last_err}"
        event_bus.emit(EventType.WARNING, AGENT_NAME, f"splits try {try_num} failed: {last_err}")

    if last_err:
        raise AgentFailedError(f"Could not create validation splits: {last_err}")

    plan.splits_path = str(splits_file)
    event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, f"plan ready: {plan.validation_scheme}")
    return plan