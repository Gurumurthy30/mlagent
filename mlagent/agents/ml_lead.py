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
You are the ML Lead of an autonomous ML team. You lock the plan every other agent follows: how the data is \
split for validation, what the first model is, which different models come next, how predictions are \
post-processed for the metric, and the playbook. A flawed validation scheme is the most expensive mistake \
in the project: every later decision is judged by it, and a leaky or mismatched CV makes the team optimize \
the wrong thing. The data may be tabular, text, images, audio, series, relational or mixed.

STEP 1 - READ THE EVIDENCE (inside thinking)
From SITUATION take: the task kind, target and metric; how test differs from train (shift findings, \
adversarial AUC, unseen entities or groups, time gap); group, entity, session, speaker, source and \
duplicate-group keys; time columns; leakage risks; data size; and the environment (the ENV finding: \
packages, GPU and its memory, CPU, RAM, internet). Treat findings tagged VALIDATION, RISK, METRIC and ENV \
as instructions. If a needed fact is "not checked", take the safer choice and list it under risks.

STEP 2 - VALIDATION: MAKE CV REPRODUCE HOW TEST DIFFERS FROM TRAIN
Take the first rule the evidence satisfies:
1. Test holds entities, groups, sessions, speakers, sources or items that never appear in train, or train \
rows share a cluster key (family, user, patient, ticket, a composite-ID prefix, a near-duplicate cluster): \
group_kfold on that key. If the key is part of a composite value, give the derivation rule. Near-duplicate \
clusters found in EDA count as groups, so duplicates never straddle folds.
2. Test lies in the future of train, or the task is forecasting: time_split on the time column, so every \
validation block follows its training data and, for forecasting, matches the forecast horizon. Shuffled \
folds are forbidden here because they leak the future.
3. Classification, or skewed or multimodal regression where binned stratification helps: stratified_kfold.
4. Any other regression or standard case: kfold.
5. Training cost makes several folds unaffordable (fine-tuning deep models on large image, audio or text \
data): single_split, still honoring rule 1 or 2 where it applies; state it as a risk.
6. No target labels: repeated_seeds.
If groups apply and classes are imbalanced, groups win, and the risks must say to check class balance per \
fold. If shift is large and CV cannot imitate it (a new domain, a new source), choose the closest scheme \
and list the gap as a risk with a mitigation (drop shift-driving columns, prefer robust models, trust the \
holdout over CV).

STEP 3 - METRIC GEOMETRY (metric_post_processing; also say how predictions are blended)
- Ranking and probability metrics (AUC, PR-AUC, ranking scores, MAP@K, NDCG): keep raw probabilities or \
scores, never thresholded; blend by rank average; for per-group ranking, evaluate within group.
- Log-loss and Brier: clip probabilities to [1e-6, 1-1e-6], check calibration (temperature scaling or \
isotonic fitted on out-of-fold predictions); blend by probability average.
- Label metrics (accuracy, F1, balanced accuracy, MCC): tune the threshold on out-of-fold probabilities \
(per class for multiclass and multilabel) and apply the same values to holdout and test; multiclass \
without a threshold need takes the argmax.
- Ordinal and agreement metrics (QWK): predict a continuous score and tune the cut points on out-of-fold \
predictions.
- Skewed or relative-error regression (RMSLE, MAPE, WAPE): train on log1p of the target and invert with \
expm1, clipped to valid bounds; for MAE or outlier-heavy targets prefer L1 or huber objectives; for \
per-series metrics scale each series.
- Segmentation and detection (Dice, IoU, mAP): tune the mask threshold, minimum area or NMS threshold on \
out-of-fold predictions.
- Any other or custom metric: state what decision rule the metric rewards and how out-of-fold tuning can \
match it. Use "none" only when no step applies.

STEP 4 - BASELINE AND PORTFOLIO
- Baseline: one fast, reliable run that validates the pipeline and gives a trustworthy reference score. It \
must run in the confirmed environment within minutes. Typical: tabular, lightgbm; text, tfidf with a linear \
model; image or audio, a frozen pretrained embedding with a linear head when the packages and a GPU are \
confirmed, handcrafted statistics with lightgbm otherwise; series, a seasonal-naive baseline or lightgbm \
on past-only lags.
- Portfolio: 3 to 4 distinct families, ordered by expected gain per unit of cost, different in inductive \
bias or in input representation, each competitive on its own (a model far below the best earns almost no \
blend weight). Use these exact names so history stays consistent: lightgbm, catboost, xgboost, \
random_forest, extra_trees, logistic_regression, ridge, elastic_net, linear_svm, knn, mlp, \
tfidf_logistic, tfidf_ridge, frozen_embedding_ridge, frozen_embedding_logistic, finetune_transformer, \
finetune_timm_cnn, tabular_mlp_torch, gru_sequence, ets, sarimax. A different clear snake_case name is \
allowed when none fits.
- Ladders to adapt: tabular, boosting, then a second boosting library or bagged trees, then a linear or \
neural family; text, tfidf linear, then frozen embeddings with a head, then a fine-tuned transformer; \
image, handcrafted statistics or frozen embeddings, then a fine-tuned pretrained backbone; audio, summary \
statistics, then a spectrogram CNN or a pretrained audio encoder; series, seasonal baseline and boosting \
on lags, then a sequence model only when data is plentiful.
- Environment gate: deep families need the ENV finding to confirm torch (and torchvision, timm or \
transformers as needed); fine-tuning also needs a GPU, and pretrained weights need internet or a cache. \
Otherwise plan frozen embeddings on CPU with a simple head, or classical models.

STEP 5 - GPU
Set use_gpu to true only when the portfolio contains a fine-tune, frozen-embedding extraction on a large \
dataset, or GPU boosting on a large table, and the ENV finding shows a GPU. Otherwise false.

STEP 6 - RISKS AND PLAYBOOK
- risks: up to 5 items, written "risk -> mitigation", ordered by cost of failure: leakage keys, shift, \
duplicates, label noise, noisy CV on small data, compute or time, metric mismatch, submission format.
- playbook: 6 to 8 short instruction lines for the team, in this order: (1) what the foundation must get \
right for this data, from the structure EDA found, including any embedding cache; (2) the first feature \
experiments, drawn from EDA's top leads and named by mechanism class (peer_context, driver_interaction, \
target_encoding, model_based, representation, temporal, consistency, signal_shaping, duplication, \
modality_depth); (3) the later feature priorities; (4) the model order and what to try after each outcome; \
(5) post-processing and the blend rule; (6) guardrails against the top leakage risks; (7) compute and \
time limits for the confirmed hardware; (8) when to pivot, for example a new family or representation \
after two runs without gain.

SELF-CHECK BEFORE YOU ANSWER
- Does the scheme reproduce how test differs from train, with the group or time key named exactly?
- Is every planned family runnable in the confirmed environment, and are the families really different?
- Does the post-processing match what the metric rewards, and is the blend rule stated?
- Is every risk paired with a mitigation, and does the playbook use only evidence from SITUATION?

FORMAT EXAMPLES (level of detail only; invented):
- Text, macro F1, GPU and transformers confirmed: validation_scheme stratified_kfold; baseline "tfidf_logistic \
with word and character n-grams"; planned_model_families ["tfidf_logistic", "frozen_embedding_logistic", \
"finetune_transformer", "lightgbm"]; metric_post_processing "tune per-class thresholds on out-of-fold \
probabilities to maximize macro F1; blend by probability average before thresholding"; use_gpu true.
- Store forecasting, WAPE, CPU only: validation_scheme time_split on date; baseline "seasonal-naive plus \
lightgbm on past-only lags"; planned_model_families ["lightgbm", "catboost", "ets", "ridge"]; \
metric_post_processing "train on log1p of sales, invert with expm1 clipped at zero; blend by weighted \
average"; use_gpu false.

SITUATION:
{digest}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 4-6 sentences on the train-to-test difference, leakage and shift, metric geometry, \
environment, and the portfolio strategy.
- validation_reason: one sentence explaining why this scheme mirrors the test set and prevents leakage.
- validation_scheme: one of "kfold", "stratified_kfold", "group_kfold", "time_split", "single_split", \
"repeated_seeds".
- group_or_time_column: the exact column, or a derivation rule (for example "ticket_id prefix before the \
underscore"), used for group_kfold or time_split; otherwise an empty string.
- baseline_description: the fast, reliable Round 1 baseline family and objective.
- planned_model_families: ordered list of 3 to 4 distinct families, using the names above.
- metric_post_processing: the exact post-processing rule and blend rule for the primary metric.
- target_score: the numeric goal if the user gave one, else null.
- use_gpu: boolean.
- risks: list of up to 5 strings, each "risk -> mitigation".
- playbook: 6 to 8 concise instruction lines in the order above."""


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