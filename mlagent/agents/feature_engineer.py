from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel

from mlagent.agents.base import AgentContext
from mlagent.agents.coder import run_coder
from mlagent.digest import build_digest
from mlagent.events import EventType
from mlagent.scoring import all_finite, is_counted_gain, score_improvement
from mlagent.script_contract import SPLITS_FORMAT_TEXT
from mlagent.settings import RANDOM_SEED
from mlagent.state import FeatureRecord, Ledger

AGENT_KEY = "feature_engineer"
AGENT_NAME = "Feature Engineer"

FOUNDATION_RESULT_TAG = "FEATURES"
SCREEN_RESULT_TAG = "SCREEN"

FOUNDATION_PROMPT = """\
You are the Feature Engineer of an autonomous ML team. In this round you write the plan for features_v1, \
the preparation pipeline every model will train on. A Coder implements your plan exactly as written, in \
the order you list it. Your standard is a pipeline a senior Kaggle competitor would trust without \
rewriting: correct, leak-free, complete and fast. Clever experimental features come in later rounds; \
do not add them here.

HOW TO WORK
Defaults first, evidence second. Apply every default below to each column it fits. Change, add or skip \
a treatment only when SITUATION (EDA findings, column roles, planned model families, train/test shift) \
gives a reason, and state that reason. Every raw column must end up accounted for: used as is, \
transformed, decomposed, or dropped with a stated reason. For wide tables, plan by role or name \
pattern, not column by column.

A. READ THE DATA
Identify the modality (tabular, text, image, audio, time series, multi-table or mixed), the metric, the \
planned model families (tree-based or not), and any column that exists in train but not in test.

B. REPAIR (first, so later steps see clean values)
- Normalize category labels: strip, unify case and punctuation, merge obvious variants.
- Turn missing-value tokens ("?", "NA", "N/A", "null", "none", "unknown", "-", empty) into real missing values.
- Numeric sentinels (-1, -999, 9999, 99999 and similar) and impossible values (negative age, price or \
duration, future dates, coordinates out of range) become missing only when the evidence supports it: a \
single extreme value covering over 1 percent of rows, or values that cannot be real. Keep a missing flag.
- Numbers stored as text (currency, commas, percent signs, units): convert to numbers. Mixed-type \
columns: keep the dominant type. Yes/no, y/n, true/false, t/f and 1/0 strings: map to 0/1.

C. STRUCTURE (before anything is encoded or dropped)
- Split or parse composite and coded strings (parts joined by / - _ : or fixed-width codes) and dates \
stored as text.
- Delimited lists (tags, genres, "a|b|c"): item count plus multi-hot for the 20 most frequent training items.
- Names: keep only a title or family key as an entity key; drop the full name. Paths, URLs and emails: \
extension, depth, folder or domain as categoricals.
- Coordinates: keep, and add distance to the training centroid and 45-degree rotated axes.
- Row-unique identifiers and group-carrying IDs: use them only to derive features, then drop them. Never \
output an arbitrary ID code as a feature.
- Entity and group counts (group size, key frequency, solo flag, position in group): count over ALL rows \
of train and test combined (target-free) and store the lookup in state, so a row gets the same value in \
whichever fold it lands.

D. MISSING VALUES
- Numeric: add <col>_is_missing when 1 to 95 percent is missing or EDA tied missingness to the target or \
a segment; fill with the training median, or 0 when missing logically means none.
- Categorical: fill with the level "missing". Add row_missing_count when many columns have gaps.
- Impute deliberately; the pipeline's last-resort fill is 0.0 and is often wrong.
- Drop columns that are over 95 percent missing, constant, near-constant (over 99.5 percent one value, \
unless EDA shows the rare value matters), exact duplicates, or absent from test.

E. CATEGORICAL
- Natural order (S/M/L, grades, education, rating text): ordinal map in that order, named <col>_ord. \
Otherwise arbitrary codes are <col>_cat.
- Two levels: 0/1. Up to 10 levels: <col>_cat, plus one-hot only if a linear, kNN or neural family is \
planned. Over 10 levels: <col>_cat plus <col>_freq (count over all rows of train and test).
- Group levels seen in fewer than 10 training rows (or under 0.5 percent) into "other"; unseen levels map \
to a fixed fallback. No target-based encoding in this round.
- Integer columns with 20 or fewer distinct values that are really codes (store, region, month, rating): \
keep the number and also add <col>_cat.
- String columns where nearly every value is unique and no structure was found: drop.

F. NUMERIC
- Keep the raw column. Zero-inflated (over 40 percent zeros with a positive tail): add <col>_is_zero and \
<col>_log.
- Heavy-tailed (absolute skew above 2, or max above 50 times the median): add <col>_log (log1p for \
non-negative, sign(x)*log1p(abs(x)) otherwise) whenever any planned family is not tree-based. Skip \
single-column transforms if only trees are planned. Leave bounded rates and percents alone.
- Clip to training quantiles (0.5 and 99.5 percent) only when EDA shows the extremes are errors or a \
planned linear or neural family would break on them.
- Related column groups (amounts, repeated measurements, sensor channels): row-wise sum, mean, std, max, \
nonzero count, zero-total flag.
- Drop one of any pair with absolute correlation above 0.999. Standardize as the very last step, and \
only if a linear, kNN or neural family is planned.

G. TIME, TEXT, IMAGE, AUDIO, TABLES
- Date and time: parse, then year, month, day, day of week, hour, weekend flag, sin/cos for periodic \
parts, elapsed time since the training minimum. Within an entity, add position in time order and gap \
since the previous row, from past rows only. No lag or rolling features of the target in this round.
- Free text (multi-word values, over about 20 distinct values; short label-like text is categorical): \
lowercase and strip HTML and URLs; character, word, uppercase, digit and punctuation counts; URL or \
email presence; word 1-2 gram TF-IDF (min_df 3, at most 20000 features) reduced by TruncatedSVD to 32 \
dense components, fitted on training data.
- Images (pillow, numpy, joblib): shrink to a thumbnail of at most 64 pixels before any array work; \
width, height, aspect ratio, per-channel mean and std, brightness, contrast, coarse histogram. \
Unreadable files become missing values, never crashes.
- Audio (scipy): duration, sample rate, RMS energy, zero-crossing rate, coarse spectral band energies, \
from at most the first 30 seconds. Unsupported formats become missing values.
- Multi-table: aggregate each child table to its join key (count, sum, mean, min, max, nunique) before \
merging, then confirm the row count is unchanged.

H. SHIFT GUARD
Using the train/test shift in EDA, drop or convert to a relative form any column that identifies the \
split (row index, absolute timestamps outside the test range, batch or source IDs), unless EDA shows it \
carries real signal and the shift is mild.

HARD LIMITS
- Never use the target. Every learned statistic (medians, quantiles, vocabularies, rare-level lists, \
TF-IDF, SVD, scaler) is fitted on training data only and applied unchanged to validation and test. The \
only exception is target-free counting of keys, groups and categories over all rows (sections C and E).
- Use nothing that would be unknown at prediction time. Respect time order; look backward only.
- Use only the installed toolbox: pandas, numpy, scipy, scikit-learn, statsmodels, pillow, joblib, \
lightgbm, xgboost, catboost. No plotting, no deep-learning, audio or vision packages, no internet. If an \
ideal representation needs one, plan the closest cheap substitute.
- The final table is numeric and finite, with fixed column order, and contains no raw strings, raw \
identifiers or target.
- Keep it fast and reproducible: vectorized pandas and numpy, no per-row Python loops on large data, no \
randomness beyond the given seed, row count preserved.

NAMING
Use the suffixes _is_missing, _is_zero, _log, _cat, _ord, _freq and _cnt so later rounds and models can \
recognize each column's role.

SELF-CHECK BEFORE YOU ANSWER
- Is every raw column accounted for, and every override of a default justified by evidence?
- Do the steps run in order (repair, structure, missing, categorical, numeric, time and modality, scale)?
- Does each step name exact columns and an exact rule, so a Coder can implement it without guessing?
- Are similar operations merged into one step?
- Does anything touch the target, validation or test statistics (beyond target-free counts), or the future?

FORMAT EXAMPLE (level of detail only; these columns are invented):
- name: repair_sentinels | why: income has 1,840 rows equal to -999 (4 percent) | what_to_do: set income \
== -999 to missing, add income_is_missing, then fill with the training median.
- name: add_group_counts | why: ticket_id repeats across rows and EDA shows shared tickets behave alike | \
what_to_do: compute ticket_cnt = count of rows per ticket_id over train and test combined, plus \
ticket_is_solo = (ticket_cnt == 1); store the lookup in state; unseen ticket_id gets 1.

SITUATION:
{digest}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: at most 10 sentences covering the modality, the column groups (role, problem, planned \
treatment), every override of a default and its evidence, and what the planned models need.
- steps: ordered list of objects with why (the finding or data fact behind the step), name \
(snake_case), and what_to_do (exact columns, rule or formula, fill values, and which statistics are \
fitted on training data). Typically 8 to 16 steps."""

PROPOSE_PROMPT = """\
You are the Feature Engineer of an autonomous ML team, now running a feature experiment round. \
features_v{current_version} already holds the foundation and every feature that has won so far. Your job \
is to design the next set of experiments, up to {max_candidates} candidates, that most increase the \
cross-validation score. Think like a research scientist with a small budget: every candidate is a \
hypothesis with a mechanism, an expected effect and a way to be proven wrong.

HOW YOUR CANDIDATES ARE JUDGED
- Each candidate is added to the current feature table on its own and scored against the baseline with a \
fast proxy model (the best current model family), the same seed and the same CV folds. A candidate is \
kept only if its gain clearly exceeds fold-to-fold noise. Small gains are ties, not wins.
- All winners are then merged into features_v{current_version}+1 together, untested as a group. So make \
candidates independent of each other. If columns only make sense together, put them in ONE candidate.
- One candidate may produce several columns, but it must express ONE mechanism.
- The proxy can only reward information it can use. A feature that mainly helps a different model family \
may not show a gain here.
- Each candidate costs a screening run. Spend your slots on the ideas with the best expected return.

STEP 1 - DIAGNOSE (do this inside thinking, from SITUATION, in this priority order)
1. The Reviewer's latest bottleneck and instructions, and any newest Data Analyst findings. These override \
everything else.
2. Error analysis of out-of-fold predictions, if present: which segments, ranges or classes are predicted \
worst? A feature that attacks the biggest error segment beats a feature that is merely plausible.
3. Importance or ablation evidence, if present: where does the signal concentrate? Build around the \
strongest drivers.
4. Open EDA feature ideas and findings not yet tested.
5. History in FEATURES TRIED and DEAD ENDS: which mechanism families won, which lost, and why (a clear \
loss, a gain inside the noise, or an error or crash). A crashed or errored idea may be retried once with \
the bug fixed. A clear loss is not retried in any variation of the same mechanism.

STEP 2 - PICK THE PHASE, THEN THE PORTFOLIO
- Early (few or no feature rounds yet): breadth. Cover different mechanism classes whose preconditions \
this data meets, led by the strongest evidence.
- Productive (the last round had winners): exploit. Deepen the winning mechanism: more keys, columns, \
windows or aggregations, combined ingredients, or the part it failed to capture. Keep at most one slot \
for a new mechanism.
- Stalled (the last two rounds produced no winner): change the mechanism class or the target of the attack \
(a different error segment), or pass.
Never fill slots with near-duplicates. Different candidates should test different mechanisms.

STEP 3 - MECHANISM CLASSES (a menu, not a checklist; use only classes whose preconditions hold here, and \
invent others if the data demands it)
- peer_context: rows belong to entities, groups, categories or neighbors. Compare each row with its peers: \
group mean, std, min, max, nunique, rank, deviation, ratio to the group mean, share of the group total, \
group size, position in group.
- driver_interaction: a few strong drivers dominate. Differences, ratios, products and sums between them; \
crosses of two strong categoricals (then encoded); a value conditioned on a category or flag. Tree models \
split single columns well but struggle with ratios and differences across columns.
- target_encoding: high-cardinality categoricals, crosses or group keys with real target signal. Smoothed \
target mean or rate (and std or quantiles for regression), computed out-of-fold inside the training fold.
- model_based: another model view adds information. Out-of-fold predictions of a simple model (ridge, \
logistic, shallow tree, naive Bayes) on a block of columns, on TF-IDF or on image statistics; KMeans \
cluster id and centroid distances; PCA or SVD components of a column block; kNN neighbor statistics (small \
and medium data only); anomaly scores. Anything supervised is out-of-fold.
- temporal: time, order or entity histories exist. Lags, differences, rolling and expanding statistics, \
time since the last event, trailing event counts, trend slope, seasonal baselines per entity. Past values \
only, computed per entity in time order.
- consistency: columns constrain each other or missingness is structured. Flags for impossible or \
inconsistent combinations, inferring a missing value from a related column or from the same group, \
missingness patterns per group.
- signal_shaping: a numeric is noisy, discrete or periodic. Quantile or rank transforms (mainly for \
non-tree families), binning on training quantiles, digit or rounding patterns, distance to a meaningful \
reference point, extreme-value flags.
- duplication: exact or near-duplicate rows (row hash counts), rare combination frequencies, key overlap \
between splits.
- modality_depth: text (character n-grams, pattern flags, length ratios, out-of-fold linear probability on \
TF-IDF); image (gradient and edge energy from numpy differences, color histograms, grid-region statistics, \
sharpness proxies); audio (spectral centroid, rolloff and band ratios from FFT or Welch, \
autocorrelation periodicity); multi-table (recency-window aggregates, ratios between aggregates).
Use these exact class names as the family value, or "other" for anything outside them. Consistent labels \
let the team see which kinds of ideas pay off.

STEP 4 - CHOOSE THE BEST {max_candidates}
For every possible idea, judge four things:
- Evidence: what in SITUATION points to it?
- Upside: how large is the gain if it works? New information beats re-expressing information the model \
already has.
- Novelty: can the model already derive it from existing features, or has it been tried?
- Cost and risk: leakage risk, runtime on this many rows, fragility.
Keep the ideas with the best evidence, upside and novelty for the lowest cost and risk.

STEP 5 - SPECIFY SO A CODER CAN IMPLEMENT WITHOUT GUESSING
Every idea must state: the exact input columns (raw names or existing feature names such as <col>_cat, \
<col>_freq, <col>_log), the exact computation, which statistics are fitted on the training fold and how \
(out-of-fold for anything that touches the target), the fallback for unseen keys and missing values, and \
the output column names.

HARD LIMITS
- Leakage: anything built from the target is computed out-of-fold inside the training fold, smoothed, so \
training rows never see their own label. No statistic is ever fitted on validation or test rows. Time and \
order features look backward only. Do not use row order or row position as a feature unless EDA proved it \
is meaningful.
- Toolbox only: pandas, numpy, scipy, scikit-learn, statsmodels, pillow, joblib, lightgbm, xgboost, \
catboost. No plotting, no deep-learning, audio or vision packages, no internet.
- Cost: vectorized code only. No per-row Python loops and no all-pairs computation on large data; sample \
or approximate instead.
- Safe outputs: finite values, a fallback for unseen keys, no dependence on columns absent from test.
- Names: every name is unique snake_case and never reuses a name from FEATURES TRIED or DEAD ENDS.

PASS RULE
Return an empty list only if every mechanism class whose preconditions hold has had a fair attempt, or \
the last two rounds produced no winners and nothing new in SITUATION points elsewhere. Otherwise propose \
your best ideas.

FORMAT EXAMPLE (level of detail only; these columns are invented):
- finding_id: "error_analysis" | family: "peer_context" | name: store_sales_vs_region_mean | reason: \
Error analysis shows the largest errors on stores that differ most from their region, so the model lacks \
peer context. If the deviation carries signal, those errors shrink; if the gain is inside the noise, the \
mechanism is refuted. | idea: Per region, compute mean and std of weekly_sales on the training fold and \
store them in state; create store_sales_dev = weekly_sales - region_mean and store_sales_z = store_sales_dev \
/ (region_std + 1e-5); unseen regions get 0 for both.
- finding_id: "finding_006" | family: "target_encoding" | name: tgt_enc_brand_x_color | reason: brand and \
color separately show weak signal but the cross is sparse (about 400 levels) and EDA shows large \
differences between combinations. If the cross-level target rate is stable, CV improves; otherwise the \
cross is noise. | idea: Build cross = brand + "_" + color; within the training fold compute smoothed \
out-of-fold target mean with an inner 5-fold split, (n * level_mean + 20 * global_mean) / (n + 20); for \
validation rows use the full-training-fold level means; unseen levels get the global mean; output \
tgt_enc_brand_x_color.

SITUATION:
{digest}

EXISTING FUNCTIONS IN FEATURES FILE:
{existing_features}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: at most 10 sentences covering the bottleneck and the strongest evidence, what history says \
about winning and losing mechanisms, the phase you chose, and why these slots and not others.
- candidates: up to {max_candidates} objects {{"finding_id": str, "family": str, "reason": str, "name": \
str, "idea": str}}. finding_id is a finding id from EDA, or one of reviewer_diagnosis, error_analysis, \
history, exploration. reason is the hypothesis: the mechanism, the expected effect, and what result would \
refute it. Return [] to pass."""


class FoundationStep(BaseModel):
    why: str = ""
    name: str
    what_to_do: str


class FoundationDraft(BaseModel):
    thinking: str = ""
    steps: list[FoundationStep] = []


class CandidateSpec(BaseModel):
    finding_id: str = ""
    family: str = "general"
    reason: str = ""
    name: str
    idea: str


class CandidateList(BaseModel):
    thinking: str = ""
    candidates: list[CandidateSpec] = []


def inspect_features_file(path: Path) -> str:
    """Extract top-level function names, docstrings, and key operations from a features module."""
    if not path.is_file():
        return "No features file exists yet."
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception as e:
        return f"Error reading features file: {e}"
    lines = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node) or ""
            doc_line = doc.strip().splitlines()[0] if doc.strip() else "no docstring"
            args = ", ".join(a.arg for a in node.args.args)
            lines.append(f"- def {node.name}({args}): {doc_line}")
    text = "\n".join(lines) or "No functions defined."
    if len(text) > 1500:
        return text[:1490] + " ..."
    return text


def build_foundation(ledger: Ledger, agent_context: AgentContext) -> dict:
    """Round 0: Create features_v1.py."""
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "creating foundation features_v1.py")

    digest = build_digest(AGENT_KEY, ledger, char_limit=agent_context.config.llm.context)
    prompt = FOUNDATION_PROMPT.format(digest=digest)
    draft = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, FoundationDraft)

    workspace = agent_context.workspace
    features_v1_path = workspace.features_file(1)
    workspace.features_folder.mkdir(parents=True, exist_ok=True)

    steps_text = "\n".join(
        f"{i}. {s.name}: {s.what_to_do} (Why: {s.why})"
        for i, s in enumerate(draft.steps, 1)
    )
    raw_keep_text = ", ".join(ledger.eda.raw_columns_to_keep) or "all raw string/ID/categorical columns until final return"
    role_cols_text = (
        f"Target column (NEVER include in output features): {ledger.brief.target or 'none'}\n"
        f"ID column (use for prefix/group extraction if applicable, then drop before return): {ledger.brief.id_column or 'none'}\n"
        f"Group columns: {ledger.brief.group_columns}\n"
        f"Time columns: {ledger.brief.time_columns}\n"
        f"Text columns: {ledger.brief.text_columns}\n"
        f"Raw columns to parse before dropping: {raw_keep_text}"
    )

    coder_task = (
        f"Create the foundational features module at {features_v1_path}.\n"
        "Implement complete Round 1 data preparation and feature engineering based on these steps:\n"
        f"{steps_text}\n\n"
        f"COLUMN ROLES:\n{role_cols_text}\n\n"
        "CONTRACT INTERFACE (must be strictly followed):\n"
        "1. fit_preparation(train_data, seed=42) -> state dict: fits all imputers, frequency maps, ordinal encoders, "
        "scalers, or SVD strictly on train_data (ignoring the target column if present) and stores `state['feature_columns']`.\n"
        "2. apply_preparation(data, state) -> pd.DataFrame: transforms `data` using fitted `state`:\n"
        "   - Keep raw string/categorical/ID columns accessible while computing decomposed/engineered features.\n"
        "   - Handle unseen categories in test/val safely (e.g. `.map(mapping).fillna(-1)` or `.fillna(0)`).\n"
        "   - Drop the target column (if present in `data`), raw unencoded string/object columns, and raw ID columns ONLY at the very end.\n"
        "   - Reindex/select `state['feature_columns']`, cast to clean numeric dtypes (`float32`/`int32`), and replace any remaining `inf`/`-inf`/`NaN` with `0.0`.\n\n"
        f"Project folder (read-only): {workspace.project_folder}\n\n"
        "Under `if __name__ == '__main__':`, load a sample of train and test data, run `state = fit_preparation(train_sample)` "
        "and `X_tr = apply_preparation(train_sample, state)`, assert `X_tr.isna().sum().sum() == 0` and `len(X_tr) == len(train_sample)`, and print JSON:\n"
        f'<{FOUNDATION_RESULT_TAG}>{{"status": "ok"}}</{FOUNDATION_RESULT_TAG}>'
    )

    _ = run_coder(
        task_text=coder_task,
        result_tag=FOUNDATION_RESULT_TAG,
        result_keys_text='- "status": "ok"',
        working_folder=workspace.artifacts_folder / "features",
        script_name="features_v1.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )

    # Guarantee features_v1.py exists if Coder happened to write create_features_v1.py
    v1_file = workspace.features_file(1)
    legacy_v1 = workspace.artifacts_folder / "features" / "create_features_v1.py"
    if not v1_file.is_file() and legacy_v1.is_file():
        v1_file.write_text(legacy_v1.read_text(encoding="utf-8"), encoding="utf-8")

    new_features: list[FeatureRecord] = []
    for s in draft.steps:
        new_features.append(
            FeatureRecord(
                feature_id=f"feature_{len(new_features) + 1:03d}",
                name=s.name,
                family="foundation",
                reason=s.what_to_do,
                kept=True,
                feature_version=1,
                iteration=ledger.iteration,
            )
        )

    event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, "features_v1.py created")
    return {
        "features": new_features,
        "current_feature_version": 1,
        "last_features_changed": True,
    }


def run_feature_engineer(ledger: Ledger, agent_context: AgentContext) -> dict:
    """Propose and screen new feature candidates or build foundation."""
    event_bus = agent_context.event_bus

    # If version is 0, build initial foundation features
    if ledger.current_feature_version == 0:
        return build_foundation(ledger, agent_context)

    event_bus.emit(
        EventType.AGENT_STARTED, AGENT_NAME, f"features round {ledger.iteration}: proposing candidates"
    )

    # 1. Propose candidates
    workspace = agent_context.workspace
    current_features_path = workspace.features_file(ledger.current_feature_version)
    existing_funcs = inspect_features_file(current_features_path)
    max_candidates = agent_context.config.limits.candidates_per_feature_round
    digest = build_digest(AGENT_KEY, ledger, char_limit=agent_context.config.llm.context)
    propose_prompt = PROPOSE_PROMPT.format(
        digest=digest,
        existing_features=existing_funcs,
        max_candidates=max_candidates,
        current_version=ledger.current_feature_version,
    )
    event_bus.emit(
        EventType.STATUS,
        AGENT_NAME,
        f"proposing candidates for round {ledger.iteration}",
        {"reason": "exploring feature engineering candidates"},
    )
    proposal = agent_context.llm_client.ask_structured(AGENT_NAME, propose_prompt, CandidateList)

    candidates = proposal.candidates
    if not candidates:
        summary_line = f"Features round {ledger.iteration}: passed (no idea worth testing)."
        event_bus.emit(EventType.INFO, AGENT_NAME, summary_line)
        return {
            "features": ledger.features,
            "current_feature_version": ledger.current_feature_version,
            "last_features_changed": False,
        }

    # 2. Screen candidates with Coder
    event_bus.emit(
        EventType.STATUS,
        AGENT_NAME,
        f"screening {len(candidates)} candidate features",
        {"reason": f"testing {', '.join(c.name for c in candidates[:3])}"},
    )
    candidates_text = "\n".join(
        f"- Name: {c.name} | Family: {c.family} | Idea: {c.idea} (Reason: {c.reason})"
        for c in candidates
    )
    direction = "higher" if ledger.brief.higher_is_better else "lower"
    screening_folder = workspace.artifacts_folder / "screening"
    screening_folder.mkdir(parents=True, exist_ok=True)
    best_run = ledger.best_approved_run()
    proxy_family = best_run.model_family if best_run and best_run.model_family else "fast gradient boosted trees (e.g. LightGBM)"

    screen_task = (
        f"Screen these {len(candidates)} candidate features on the validation splits:\n"
        f"{candidates_text}\n\n"
        f"Current features file: {current_features_path}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Primary metric: {ledger.brief.primary_metric} ({direction} is better)\n"
        f"Metric post-processing hint: {ledger.plan.metric_post_processing or 'none'}\n"
        f"Preferred fast proxy model family: {proxy_family}\n"
        f"Project folder: {workspace.project_folder}\n"
        f"Target column: {ledger.brief.target or 'none'}, ID column: {ledger.brief.id_column or 'none'}\n"
        f"Seed: {RANDOM_SEED}\n\n"
        "STRICT FOLD-SAFE SCREENING PROTOCOL:\n"
        "1. Load training data and splits (`fold >= 0` are CV folds; ignore holdout `fold == -1` during screening).\n"
        "2. For each CV fold `k`: call `base_state = fit_preparation(train_fold, seed=Seed)` and transform `train_fold` and `val_fold` "
        "with `apply_preparation`. Train the fast proxy model with fixed hyperparameters and `random_state=Seed` to compute `baseline_scores` (one float per CV fold).\n"
        "3. For each candidate feature:\n"
        "   - Compute the candidate feature column(s) from raw `train_fold` / `val_fold` (fitting any group means, target encodings, or frequency maps STRICTLY on `train_fold` only, never on full data!).\n"
        "   - Replace `inf`/`-inf`/`NaN` in the new feature column(s) with `0.0`, append to the base feature matrix for fold `k`, and evaluate with the EXACT same proxy model and seed across all CV folds.\n"
        f"4. Print JSON <{SCREEN_RESULT_TAG}> with `baseline_scores` and `candidates` list (`name`, `scores`, `error`).\n\n"
        f"{SPLITS_FORMAT_TEXT}"
    )

    coder_res = run_coder(
        task_text=screen_task,
        result_tag=SCREEN_RESULT_TAG,
        result_keys_text='- "baseline_scores": list of numbers\n- "candidates": list of {"name": str, "scores": list of numbers}',
        working_folder=screening_folder,
        script_name=f"screen_round_{ledger.iteration}.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )

    screen_data: dict = coder_res.result_data if (coder_res.succeeded and isinstance(coder_res.result_data, dict)) else {}
    base_scores = screen_data.get("baseline_scores") or []

    higher_is_better = ledger.brief.higher_is_better
    winners: list[tuple[CandidateSpec, float]] = []
    losers: list[tuple[CandidateSpec, float | None]] = []

    spec_by_name = {c.name: c for c in candidates}
    if all_finite(base_scores):
        raw_candidates = screen_data.get("candidates") or []
        for c_res in raw_candidates:
            if not isinstance(c_res, dict):
                continue
            c_name = c_res.get("name", "")
            spec = spec_by_name.get(c_name)
            if not spec:
                continue
            c_scores = c_res.get("scores") or []
            if all_finite(c_scores):
                gain = score_improvement(c_scores, base_scores, higher_is_better)
                counted = is_counted_gain(c_scores, base_scores, higher_is_better)
                if counted and gain > 0:
                    winners.append((spec, gain))
                else:
                    losers.append((spec, gain))
            else:
                losers.append((spec, None))
    else:
        # If screening failed, all candidates are dropped
        losers = [(c, None) for c in candidates]

    # 3. If winners exist, create next features version
    new_version = ledger.current_feature_version
    all_features = list(ledger.features)
    changed = False

    if winners:
        new_version = ledger.current_feature_version + 1
        new_features_path = workspace.features_file(new_version)
        winners_text = "\n".join(f"- {w[0].name} ({w[0].family}): {w[0].idea}" for w in winners)

        incorporate_task = (
            f"Add these winning features into a new features module at {new_features_path}:\n"
            f"{winners_text}\n\n"
            f"Base features file to copy and expand: {current_features_path}\n"
            "Requirements:\n"
            "1. Preserve existing `fit_preparation(train_data, seed=42)` and `apply_preparation(data, state)` logic.\n"
            "2. Any stateful statistics for the new features (group means/stds, target encodings, frequency maps) MUST be computed on `train_data` inside `fit_preparation` and stored in `state`, then applied in `apply_preparation`.\n"
            "3. Compute new features from raw input `data` columns or intermediate features before unused raw string/ID/target columns are dropped at the end of `apply_preparation`.\n"
            "4. Ensure `apply_preparation` returns a clean numeric DataFrame with zero `NaN`/`Inf` and consistent column order.\n"
            f"Test on a sample of project data under `if __name__ == '__main__':` and output <{FOUNDATION_RESULT_TAG}>{{\"status\": \"ok\"}}</{FOUNDATION_RESULT_TAG}>."
        )

        inc_res = run_coder(
            task_text=incorporate_task,
            result_tag=FOUNDATION_RESULT_TAG,
            result_keys_text='- "status": "ok"',
            working_folder=workspace.artifacts_folder / "features",
            script_name=f"features_v{new_version}.py",
            caller_key=AGENT_KEY,
            ledger=ledger,
            agent_context=agent_context,
        )
        # Guarantee features_v{new_version}.py exists if Coder wrote to legacy make_features_v{new_version}.py
        v_file = workspace.features_file(new_version)
        legacy_v = workspace.artifacts_folder / "features" / f"make_features_v{new_version}.py"
        if not v_file.is_file() and legacy_v.is_file():
            v_file.write_text(legacy_v.read_text(encoding="utf-8"), encoding="utf-8")
        if inc_res.succeeded:
            changed = True
            for spec, gain in winners:
                all_features.append(
                    FeatureRecord(
                        feature_id=f"feature_{len(all_features) + 1:03d}",
                        name=spec.name,
                        family=spec.family,
                        reason=spec.reason or spec.idea,
                        finding_id=spec.finding_id,
                        score_gain=round(gain, 6),
                        kept=True,
                        feature_version=new_version,
                        iteration=ledger.iteration,
                    )
                )

    for spec, gain in losers:
        all_features.append(
            FeatureRecord(
                feature_id=f"feature_{len(all_features) + 1:03d}",
                name=spec.name,
                family=spec.family,
                reason=spec.reason or spec.idea,
                finding_id=spec.finding_id,
                score_gain=round(gain, 6) if gain is not None else None,
                kept=False,
                feature_version=ledger.current_feature_version,
                iteration=ledger.iteration,
            )
        )

    # CLI summary
    if changed:
        kept_summaries = [f"{w[0].name} +{w[1]:.4f}" for w in winners]
        cli_line = (
            f"Features round {ledger.iteration}: tried {len(candidates)} ideas, "
            f"kept {len(winners)} ({', '.join(kept_summaries)}), dropped {len(losers)}."
        )
    else:
        cli_line = (
            f"Features round {ledger.iteration}: tried {len(candidates)} ideas, "
            f"kept 0, dropped {len(candidates)}."
        )

    event_bus.emit(EventType.RESULT, AGENT_NAME, cli_line)
    return {
        "features": all_features,
        "current_feature_version": new_version if changed else ledger.current_feature_version,
        "last_features_changed": changed,
    }