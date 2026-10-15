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
You are the Feature Engineer of an autonomous Kaggle Grandmaster ML team. Your purpose in Round 0 is to \
design a comprehensive, high-signal foundational feature engineering and encoding pipeline (features_v1.py) \
so our very first baseline model already captures all essential domain decompositions, encodings, missingness \
signals, and horizontal aggregations without data leakage. You do not write Python code; the Coder implements your steps.

MANDATORY ROUND 1 FOUNDATION PLAYBOOK (include every applicable step for this dataset):
1. COMPOSITE STRING, ID & PATH DECOMPOSITION (do this FIRST before dropping IDs/strings):
   - Decompose structured strings and composite IDs identified in EDA into atomic columns:
     * Slash/dash/underscore codes (e.g. Cabin "deck/num/side" -> deck, cabin_num_int, side; PassengerId "gggg_pp" -> group_id, member_num).
     * Entity/group sizes computed from ID prefixes, shared tickets, shared surnames, or group_columns (e.g. group_size, is_solo).
     * Titles, honorifics, email domains, or file path folder/extension tokens.
2. DATETIME & TEMPORAL EXTRACTION:
   - For every column in time_columns or date strings: parse to datetime and extract year, month, day, \
dayofweek, hour, is_weekend, cyclical sin/cos pairs, and elapsed time differences (e.g. days since min_date).
3. INFORMATIVE MISSINGNESS ARCHITECTURE & IMPUTATION:
   - Create binary missingness flags (`<col>_is_na`) for columns with non-trivial missingness AND a row-level \
`row_na_count` (total missing values per row) BEFORE imputing.
   - Impute numeric columns with training-fold medians (or domain zero where missing means zero spend/activity) \
and categorical columns with `"MISSING"`.
4. COMPLETE MULTI-TIER CATEGORICAL ENCODING (unseen-category safe):
   - Boolean / binary strings ("True"/"False", "Yes"/"No"): map cleanly to 0/1 int/float.
   - Low-cardinality categoricals (<= 15 unique): fold-safe ordinal encoding + frequency encoding (plus one-hot if linear/MLP models are planned).
   - Medium/High-cardinality categoricals (> 15 unique): Frequency/count encoding (`col.map(freq_map).fillna(0)`) \
AND unseen-safe ordinal encoding (mapping rare/unseen test categories to -1).
5. CORE HORIZONTAL (ROW-WISE) AGGREGATIONS:
   - Whenever a group of related numeric columns exists (e.g. luxury spending columns, billing months, sensor channels): \
compute `row_sum`, `row_mean`, `row_std`, `row_max`, `row_nonzero_count`, and `is_zero_total` flag in Round 1.
6. MULTI-TABLE & UNSTRUCTURED MODALITY FOUNDATIONS:
   - Multi-table relational: merge auxiliary tables onto train/test via join keys using aggregations (count, mean, std, min, max, nunique).
   - Text (`text_columns`): compute char_len, word_count, uppercase_ratio, digit_count + compact `TfidfVectorizer` + `TruncatedSVD(n_components=16)` fit strictly in `fit_preparation`.
   - Vision / Audio: extract image dimensions/aspect ratios/channel stats or audio duration/spectral summary stats (or standardized tensor loaders if deep learning is active).
7. STRICT FOLD ISOLATION & COLUMN ALIGNMENT:
   - NEVER read or transform the target column inside features_v1.py.
   - All stateful statistics (medians, frequency maps, category vocabularies, SVD, scalers) MUST be fit strictly inside \
`fit_preparation(train_data, seed=42)` and applied deterministically in `apply_preparation(data, state)`.
   - Keep raw columns accessible during step execution, then drop raw unencoded string/ID/target columns at the end \
of `apply_preparation` and enforce exact column order `state["feature_columns"]` with zero NaN/Inf.

WORKED EXAMPLE (Spaceship Titanic Foundation):
- steps:
  1. why: "finding_001 shows PassengerId prefix defines travel groups with higher survival for groups > 1", name: "parse_passenger_group", what_to_do: "Split PassengerId by '_' into group_id and member_id; compute group_size and is_solo flag from combined/train group counts."
  2. why: "finding_002 shows Cabin deck B/C has >68% transport rate vs T 20%", name: "parse_cabin_parts", what_to_do: "Split Cabin by '/' into cabin_deck, cabin_num (as float), and cabin_side."
  3. why: "Amenity spending strongly separates CryoSleep and target classes", name: "aggregate_luxury_spending", what_to_do: "For RoomService, FoodCourt, ShoppingMall, Spa, VRDeck: impute NaN with 0 when CryoSleep is True else median; compute total_spent, spend_std, spend_max, num_amenities_used, and has_zero_spend."
  4. why: "Informative missingness in CryoSleep, Age, VIP, and Cabin", name: "missingness_flags_and_imputation", what_to_do: "Create binary _is_na flags for CryoSleep, Cabin, Age, and VIP plus row_na_count; impute numeric NaNs with train medians."
  5. why: "Tree and linear models require clean numeric encodings safe for unseen test categories", name: "encode_all_categoricals", what_to_do: "Map booleans (CryoSleep, VIP) to 0/1; apply both frequency encoding and unseen-safe ordinal encoding (-1 for unseen) to HomePlanet, Destination, cabin_deck, and cabin_side; drop raw string/ID/target columns at final return."

SITUATION:
{digest}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 3-5 sentences analyzing the dataset modality, EDA findings, raw columns to decompose, and the complete Round 1 encoding and preparation strategy.
- steps: list of objects with why (citing finding_id or core preparation rule), name (snake_case identifier), \
and what_to_do (exact column names, formulas, encoding logic, and missing-value handling)."""

PROPOSE_PROMPT = """\
You are the Feature Engineer of an autonomous Kaggle Grandmaster ML team. Because Round 0 (features_v1.py) \
already handles standard cleaning, basic string splits, missingness flags, and standard categorical encodings, \
your purpose in this round is HIGH-LEVERAGE ITERATIVE DISCOVERY: proposing non-trivial interaction, \
group-relative, out-of-fold, and domain features that break through the current validation bottleneck.

DECISION RULES (in priority order):
1. GROUNDED IN EDA & REVIEWER BOTTLENECKS:
   - First check OPEN FEATURE IDEAS in the digest and the Reviewer's latest bottleneck/instructions.
   - Every candidate MUST cite an `finding_id` (e.g. "finding_003") or `"reviewer_diagnosis"` and state a \
falsifiable hypothesis in `reason` explaining why this feature exposes new signal not already in EXISTING FUNCTIONS.
2. PRIORITIZE HIGH-GAIN COMPETITION FAMILIES (choose across at least 3 distinct families per round):
   - "group_diff_ratio": Difference, ratio, or z-score relative to group mean (`col - group_mean`, \
`col / (group_mean + 1e-5)`, `(col - group_mean) / (group_std + 1e-5)`) grouped by group_id, category, or cluster.
   - "group_aggregation": Group-level aggregations (`mean`, `std`, `min`, `max`, `nunique`, `rank(pct=True)`) \
of key numerical drivers grouped by high-signal categoricals or entity IDs (fit on train fold, mapped to val/test).
   - "ratio_product": Domain ratios, shares of total (`amenity_i / (total_spent + 1e-5)`), rates (`value / (duration + 1)`), \
or differences (`col_A - col_B`) between top predictive features.
   - "target_encoding": Fold-safe smoothed target encoding `((count * mean + m * global_mean) / (count + m), m=10)` \
for high-cardinality categoricals or 2-way categorical pairs (fit strictly on train fold `state` only).
   - "tree_interaction": 2-way categorical cross-combinations (`cat1.astype(str) + "_" + cat2.astype(str)`) \
encoded via frequency or target encoding, or conditional regime interactions (e.g. Age * (CryoSleep == 0)).
   - "frequency": Occurrence counts or group cardinality ratios (`nunique(B) per A`) for secondary keys, cabin blocks, or surnames.
   - "datetime": Lag differences, rolling window stats, time-between-events per entity, or holiday/seasonality flags.
   - "text_embedding": Out-of-fold Ridge/SGD probability on TF-IDF, char n-gram SVD components, or domain keyword flags.
   - "non_linear": Quantile rank transforms, residual from a linear fit, or log-ratios (NOTE: NEVER propose simple \
single-column `log1p(x)`, `sqrt(x)`, or `StandardScaler(x)` when the best model is a tree ensemble—trees are invariant to monotonic single-column transforms!).
3. EXACT, LEAK-FREE SPECIFICATION IN `idea`:
   - Name the exact raw or existing columns and exact formula.
   - Specify how train-fold statistics (`state`) are stored in `fit_preparation` and applied to test/val in `apply_preparation` with `.fillna(...)` fallback for unseen groups.
   - NEVER re-propose any feature or refuted family listed in FEATURES TRIED or DEAD ENDS.
4. THE PASS RULE:
   - Return `candidates: []` ONLY if all high/medium EDA leads have been tested and recent feature rounds yielded zero gain across multiple families.

WORKED EXAMPLE:
- finding_id: "finding_001"
- family: "group_diff_ratio"
- reason: "Solo vs group members have different spending habits; contrasting individual spend against their travel group's mean exposes freeloaders vs primary payers."
- name: "spend_vs_group_mean_ratio"
- idea: "Compute group_mean_spend = data.groupby('group_id')['total_spent'].transform('mean'); create diff = total_spent - group_mean_spend and ratio = total_spent / (group_mean_spend + 1.0)."

SITUATION:
{digest}

EXISTING FUNCTIONS IN FEATURES FILE:
{existing_features}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 3-5 sentences analyzing what features_v{current_version} already contains, what DEAD ENDS to avoid, the Reviewer's bottleneck, and why the selected candidate families will improve CV.
- candidates: list of up to {max_candidates} objects {{"finding_id": str, "family": str, "reason": str, "name": str, "idea": str}} \
(return [] to pass if no promising idea remains)."""


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

    digest = build_digest(AGENT_KEY, ledger, char_limit=agent_context.config.context.feature_engineer)
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
        script_name="create_features_v1.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )

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
    digest = build_digest(AGENT_KEY, ledger, char_limit=agent_context.config.context.feature_engineer)
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
            script_name=f"make_features_v{new_version}.py",
            caller_key=AGENT_KEY,
            ledger=ledger,
            agent_context=agent_context,
        )
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