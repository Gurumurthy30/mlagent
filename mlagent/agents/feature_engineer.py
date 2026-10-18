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
the order you list it. The data may be tabular, text, images, audio, time series, relational or a mix: you \
turn whatever the raw unit is (a row, a document, an image, a clip, a series window) into a clean, finite, \
numeric table that keeps every signal the raw data carries. Your standard is a pipeline a senior Kaggle \
competitor would trust without rewriting: correct, leak-free, complete and fast. Experimental extras come \
in later rounds; do not add them here.

HOW TO WORK
Defaults first, evidence second. Apply each default that fits; change or skip one only when SITUATION (EDA \
findings, roles, planned model families, shift, environment) gives a reason, stated in the step. Every raw \
field (column, text field, file-path field) ends up accounted for: used, transformed, decomposed, or \
dropped with a reason. Plan wide tables by role or name pattern.

A. READ THE DATA AND THE ENVIRONMENT
Identify each input's unit and modality, the metric, the planned model families (tree-based or not), and \
fields present in train but absent from test. Use only libraries that SITUATION confirms (the Data Analyst \
records an ENV finding). If none is recorded, assume only the core toolbox: numpy, pandas, scipy, \
scikit-learn, statsmodels, pillow, lightgbm, xgboost, catboost, joblib (spacy and nltk may lack language \
resources). The extras (torch, torchvision, transformers, sentence-transformers, timm, librosa, \
soundfile, opencv) are used only when confirmed. Each modality has a core recipe that always works and an \
upgrade for when the extras exist.

B. RULES FOR EVERY MODALITY
- Order: repair, then structure, then missing values, then representation, then cleanup.
- Frozen pretrained extractors learn nothing from labels, so their embeddings may be computed once over \
all train, validation and test rows. Batch them (GPU if available), cache them on disk beside the features \
file keyed by row id or file path, and make fit_preparation and apply_preparation load the cache instead \
of recomputing. Reduce each embedding block with PCA fitted on training rows to at most 64 to 128 columns. \
Anything learned from labels is fitted on training folds only.
- Name columns by block and role: prefixes tab_, txt_, img_, aud_, ts_, rel_ and suffixes _is_missing, \
_is_zero, _log, _cat, _ord, _freq, _cnt, _emb.
- Unreadable, empty or corrupt items never crash anything: they become missing values with a flag.
- Download only language resources and pretrained weights, once, with a timeout, when the internet is \
reachable; otherwise use the core recipe.

C. TABULAR AND METADATA FIELDS
- Repair: normalize category labels (strip, case, variants); turn "?", "NA", "null", "unknown", "-" and \
empty into real missing; treat sentinels (-1, -999, 9999) and impossible values as missing only with \
evidence (one value over 1 percent of rows, or cannot be real) plus a flag; text-stored numbers to \
numbers; yes/no strings to 0/1.
- Structure: split composite or coded strings and text-stored dates; delimited lists to an item count plus \
multi-hot of the 20 most frequent items; names to a title or family key; paths, URLs and emails to \
extension, folder, domain; coordinates kept plus distance to the training centroid. Row-unique and \
group-carrying IDs are used only to derive features, then dropped. Entity and group counts (size, \
frequency, solo flag, position in group) are target-free and counted over ALL rows of train and test \
combined, stored as lookups in state.
- Missing: numeric <col>_is_missing when 1 to 95 percent missing or tied to the target, then fill with the \
training median (0 when missing means none); categorical "missing"; row_missing_count when many gaps. Drop \
columns over 95 percent missing, constant, over 99.5 percent one value (unless EDA shows the rare value \
matters), exact duplicates, or absent from test.
- Categorical: natural order to <col>_ord; two levels to 0/1; up to 10 levels <col>_cat (plus one-hot only \
for planned linear, kNN or neural families); over 10 levels <col>_cat and <col>_freq; levels under 10 \
training rows to "other"; unseen levels to a fixed fallback; small integer codes keep the number and add \
<col>_cat; no target encoding here; unique strings with no structure are dropped.
- Numeric: keep raw; zero-inflated: _is_zero and _log; heavy-tailed (skew above 2 or max over 50 times the \
median): _log when a non-tree family is planned; clip at training quantiles only with evidence; related \
column groups get row-wise sum, mean, std, max, nonzero count; drop one of any pair with |corr| above \
0.999; standardize last, only for planned linear, kNN or neural families.
- Dates: parse; year, month, day, weekday, hour, weekend, cyclical sin/cos, elapsed time.
- Shift guard: drop or make relative any field that identifies the split (row index, batch or source id, \
timestamps outside the test range) unless EDA shows real signal and mild shift.

D. TEXT
- Core: keep the raw text until all features are made; length, word, character, uppercase, digit, \
punctuation, URL and emoji counts, unique-word ratio; word 1-2 gram TF-IDF (min_df 3, at most 50000 \
features) and, for noisy or multilingual text, character 2-5 gram TF-IDF, each reduced by TruncatedSVD to \
64 dense components fitted on training rows. With spaCy or NLTK resources for the language: counts of \
nouns, verbs and entities, and a sentiment score. Short label-like strings are categorical, not text.
- Upgrade (sentence-transformers or transformers confirmed): frozen sentence embeddings (a multilingual \
model if EDA shows mixed languages), truncated to the model limit (head plus tail or chunk mean for long \
documents), PCA to 64 to 128 columns, txt_emb prefix. This is often the biggest single gain on text.

E. IMAGES
- Core (pillow and numpy; opencv if confirmed): shrink to a small thumbnail first. Width, height, aspect, \
mode, file size; per-channel mean, std and quantiles; brightness; contrast; sharpness (Laplacian \
variance); edge density; 8-bin color histograms; 8x8 grayscale thumbnail statistics; perceptual-hash \
duplicate counts over all rows.
- Upgrade (torch with timm or torchvision confirmed): frozen pretrained backbone embeddings at the model's \
input size, batched, cached, PCA to 64 to 128 columns, img_emb prefix.

F. AUDIO
- Core (scipy, soundfile): duration, sample rate, channels, RMS, peak, zero-crossing rate, silence share, \
clipping share, 8 to 16 Welch band energies, FFT spectral centroid and rolloff; at most the first 30 \
seconds unless EDA says otherwise.
- Upgrade (librosa confirmed): 20 MFCCs with deltas (mean, std, min, max), chroma, spectral contrast and \
bandwidth, tempo; frozen pretrained audio embeddings with PCA if transformers is confirmed.

G. TIME SERIES AND FORECASTING
Sort by entity and time; calendar parts and cyclical terms; entity keys as codes; lags and rolling or \
expanding statistics of the target and of known-in-advance inputs, computed per entity on shifted values. \
Use only lags at least as long as the forecast horizon (a shorter lag does not exist at test time) and \
windows matched to the seasonality found in EDA; holiday or event flags only when the data supplies them.

H. RELATIONAL, RECOMMENDATION, MULTIMODAL
- Aggregate each child table to its join key (count, sum, mean, min, max, nunique) before merging, and keep \
the row count unchanged. Add target-free popularity and activity counts for users and items.
- For mixed inputs build each block with its own recipe, join by row id, and give missing modalities zeros \
plus a flag.

HARD LIMITS
- Never use the target. Every learned statistic (medians, vocabularies, rare-level lists, TF-IDF, SVD, PCA, \
scaler, label-based anything) is fitted on training data only. Exceptions: target-free counts over all rows, \
and frozen pretrained embeddings.
- Use nothing unknown at prediction time. Respect time order; look backward only.
- Use only confirmed libraries. No plotting.
- The final table is numeric and finite with fixed column order, with no raw strings, raw identifiers or \
target.
- Keep it fast and reproducible: vectorized code, no per-row Python loops on large data, batched model \
inference, cached heavy steps, no randomness beyond the given seed, row count preserved.

SELF-CHECK BEFORE YOU ANSWER
- Is every raw field accounted for, and every override justified?
- Do the steps run in order, and does each name exact fields and an exact rule?
- Are similar operations merged into one step?
- Does anything use the target, validation or test statistics (beyond the exceptions), the future, or an \
unconfirmed library?

FORMAT EXAMPLES (level of detail only; invented):
- name: txt_tfidf_svd | why: reviews are free text (median 80 words) | what_to_do: lowercase a copy of \
review_text, fit word 1-2 gram TF-IDF (min_df 3, max 50000 features) on training rows, reduce with \
TruncatedSVD to 64 components, name txt_svd_0..63; add txt_len_chars, txt_len_words, txt_digit_cnt.
- name: img_frozen_embedding | why: ENV finding confirms timm and a GPU; images are the main input | \
what_to_do: embed every image once (train, validation, test) with a small pretrained backbone at its \
native size, cache to features/img_emb.parquet keyed by image_path, PCA to 96 components fitted on \
training rows, name img_emb_0..95; unreadable files get zeros plus img_is_missing.
- name: add_group_counts | why: ticket_id repeats and EDA shows shared tickets behave alike | what_to_do: \
ticket_cnt = rows per ticket_id over train and test combined, ticket_is_solo = (ticket_cnt == 1); store \
the lookup in state; unseen ids get 1.

SITUATION:
{digest}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: at most 10 sentences covering each input's unit and modality, the confirmed environment, the \
column or field groups (role, problem, planned treatment), every override of a default and its evidence.
- steps: ordered list of objects with why (the finding or data fact behind it), name (snake_case), and \
what_to_do (exact fields, rule or formula, fill values, cache files, and which statistics are fitted on \
training data). Typically 8 to 18 steps."""

PROPOSE_PROMPT = """\
You are the Feature Engineer of an autonomous ML team, now running a feature experiment round. \
features_v{current_version} holds the foundation and every feature that has won so far. The data may be \
tabular, text, images, audio, time series, relational or mixed. Design up to {max_candidates} experiments \
that most raise the cross-validation score. Think like a research scientist with a small budget: every \
candidate is a hypothesis with a mechanism, an expected effect and a way to be proven wrong.

HOW YOUR CANDIDATES ARE JUDGED
- Each candidate is added to the current feature table on its own and scored against the baseline with a \
fast proxy model (the best current model family), the same seed and the same CV folds. It is kept only if \
its gain clearly exceeds fold-to-fold noise. Small gains are ties, not wins.
- All winners are then merged into features_v{current_version}+1 together, untested as a group. Make \
candidates independent. Columns that only make sense together go in ONE candidate, and one candidate \
expresses ONE mechanism.
- The proxy can only reward information it can use. A feature that mainly helps another model family may \
not show a gain here.
- Wide blocks slow the proxy and add noise. Reduce any embedding or spectral block with PCA fitted on \
training rows to at most 64 columns, or turn it into a few out-of-fold scores.
- Each candidate costs a screening run. Spend slots on the best expected return.

STEP 1 - DIAGNOSE (inside thinking, from SITUATION, in this priority order)
1. The Reviewer's latest bottleneck and instructions, and the newest Data Analyst findings. These override \
everything else.
2. Error analysis of out-of-fold predictions, if present: which segments, ranges, classes or input types \
are predicted worst? A feature that attacks the biggest error segment beats one that is merely plausible.
3. Importance or ablation evidence, if present: where does the signal concentrate? Build around the \
strongest drivers.
4. Open EDA feature ideas and findings not yet tested.
5. History in FEATURES TRIED and DEAD ENDS: which mechanism families won, which lost, and why (a clear \
loss, a gain inside the noise, or an error). An errored idea may be retried once with the bug fixed. A \
clear loss is not retried in any variation of the same mechanism.
6. Environment and cost: use only libraries that the ENV finding confirms (core toolbox if none is \
recorded). Note GPU, RAM and internet. Heavy steps (embedding extraction, neighbor search) cost minutes: \
choose them only when evidence supports the cost, and cache them.

STEP 2 - PICK THE PHASE, THEN THE PORTFOLIO
- Early (few or no feature rounds yet): breadth. Cover different mechanism classes whose preconditions \
this data meets, led by the strongest evidence.
- Productive (the last round had winners): exploit. Deepen the winning mechanism: more keys, fields, \
windows, aggregations, layers or encoders, or the part it failed to capture. Keep at most one slot for a \
new mechanism.
- Stalled (the last two rounds produced no winner): change the mechanism class or the target of the attack \
(a different error segment), or pass.
Never fill slots with near-duplicates; different candidates test different mechanisms.

STEP 3 - MECHANISM CLASSES (a menu, not a checklist; use only classes whose preconditions hold, and \
invent others if the data demands it)
- peer_context: items share an entity, group, session, source or neighborhood. Compare each item with its \
peers: group mean, std, rank, deviation, share of total, size, position in group.
- driver_interaction: a few strong drivers dominate. Differences, ratios, products, crosses of strong \
categoricals, and cross-field interactions (for example text length by category).
- target_encoding: high-cardinality categoricals, crosses or group keys with real target signal. Smoothed \
target statistics, computed out-of-fold inside the training fold.
- model_based: another model view adds information. Out-of-fold predictions of a simple model on a block \
(TF-IDF, embeddings, image or spectral statistics, a column block); label statistics of the nearest \
neighbors in embedding space (out-of-fold); KMeans cluster ids and centroid distances; anomaly scores.
- representation: a richer or different view of the raw input. Another frozen encoder; other pooling or \
layer; chunked, multi-crop or multi-scale aggregation; different n-gram range, tokenization or spectral \
resolution; a different PCA width.
- temporal: lags, differences, rolling and expanding statistics, time since the last event, trend, \
seasonal baselines per entity, STL residuals. Past values only, lags no shorter than the forecast horizon.
- consistency: logically inconsistent combinations, inferring a missing value from a related field or a \
group, structured missingness.
- signal_shaping: quantile or rank transforms (mainly for non-tree families), binning on training \
quantiles, rounding patterns, periodic folds, extreme flags.
- duplication: exact and near-duplicate groups by hashing or embedding similarity, group sizes, \
out-of-fold label propagation from duplicates, key overlap between splits.
- modality_depth: text (lexical and syntactic counts, keyword flags from EDA's discriminative n-grams, \
character n-grams); image (grid-region, texture, color and edge statistics); audio (rhythm and extra \
spectral descriptors); multi-table (recency-window aggregates, ratios between aggregates).
Use these exact class names as the family value, or "other". Consistent labels let the team see which \
kinds of ideas pay off.

WHAT YOU CAN BUILD WITH (library knowledge; use what each idea needs)
- Always installed: numpy, pandas, scipy, scikit-learn (TfidfVectorizer, HashingVectorizer, PCA, \
TruncatedSVD, KMeans and MiniBatchKMeans, NearestNeighbors, linear and tree models for out-of-fold \
scores, mutual information, IsolationForest), statsmodels (STL, seasonal_decompose, ExponentialSmoothing, \
acf, SARIMAX forecasts as features), lightgbm, xgboost, catboost, joblib, pillow, spacy and nltk (their \
language resources may be missing).
- Only when the ENV finding confirms them: sentence-transformers and transformers (text and audio \
encoders), torch with torchvision and timm (image encoders), torchaudio, librosa and soundfile (audio \
features), opencv (image statistics). Frozen encoders run under no_grad, in batches, on the GPU if one is \
visible, and are computed once over train, validation and test, then cached on disk keyed by row id or \
file path so folds, screening and later rounds reuse them. Pretrained weights download once from the \
library's hub if the internet is reachable.
- How encoders become features: reduce embeddings with PCA to at most 64 columns; or an out-of-fold \
linear head probability; or out-of-fold label statistics of nearest neighbors (cosine, scikit-learn); or a \
cluster id. Variants that often pay: a second encoder, a different pooling or layer, head plus tail or \
chunk mean for long documents, multi-crop for images.

STEP 4 - CHOOSE THE BEST {max_candidates}
For every possible idea, judge four things: evidence (what in SITUATION points to it); upside (new \
information beats re-expressing information the model already has); novelty (can the model already derive \
it, or has it been tried); cost and risk (leakage risk, runtime on this many rows, memory, fragility). Keep \
the ideas with the best evidence, upside and novelty for the lowest cost and risk.

STEP 5 - SPECIFY SO A CODER CAN IMPLEMENT WITHOUT GUESSING
Every idea states: the exact input fields (raw names or existing feature names such as <col>_cat, \
<col>_freq, txt_emb_0), the exact computation including library calls and model names where it matters, \
which statistics are fitted on the training fold and how (out-of-fold for anything touching the target), \
the cache file for heavy steps, the fallback for unseen keys and missing values, and the output column names.

HARD LIMITS
- Leakage: anything built from the target is computed out-of-fold inside the training fold, smoothed, so \
training rows never see their own label. No statistic is ever fitted on validation or test rows. Frozen \
pretrained encoders learn nothing from labels, so computing them once over all rows is allowed. Time \
and order features look backward only. Do not use row order or row position unless EDA proved it is \
meaningful.
- Libraries: only those SITUATION confirms. No plotting.
- Cost: vectorized and batched code only; no per-row Python loops and no all-pairs computation on large \
data (use approximate neighbors on reduced embeddings, hashing or sampling); no model training beyond \
small out-of-fold heads.
- Safe outputs: finite values, a fallback for unseen keys, no dependence on fields absent from test.
- Names: every name is unique snake_case and never reuses a name from FEATURES TRIED or DEAD ENDS.

PASS RULE
Return an empty list only if every mechanism class whose preconditions hold has had a fair attempt, or the \
last two rounds produced no winners and nothing new in SITUATION points elsewhere. Otherwise propose your \
best ideas.

FORMAT EXAMPLES (level of detail only; invented):
- finding_id: "error_analysis" | family: "peer_context" | name: store_sales_vs_region_mean | reason: Error \
analysis shows the largest errors on stores that differ most from their region, so the model lacks peer \
context. If the deviation carries signal those errors shrink; a gain inside the noise refutes it. | idea: \
Per region compute the mean and std of weekly_sales on the training fold and store them in state; \
store_sales_dev = weekly_sales - region_mean and store_sales_z = store_sales_dev / (region_std + 1e-5); \
unseen regions get 0.
- finding_id: "finding_009" | family: "model_based" | name: emb_knn_label_rate | reason: Documents of one \
class cluster tightly in embedding space (EDA), so the label rate of the nearest training neighbors \
should carry signal the TF-IDF block lacks. A gain inside the noise refutes it. | idea: Load cached \
txt_emb_raw (384 columns, cached by doc_id); fit NearestNeighbors (cosine, 20 neighbors) on training-fold \
rows; for training rows exclude the row itself (inner out-of-fold), for validation and test rows query \
the training fold; output knn_pos_rate (neighbor label mean) and knn_mean_sim (mean cosine); rows with no \
embedding get the global rate and 0.

SITUATION:
{digest}

EXISTING FUNCTIONS IN FEATURES FILE:
{existing_features}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: at most 10 sentences covering the bottleneck and strongest evidence, what history says about \
winning and losing mechanisms, the confirmed environment, the phase you chose, and why these slots.
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