"""The Feature Engineer: builds the input preparation and tests which new steps help.

Round 0 turns the EDA into a to-do list (the foundation plan) and writes features_v1.py, the
cleaning and making the data model-ready.
Every features round after that proposes a batch of candidates (or passes when it has no good
idea), checks and screens them on the fixed validation splits, keeps those with a counted
gain, and writes the next features_vN.py.

For tables these are features. For images, text and audio they are preprocessing,
augmentation or embedding steps. The agent does not care which.
"""

import ast
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from mlagent.agents.base import (
    CHARS_PER_TOKEN,
    MIN_DIGEST_CHARS,
    PROMPT_TEXT_ALLOWANCE_CHARS,
    AgentContext,
    AgentFailedError,
    build_common_prompt_values,
    display_name,
    render_prompt,
)
from mlagent.agents.coder import last_error_line, run_coder
from mlagent.digest import (
    describe_brief,
    describe_eda,
    describe_plan,
    describe_reviews,
    describe_run,
    describe_runs,
    describe_task,
    join_items,
    trim_text,
)
from mlagent.events import EventType
from mlagent.llm_client import LlmUnavailableError
from mlagent.scoring import all_finite, is_counted_gain, score_improvement
from mlagent.script_contract import SPLITS_FORMAT_TEXT
from mlagent.settings import (
    BIG_FILE_MB,
    MAX_CANDIDATE_CODE_CHARS,
    MAX_CANDIDATES,
    MAX_FEATURE_SOURCE_CHARS,
    RANDOM_SEED,
    FOUNDATION_PLAN_ENABLED,
    PROMPT_MIN_CANDIDATES,
    PROMPT_MAX_CANDIDATES,
    SCREEN_TIME_FRACTION,
    NEW_VERSION_SOURCE_CHARS,
    FOUNDATION_DIGEST_CHARS,
    FOUNDATION_MAX_STEPS,
    FEATURE_REASON_CHARS,
    ANALYSIS_EVENT_CHARS,
    RECORD_NOTE_CHARS,
    BASIC_PREPARATION_NAME,
    FEATURE_STEPS_NAME,
    SKIPPED_FOR_TIME_PREFIX,
    CANDIDATE_FILE_HEADER
)
from mlagent.state import FeatureRecord, Ledger

AGENT_KEY = "feature_engineer"
AGENT_NAME = display_name(AGENT_KEY)
FEATURES_RESULT_TAG = "FEATURES"
SCREEN_RESULT_TAG = "SCREEN"
SCREENING_FOLDER_NAME = "screening"


FEATURE_FAMILIES = (
    "ratio",
    "interaction",
    "aggregation",
    "frequency",
    "string_parse",
    "datetime",
    "transform",
    "missing_flag",
    "text_stat",
    "embedding",
    "augmentation",
    "resize_normalize",
    "lag_rolling",
    "other",
)
FAMILY_PREFIX_PATTERN = re.compile(r"^\[([a-z_]+)\] ")

FEATURES_RESULT_KEYS_TEXT = """\
- "module_path": the path of the features file (this script)
- "functions": list of the function names other scripts can import
- "steps": list of the names in FEATURE_STEPS (an empty list when there are none)
- "n_output_columns": number of columns apply_preparation returned on the check sample
- "handled": list of objects {"step": the number of the to-do step or the name of the \
candidate, "status": "done" or "skipped", "note": short text}. An empty list when there was no \
to-do list.
- "description": one short sentence about what the file does"""

SCREEN_RESULT_KEYS_TEXT = """\
- "baseline_scores": list of numbers, the primary metric of the current setup on each \
validation split
- "candidates": list of objects {"name": candidate name, "scores": list of numbers in split \
order, or null when it failed, "error": short text or "", "seconds": number}
- "proxy": one short text: the model, its settings and the sample size used"""

FEATURES_FILE_RULES_TEXT = """\
Your script IS the features file. It defines the functions other scripts import. Under \
if __name__ == "__main__": it runs a quick check on a small sample of the real data and \
prints the result line. Importing the file must have no side effects: no printing, no file \
writing, no training.
Interface that every later script relies on. Keep these two names for every kind of data:
- fit_preparation(train_data, seed=42) returns a state. It learns everything that depends on \
data (means, counts, category maps, vocabularies, scalers, fitted embeddings) from the \
training examples it receives, and from nothing else.
- apply_preparation(data, state) returns the model inputs. It works on any examples, training \
or new. For tables it returns a DataFrame of numeric columns with one row per input row, in the \
same order.
Layout for tables, so later versions can add ideas without touching the rest: a list \
FEATURE_STEPS of tuples (name, fit_function, apply_function). fit_function(train_table) returns \
a state, or None when nothing is learned. apply_function(table, state) returns a DataFrame of \
new numeric columns only, with the same index as table. fit_preparation fits the base \
preparation and every step. apply_preparation joins the base columns with the columns of every \
step. Name every new column with the step name as its start, so columns never clash. \
FEATURE_STEPS may be empty.
Rules for every function:
- Never use the target or labels in any function.
- apply_preparation must return only numeric feature columns for the model: never the target, \
and never unencoded strings or text.
- Dropped columns: exclude them completely; do not create steps that return the remaining table.
- Never drop, filter or reorder rows.
- Handle missing values, unseen categories, empty inputs and corrupt files without crashing.
- Be deterministic: take the seed where randomness is used.
- Be efficient: work on whole columns or batches, never row by row on big data.
- Keep names and arguments stable. Other scripts depend on them.
Start the file with a docstring that says what each function takes and returns."""

FOUNDATION_FIXED_RULES_TEXT = """\
Version 1 is cleaning and making the data model-ready. It does not try clever new inputs, those \
come in later rounds.
- Never use as an input: the id column, the target, and every column named in the leakage \
risks, unless the to-do list says it is safe.
- Remove inputs that are empty or constant in the training data, and inputs that exist in train \
but not in test, because they cannot be used at prediction time.
- Missing values: learn the fill value on the training examples in fit. Keep a missing \
indicator column only when the to-do list says missingness matters.
- Categories: an unseen category must not crash. Map it to an explicit unknown code.
- Strings that hold numbers or dates are parsed. Placeholder values named in the EDA, for \
example -999, become missing.
- Do not scale or normalize tables. The model script does that when its model needs it.
- The returned DataFrame of apply_preparation must contain only numeric feature columns. \
It must never include the target or raw string columns.
- Images, text and audio: loading, fixed size or length, normalization, and a safe fallback for \
corrupt files."""

SCREENING_RULES_TEXT = """\
- Baseline: fit_preparation and apply_preparation of the current file, one fast model, scored \
on every validation split. Report the primary metric per split as baseline_scores.
- Candidate: the same pipeline plus this one candidate. For a table candidate, \
fit_<name>(train_table) learns its state on the training part of the split only, \
apply_<name>(table, state) returns new columns, and they are joined to the base columns of \
the training part and of the validation part. For other data follow the style of the current \
file. Score it the same way.
- Use exactly the same model, settings and seed for the baseline and for every candidate, so a \
difference comes from the candidate only.
- Model: the family of the best run so far, in a small fast setting (for example 100 to 200 \
trees, or a few epochs on a subsample). If that family is slow or unknown, use a small \
gradient boosting model for tables. Pass only numeric columns to the model (never the target \
or raw strings).
- Sample: at most 50000 training rows per split. State the sample size in proxy.
- Time: score the candidates in the given order. Do not start a new candidate after 70 percent \
of the time budget is used. Report the rest with scores null and error "skipped: time".
- One failing candidate (an error, a wrong number of rows, NaN or infinite values) never stops \
the script. Record scores null and the error."""

FOUNDATION_PROMPT = """\
You are the Feature Engineer of an ML team. Version 1 of the input preparation is the \
foundation under every later experiment: clean, safe and model-ready, and nothing clever. You \
turn what the EDA found into an ordered to-do list for it. A Coder will implement the list, so \
every step must be concrete. You do not write code.

{run_rules}

WHAT THE TEAM KNOWS:
{digest}

Start with analysis, 3 to 5 sentences. This is your thinking, so be honest in it:
- which inputs are unsafe or useless: the id column, columns that may leak the answer, empty \
or constant columns, columns that exist only in train
- what is broken: missing values, wrong types, strange placeholder values, corrupt files
- what the model cannot read yet: text categories, dates, raw files that need a fixed size or \
length

Then write 6 to 14 steps, in this order: first what is removed from the inputs, then repairs, \
then conversions.
- scope: the column, file or group the step is about, exactly as the facts name it. If the \
facts do not name the columns, state a rule the Coder can apply (for example "columns that are \
empty in train").
- action: one of drop_input, fix_missing, add_missing_flag, parse_or_cast, \
fix_sentinel_or_outlier, encode_category, split_datetime, resize_or_normalize, clean_text, \
fix_corrupt, other
- reason: one sentence with the evidence, a number or a finding id
- finding_id: the EDA finding behind the step, or an empty string

Rules:
- Every data issue and every leakage risk in the facts gets a step, or is named in analysis \
with the reason why nothing is done.
- Cleaning only. Ratios, interactions, aggregations and other new inputs are for later rounds.
- Never invent a column or a number that is not in the facts.
- Rows are never dropped here. Rows with a missing label are left out when the splits are \
made, so do not write a step for them.
Return JSON: analysis, steps (list of objects with scope, action, reason, finding_id)."""

PROPOSE_PROMPT = """\
You are the Feature Engineer of an ML team. On most real ML problems the inputs decide the \
score more than the model does, so your job is the most important one after the data \
analysis. You work like a strong practitioner: you plan a batch of diverse ideas, the team \
screens all of them in one run on the fixed validation splits, and only ideas with a counted \
gain are kept. For tables the ideas are new columns. For images, text, audio and other data \
they are preprocessing, augmentation or embedding steps. The basic cleaning is already done. \
You add ideas on top of it.

{run_rules}

WHAT THE TEAM KNOWS:
{digest}

CURRENT FEATURES FILE (version {feature_version}), the interface you must follow:
{features_source}

Start with analysis, 4 to 6 sentences. This is your thinking, so be honest in it:
1. What do the Reviewer's instructions and the open feature ideas ask for? They come first. \
Name their finding ids.
2. Read FEATURES TRIED and FEATURE FAMILIES. A family with kept ideas shows a direction that \
works: go deeper in a clearly different form. A family with several dropped ideas is a dead \
end: skip it. Never repeat an idea and never submit a renamed copy.
3. What could the inputs still be missing, judging by the EDA and the results so far: skew, \
missing patterns, string or id structure, groups, time, interactions, noise, imbalance, \
train versus test differences?
4. What would an experienced practitioner try on this kind of data that nobody has tried yet?
5. Decide: try or pass. Pass by returning an empty list when you cannot name an idea with a \
realistic gain that differs from everything tried, for example when the last feature rounds \
kept nothing and the open ideas are minor. Then say why in the last sentence of the analysis. \
A pass costs no compute, and it gives the round back to the Reviewer.

Then write {min_candidates} to {max_candidates} candidates, fewer when you have fewer good \
ideas. Quality matters more than count. Never pad the list. Fewer when each training run is \
slow, for example deep learning.
- Spread the batch across families so one test covers a lot of ground. Tag each with one \
family: ratio, interaction, aggregation, frequency, string_parse, datetime, transform, \
missing_flag, text_stat, embedding, augmentation, resize_normalize, lag_rolling, other. \
Examples by data type:
  - tables: ratios and differences, interactions, group aggregations and frequency counts \
(without the target), parsing of strings and ids, date and time parts, log, rank or binning of \
skewed columns, missing indicators
  - images: resizing, crops, color or normalization changes, augmentation, pretrained \
embeddings
  - text: cleaning, token or n-gram variants, length and style statistics, embeddings
  - audio: spectrogram settings, trimming or padding, augmentation, pretrained embeddings
  - time series: lags, rolling statistics, calendar parts, differences
- Mix safe, standard ideas with a few bolder ones. Put the highest expected gain first.
- Every candidate works alone. Screening adds each one separately to the current setup, so \
never build one candidate on top of another.
- For tables a candidate is two functions: fit_<name>(train_table) returns a state, or None \
when nothing is learned, and apply_<name>(table, state) returns a DataFrame of new numeric \
columns whose names start with the candidate name, with the same index as table. For other data \
follow the style of the current file.
- The code is complete and runnable, under {max_code_chars} characters, with no placeholders \
and no pseudo code. numpy as np and pandas as pd are already imported. It never uses the \
target or labels. Anything learned from data is learned in fit from the training examples it \
receives.
- Keep the cost low: a candidate must not make the screening much slower than the current \
setup.
- finding_id: the exact EDA finding id behind the idea (for example finding_003), or an empty \
string for a hypothesis of your own.
- reason: one sentence "I expect ... because ...", naming the signal you expect.
- name: lowercase snake_case and specific (ratio_income_to_age, not feature_1).

Example of the contract only, for a table (do not copy its content):
finding_id: finding_004
reason: I expect a gain because spend per month of tenure separates new heavy users from old \
light users, and finding_004 shows churn is highest in the first 6 months.
family: ratio
name: spend_per_tenure_month
code:
def fit_spend_per_tenure_month(train_table):
    return None

def apply_spend_per_tenure_month(table, state):
    ratio = table["monthly_charges"] / (table["tenure"] + 1)
    return ratio.rename("spend_per_tenure_month").to_frame()
Return JSON: analysis, candidates (list of objects with finding_id, reason, family, name, \
code). An empty list is a valid answer."""


def require_all_fields(schema: dict) -> None:
    """Send every field as required in the JSON schema, in field order.

    Validation keeps its defaults, so a missing field never fails an answer. The schema makes
    the model write every field, and in this order (the thinking field comes first).
    """
    schema["required"] = list(schema.get("properties", {}))


class Candidate(BaseModel):
    model_config = ConfigDict(json_schema_extra=require_all_fields)

    finding_id: str = ""
    reason: str = ""
    family: str = "other"
    name: str
    code: str


class CandidateList(BaseModel):
    model_config = ConfigDict(json_schema_extra=require_all_fields)

    analysis: str = ""
    candidates: list[Candidate]


class FoundationStep(BaseModel):
    model_config = ConfigDict(json_schema_extra=require_all_fields)

    scope: str
    action: str
    reason: str
    finding_id: str = ""


class FoundationPlan(BaseModel):
    model_config = ConfigDict(json_schema_extra=require_all_fields)

    analysis: str = ""
    steps: list[FoundationStep]


class ScreenedCandidate(BaseModel):
    name: str
    scores: list[float] | None = None
    error: str = ""
    seconds: float | None = None


class ScreenResult(BaseModel):
    baseline_scores: list[float]
    candidates: list[ScreenedCandidate]
    proxy: str = ""


class FeatureStepResult(BaseModel):
    features: list[FeatureRecord]
    current_feature_version: int
    changed: bool


def normalize_candidate_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")


def normalize_family(family: str) -> str:
    cleaned_family = family.strip().lower()
    return cleaned_family if cleaned_family in FEATURE_FAMILIES else "other"


def clean_candidates(candidates: list[Candidate], tried_names: set[str]) -> list[Candidate]:
    """Drop empty, repeated, already tried and over-long candidates. Keep at most the maximum."""
    cleaned_candidates: list[Candidate] = []
    seen_names = set(tried_names)
    for candidate in candidates:
        name = normalize_candidate_name(candidate.name)
        code = candidate.code.strip()
        if not name or not code or name in seen_names or len(code) > MAX_CANDIDATE_CODE_CHARS:
            continue
        seen_names.add(name)
        cleaned_candidates.append(
            candidate.model_copy(
                update={"name": name, "code": code, "family": normalize_family(candidate.family)}
            )
        )
    return cleaned_candidates[:MAX_CANDIDATES]


def read_features_source(agent_context: AgentContext, feature_version: int) -> str:
    features_path = agent_context.workspace.features_file(feature_version)
    return trim_text(features_path.read_text(encoding="utf-8"), MAX_FEATURE_SOURCE_CHARS)


def read_full_features_source(agent_context: AgentContext, feature_version: int) -> str:
    features_path = agent_context.workspace.features_file(feature_version)
    return features_path.read_text(encoding="utf-8")


def read_feature_step_names(tree: ast.Module) -> list[str]:
    step_names: list[str] = []
    for node in tree.body:
        is_steps_assignment = isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == FEATURE_STEPS_NAME
            for target in node.targets
        )
        if not is_steps_assignment or not isinstance(node.value, ast.List):
            continue
        for element in node.value.elts:
            is_tuple_with_items = isinstance(element, ast.Tuple) and bool(element.elts)
            first_item = element.elts[0] if is_tuple_with_items else None
            if isinstance(first_item, ast.Constant) and isinstance(first_item.value, str):
                step_names.append(first_item.value)
    return step_names


def summarize_features_source(source: str) -> str:
    """The interface of a features file: its docstring, function signatures and step names."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return trim_text(source, MAX_FEATURE_SOURCE_CHARS)
    lines = [ast.get_docstring(tree) or ""]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            arguments = ", ".join(argument.arg for argument in node.args.args)
            first_doc_line = (ast.get_docstring(node) or "").split("\n")[0]
            lines.append(f"def {node.name}({arguments})  {first_doc_line}".rstrip())
    step_names = read_feature_step_names(tree)
    if step_names:
        lines.append(f"{FEATURE_STEPS_NAME}: {', '.join(step_names)}")
    summary = "\n".join(line for line in lines if line)
    return trim_text(summary, MAX_FEATURE_SOURCE_CHARS)


def describe_candidates(candidates: list[Candidate]) -> str:
    return "\n\n".join(f"### {candidate.name}\n{candidate.code}" for candidate in candidates)


def describe_candidate_list(candidates: list[Candidate]) -> str:
    return "\n".join(
        f"- {candidate.name} ({candidate.family}): {candidate.reason}" for candidate in candidates
    )


def family_of_feature(feature: FeatureRecord) -> str:
    match = FAMILY_PREFIX_PATTERN.match(feature.reason)
    return match.group(1) if match else "other"


def reason_without_family(feature: FeatureRecord) -> str:
    return FAMILY_PREFIX_PATTERN.sub("", feature.reason)


def describe_feature_memory(ledger: Ledger) -> str:
    """Features tried with a short reason and family, plus how many of each family were kept."""
    tried_features = [f for f in ledger.features if f.name != BASIC_PREPARATION_NAME]
    if not tried_features:
        return ""
    lines = [
        f"FEATURES TRIED (current features file is version {ledger.current_feature_version}):"
    ]
    for feature in tried_features[-20:]:
        gain_text = "gain n/a" if feature.score_gain is None else f"gain {feature.score_gain:+.4f}"
        kept_text = "kept" if feature.kept else "dropped"
        reason_text = trim_text(reason_without_family(feature), FEATURE_REASON_CHARS)
        lines.append(
            f"- {feature.feature_id} {feature.name} ({family_of_feature(feature)}): "
            f"{kept_text}, {gain_text}. {reason_text}"
        )
    tried_by_family: dict[str, int] = {}
    kept_by_family: dict[str, int] = {}
    for feature in tried_features:
        family = family_of_feature(feature)
        tried_by_family[family] = tried_by_family.get(family, 0) + 1
        kept_by_family[family] = kept_by_family.get(family, 0) + (1 if feature.kept else 0)
    lines.append("FEATURE FAMILIES (kept of tried):")
    for family in sorted(tried_by_family, key=lambda name: -kept_by_family[name]):
        lines.append(f"- {family}: {kept_by_family[family]} of {tried_by_family[family]} kept")
    return "\n".join(lines)


def describe_open_feature_ideas(ledger: Ledger) -> str:
    tried_finding_ids = {feature.finding_id for feature in ledger.features if feature.finding_id}
    open_ideas = [
        f"{idea.description} ({idea.finding_id or 'no finding'})"
        for idea in ledger.eda.feature_ideas
        if not idea.finding_id or idea.finding_id not in tried_finding_ids
    ]
    return join_items("FEATURE IDEAS WITH NO TRIED FEATURE YET", open_ideas)


def build_propose_digest(ledger: Ledger) -> str:
    """The digest of a features round. Every section has its own budget, so none is cut away."""
    sections = [
        (describe_task(ledger), 500),
        (describe_plan(ledger), 1800),
        (describe_brief(ledger), 1600),
        (describe_reviews(ledger), 1800),
        (describe_feature_memory(ledger), 2600),
        (describe_open_feature_ideas(ledger), 1200),
        (describe_eda(ledger), 3500),
        (describe_runs(ledger), 1800),
    ]
    return "\n\n".join(trim_text(text, limit) for text, limit in sections if text)


def build_foundation_digest(ledger: Ledger) -> str:
    sections = [
        describe_task(ledger),
        describe_brief(ledger),
        describe_eda(ledger),
        describe_plan(ledger),
    ]
    digest = "\n\n".join(text for text in sections if text)
    return trim_text(digest, FOUNDATION_DIGEST_CHARS)


def build_feature_prompt_values(
    ledger: Ledger, agent_context: AgentContext, digest_text: str
) -> dict[str, str]:
    """The common prompt values, with this agent's own digest instead of the standard one."""
    common_values = build_common_prompt_values(AGENT_KEY, ledger, agent_context)
    half_of_context_chars = agent_context.config.model.context_length * CHARS_PER_TOKEN // 2
    room_for_digest = half_of_context_chars - len(common_values["run_rules"])
    room_for_digest -= PROMPT_TEXT_ALLOWANCE_CHARS
    fitted_digest = trim_text(digest_text, max(room_for_digest, MIN_DIGEST_CHARS))
    return {**common_values, "digest": fitted_digest or "(nothing in memory yet)"}


def last_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return sentences[-1] if sentences else ""


def candidates_file_path(ledger: Ledger, agent_context: AgentContext) -> Path:
    screening_folder = agent_context.workspace.features_folder / SCREENING_FOLDER_NAME
    return screening_folder / f"screen_candidates_round_{ledger.iteration}.py"


def find_candidate_problem(candidate: Candidate, require_fit_and_apply: bool) -> str:
    """Return why a candidate cannot be used, or an empty string when it passes the gate."""
    try:
        tree = ast.parse(candidate.code)
    except SyntaxError as error:
        return f"syntax error: {error.msg}"
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if not function_names:
        return "no function defined"
    needed_names = {f"fit_{candidate.name}", f"apply_{candidate.name}"}
    if require_fit_and_apply and not needed_names <= function_names:
        return "missing fit_<name> and apply_<name>"
    return ""


def write_candidates_file(
    candidates: list[Candidate], path: Path, require_fit_and_apply: bool
) -> tuple[list[Candidate], list[tuple[Candidate, str]]]:
    """Check every candidate, then write the accepted ones into one importable module.

    One broken candidate would otherwise break the import for all of them.
    """
    accepted_candidates: list[Candidate] = []
    rejected_candidates: list[tuple[Candidate, str]] = []
    for candidate in candidates:
        problem = find_candidate_problem(candidate, require_fit_and_apply)
        if problem:
            rejected_candidates.append((candidate, problem))
        else:
            accepted_candidates.append(candidate)
    body = "\n\n".join(candidate.code.strip() + "\n" for candidate in accepted_candidates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CANDIDATE_FILE_HEADER + body, encoding="utf-8")
    return accepted_candidates, rejected_candidates


def build_basic_preparation_task(
    ledger: Ledger,
    agent_context: AgentContext,
    foundation_plan: FoundationPlan | None = None,
) -> str:
    workspace = agent_context.workspace
    features_path = workspace.features_file(1)
    steps = foundation_plan.steps if foundation_plan is not None else []
    step_lines = [
        f"{number}. [{step.action}] {step.scope}: {step.reason}"
        for number, step in enumerate(steps, start=1)
    ]
    todo_text = "\n".join(step_lines) or "none: use the usual first step for this data"
    eda_text = "\n".join(
        line
        for line in (
            join_items("Data issues", ledger.eda.issues),
            join_items("Leakage risks", ledger.eda.leakage_risks),
        )
        if line
    )
    return (
        f"Write the file {features_path}: version 1 of the input preparation, the basic "
        "preparation that makes this project's data model-ready.\n\n"
        "TO-DO LIST (do every step in order. If a step cannot be done, say why in handled):\n"
        f"{todo_text}\n\n"
        f"{FOUNDATION_FIXED_RULES_TEXT}\n\n"
        f"{FEATURES_FILE_RULES_TEXT}\n\n"
        f"{describe_brief(ledger)}\n\n"
        f"{eda_text}\n\n"
        f"Project folder (read only): {workspace.project_folder}\n"
        f"Never fully load a file larger than {BIG_FILE_MB} MB; use a sample for the check.\n"
        f"Playbook for this task: {ledger.plan.playbook}"
    )


def build_screening_task(
    candidates: list[Candidate], ledger: Ledger, agent_context: AgentContext
) -> str:
    workspace = agent_context.workspace
    direction = "higher" if ledger.plan.higher_is_better else "lower"
    current_file = workspace.features_file(ledger.current_feature_version)
    candidates_file = candidates_file_path(ledger, agent_context)
    time_limit = agent_context.config.limits.code_timeout_minutes
    budget_minutes = max(int(time_limit * SCREEN_TIME_FRACTION), 1)
    best_run = ledger.best_approved_run()
    best_text = describe_run(ledger, best_run) if best_run else "none yet"
    if candidates_file.exists():
        candidates_location_text = (
            f"Candidates: {candidates_file} (add {candidates_file.parent} to sys.path, then "
            f"import {candidates_file.stem}). Import the functions, do not copy them.\n"
        )
        candidates_text = (
            "CANDIDATES (their code is in the candidates file):\n"
            f"{describe_candidate_list(candidates)}"
        )
    else:
        candidates_location_text = ""
        candidates_text = (
            f"CANDIDATES (put their code in your script):\n{describe_candidates(candidates)}"
        )
    interface_text = summarize_features_source(
        read_full_features_source(agent_context, ledger.current_feature_version)
    )
    return (
        "Screen candidate input-preparation ideas on the fixed validation splits, with ONE "
        "script.\n"
        f"Current preparation: {current_file} (add {workspace.features_folder} to sys.path, "
        f"then import {current_file.stem})\n"
        f"{candidates_location_text}"
        f"Splits file: {workspace.splits_path}\n"
        f"Primary metric: {ledger.plan.primary_metric} ({direction} is better)\n"
        f"Best run so far, to choose the proxy model: {best_text}\n"
        f"Seed: {RANDOM_SEED}\n"
        f"Time budget: the whole script finishes within {budget_minutes} minutes.\n"
        f"Project folder (read only): {workspace.project_folder}\n\n"
        f"RULES:\n{SCREENING_RULES_TEXT}\n\n"
        f"{candidates_text}\n\n"
        f"Interface of the current preparation:\n{interface_text}\n\n"
        f"{describe_brief(ledger)}\n\n"
        f"{SPLITS_FORMAT_TEXT}"
    )


def build_new_version_task(
    kept_candidates: list[Candidate], ledger: Ledger, agent_context: AgentContext
) -> str:
    workspace = agent_context.workspace
    new_file = workspace.features_file(ledger.current_feature_version + 1)
    current_file = workspace.features_file(ledger.current_feature_version)
    current_source = trim_text(
        read_full_features_source(agent_context, ledger.current_feature_version),
        NEW_VERSION_SOURCE_CHARS,
    )
    return (
        f"Write the file {new_file}. It is the current file {current_file} plus the accepted "
        "candidates below.\n"
        "Copy the code of every accepted candidate unchanged into the file (do not import the "
        f"candidates file). If the current file has a {FEATURE_STEPS_NAME} list, add the tuple "
        f"(name, fit_function, apply_function) for each candidate to {FEATURE_STEPS_NAME}. "
        "Otherwise wire the candidates in following the style of the file. Change nothing "
        "else: every existing function keeps its name, arguments and code.\n"
        "In the quick check under if __name__ == \"__main__\": run fit_preparation and "
        "apply_preparation on a sample of the real data. For tables the output must have one "
        "row per input row, no duplicate column names and no infinite values. Print the number "
        "of output columns in the result.\n\n"
        f"{FEATURES_FILE_RULES_TEXT}\n\n"
        f"Current file source:\n{current_source}\n\n"
        f"Accepted candidates:\n{describe_candidates(kept_candidates)}\n\n"
        f"Project folder (read only): {workspace.project_folder}"
    )


def propose_foundation(ledger: Ledger, agent_context: AgentContext) -> FoundationPlan | None:
    """Round 0: turn the EDA into an ordered to-do list for features_v1.py."""
    values = build_feature_prompt_values(ledger, agent_context, build_foundation_digest(ledger))
    prompt = render_prompt(FOUNDATION_PROMPT, **values)
    event_bus = agent_context.event_bus
    try:
        foundation_plan = agent_context.llm_client.ask_structured(
            AGENT_NAME, prompt, FoundationPlan
        )
    except LlmUnavailableError:
        event_bus.emit(EventType.WARNING, AGENT_NAME, "no to-do list, using the usual first step")
        return None
    trimmed_plan = foundation_plan.model_copy(
        update={"steps": foundation_plan.steps[:FOUNDATION_MAX_STEPS]}
    )
    event_bus.emit(
        EventType.INFO, AGENT_NAME, f"to-do list with {len(trimmed_plan.steps)} steps"
    )
    return trimmed_plan


def build_basic_preparation(
    ledger: Ledger,
    agent_context: AgentContext,
    foundation_plan: FoundationPlan | None = None,
) -> FeatureStepResult:
    coder_result = run_coder(
        task_text=build_basic_preparation_task(ledger, agent_context, foundation_plan),
        result_tag=FEATURES_RESULT_TAG,
        result_keys_text=FEATURES_RESULT_KEYS_TEXT,
        working_folder=agent_context.workspace.features_folder,
        script_name="features_v1.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )
    if not coder_result.succeeded:
        error_line = last_error_line(coder_result.error_text)
        log_path = agent_context.workspace.commands_log_path
        raise AgentFailedError(
            f"The Feature Engineer could not build the basic input preparation: "
            f"{error_line}. Full log: {log_path}"
        )
    reason_text = str(coder_result.result_data.get("description", ""))
    if foundation_plan is not None:
        reason_text += f" ({len(foundation_plan.steps)} planned steps)"
    skipped_notes = describe_skipped_steps(coder_result.result_data.get("handled"))
    if skipped_notes:
        agent_context.event_bus.emit(
            EventType.WARNING, AGENT_NAME, f"steps not done: {skipped_notes}"
        )
        reason_text += f" Skipped: {skipped_notes}"
    basic_record = FeatureRecord(
        feature_id=ledger.next_feature_id(),
        name=BASIC_PREPARATION_NAME,
        reason=trim_text(reason_text, RECORD_NOTE_CHARS),
        kept=True,
        feature_version=1,
    )
    return FeatureStepResult(
        features=[*ledger.features, basic_record], current_feature_version=1, changed=True
    )


def describe_skipped_steps(handled: object) -> str:
    """Read the optional handled list of the features result. Anything odd is ignored."""
    if not isinstance(handled, list):
        return ""
    notes = [
        f"{item.get('step', '?')}: {item.get('note', '')}"
        for item in handled
        if isinstance(item, dict) and item.get("status") == "skipped"
    ]
    return "; ".join(notes)


def propose_candidates(ledger: Ledger, agent_context: AgentContext) -> list[Candidate]:
    values = build_feature_prompt_values(ledger, agent_context, build_propose_digest(ledger))
    features_source = summarize_features_source(
        read_full_features_source(agent_context, ledger.current_feature_version)
    )
    prompt = render_prompt(
        PROPOSE_PROMPT,
        **values,
        feature_version=ledger.current_feature_version,
        features_source=features_source,
        min_candidates=PROMPT_MIN_CANDIDATES,
        max_candidates=PROMPT_MAX_CANDIDATES,
        max_code_chars=MAX_CANDIDATE_CODE_CHARS,
    )
    candidate_list = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, CandidateList)
    if candidate_list.candidates:
        analysis_text = trim_text(candidate_list.analysis, ANALYSIS_EVENT_CHARS)
        agent_context.event_bus.emit(EventType.INFO, AGENT_NAME, f"analysis: {analysis_text}")
    else:
        pass_reason = last_sentence(candidate_list.analysis) or "no reason given"
        agent_context.event_bus.emit(EventType.INFO, AGENT_NAME, f"passed: {pass_reason}")
    tried_names = {normalize_candidate_name(feature.name) for feature in ledger.features}
    return clean_candidates(candidate_list.candidates, tried_names)


def screen_candidates(
    candidates: list[Candidate], ledger: Ledger, agent_context: AgentContext
) -> ScreenResult | None:
    coder_result = run_coder(
        task_text=build_screening_task(candidates, ledger, agent_context),
        result_tag=SCREEN_RESULT_TAG,
        result_keys_text=SCREEN_RESULT_KEYS_TEXT,
        working_folder=agent_context.workspace.features_folder / SCREENING_FOLDER_NAME,
        script_name=f"screen_after_round_{ledger.iteration}.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )
    if not coder_result.succeeded:
        return None
    try:
        screen_result = ScreenResult.model_validate(coder_result.result_data)
    except ValidationError:
        return None
    return screen_result if all_finite(screen_result.baseline_scores) else None


def judge_candidates(
    candidates: list[Candidate], screen_result: ScreenResult, higher_is_better: bool
) -> list[tuple[Candidate, float | None, bool]]:
    """Return (candidate, score gain, kept) for each candidate. Kept means a counted gain."""
    screened_by_name = {screened.name: screened for screened in screen_result.candidates}
    baseline_scores = screen_result.baseline_scores
    judged_candidates = []
    for candidate in candidates:
        screened = screened_by_name.get(candidate.name)
        has_scores = (
            screened is not None
            and screened.scores is not None
            and len(screened.scores) == len(baseline_scores)
            and all_finite(screened.scores)
        )
        if not has_scores:
            judged_candidates.append((candidate, None, False))
            continue
        gain = score_improvement(screened.scores, baseline_scores, higher_is_better)
        kept = is_counted_gain(screened.scores, baseline_scores, higher_is_better)
        judged_candidates.append((candidate, gain, kept))
    return judged_candidates


def write_new_version(
    kept_candidates: list[Candidate], ledger: Ledger, agent_context: AgentContext
) -> bool:
    new_version = ledger.current_feature_version + 1
    coder_result = run_coder(
        task_text=build_new_version_task(kept_candidates, ledger, agent_context),
        result_tag=FEATURES_RESULT_TAG,
        result_keys_text=FEATURES_RESULT_KEYS_TEXT,
        working_folder=agent_context.workspace.features_folder,
        script_name=f"features_v{new_version}.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )
    return coder_result.succeeded


def make_feature_record(
    candidate: Candidate,
    ledger: Ledger,
    earlier_records: list[FeatureRecord],
    score_gain: float | None,
    kept: bool,
    feature_version: int,
    note: str = "",
) -> FeatureRecord:
    reason_text = f"[{candidate.family}] {note or candidate.reason}"
    return FeatureRecord(
        feature_id=f"feature_{len(ledger.features) + len(earlier_records) + 1:03d}",
        name=candidate.name,
        reason=reason_text,
        finding_id=candidate.finding_id,
        score_gain=score_gain,
        kept=kept,
        feature_version=feature_version,
    )


def run_features_round(ledger: Ledger, agent_context: AgentContext) -> FeatureStepResult:
    event_bus = agent_context.event_bus
    unchanged = FeatureStepResult(
        features=ledger.features,
        current_feature_version=ledger.current_feature_version,
        changed=False,
    )

    candidates = propose_candidates(ledger, agent_context)
    if not candidates:
        event_bus.emit(EventType.WARNING, AGENT_NAME, "no new candidates to test")
        return unchanged

    current_source = read_full_features_source(agent_context, ledger.current_feature_version)
    accepted_candidates, rejected_candidates = write_candidates_file(
        candidates,
        candidates_file_path(ledger, agent_context),
        require_fit_and_apply=FEATURE_STEPS_NAME in current_source,
    )
    if not accepted_candidates:
        event_bus.emit(EventType.WARNING, AGENT_NAME, "no candidate passed the code check")
        return unchanged
    event_bus.emit(
        EventType.INFO,
        AGENT_NAME,
        f"screening {len(accepted_candidates)} candidates"
        f" ({len(rejected_candidates)} failed the code check)",
    )

    screen_result = screen_candidates(accepted_candidates, ledger, agent_context)
    if screen_result is None:
        event_bus.emit(EventType.WARNING, AGENT_NAME, "the screening script did not work")
        return unchanged

    judged_candidates = judge_candidates(
        accepted_candidates, screen_result, ledger.plan.higher_is_better
    )
    kept_candidates = [candidate for candidate, _, kept in judged_candidates if kept]
    new_version = ledger.current_feature_version + 1
    version_written = bool(kept_candidates) and write_new_version(
        kept_candidates, ledger, agent_context
    )
    if kept_candidates and not version_written:
        event_bus.emit(EventType.WARNING, AGENT_NAME, "could not write the new features file")

    skipped_names = {
        screened.name
        for screened in screen_result.candidates
        if screened.error.startswith(SKIPPED_FOR_TIME_PREFIX)
    }
    new_records: list[FeatureRecord] = []
    for candidate, problem in rejected_candidates:
        new_records.append(
            make_feature_record(
                candidate, ledger, new_records, None, False, 0, note=f"rejected: {problem}"
            )
        )
    for candidate, gain, kept in judged_candidates:
        if candidate.name in skipped_names:
            continue
        is_in_new_version = kept and version_written
        new_records.append(
            make_feature_record(
                candidate,
                ledger,
                new_records,
                gain,
                is_in_new_version,
                new_version if is_in_new_version else 0,
            )
        )
    event_bus.emit(
        EventType.INFO,
        AGENT_NAME,
        f"{len(kept_candidates)} of {len(accepted_candidates)} candidates showed a counted gain",
    )
    return FeatureStepResult(
        features=[*ledger.features, *new_records],
        current_feature_version=new_version if version_written else ledger.current_feature_version,
        changed=version_written,
    )


def run_feature_engineer(ledger: Ledger, agent_context: AgentContext) -> FeatureStepResult:
    event_bus = agent_context.event_bus
    is_first_round = ledger.current_feature_version == 0
    start_message = "building the basic input preparation" if is_first_round else "features round"
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, start_message)

    if is_first_round:
        foundation_plan = (
            propose_foundation(ledger, agent_context) if FOUNDATION_PLAN_ENABLED else None
        )
        result = build_basic_preparation(ledger, agent_context, foundation_plan)
    else:
        result = run_features_round(ledger, agent_context)

    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"features file version {result.current_feature_version}"
        f"{'' if result.changed else ' (unchanged)'}",
    )
    return result
