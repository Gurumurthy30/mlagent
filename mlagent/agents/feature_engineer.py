"""The Feature Engineer: builds the input preparation and tests which new steps help.

Round 0 writes features_v1.py, the basic preparation that makes the data model-ready.
Every features round after that proposes candidates, screens them on the fixed validation
splits, keeps those with a counted gain, and writes the next features_vN.py.

For tables these are features. For images, text and audio they are preprocessing,
augmentation or embedding steps. The agent does not care which.
"""

import re

from pydantic import BaseModel, ValidationError

from mlagent.agents.base import (
    AgentContext,
    AgentFailedError,
    build_common_prompt_values,
    display_name,
    render_prompt,
)
from mlagent.agents.coder import last_error_line, run_coder
from mlagent.digest import describe_brief, trim_text
from mlagent.events import EventType
from mlagent.scoring import all_finite, is_counted_gain, score_improvement
from mlagent.script_contract import SPLITS_FORMAT_TEXT
from mlagent.settings import (
    BIG_FILE_MB,
    MAX_CANDIDATE_CODE_CHARS,
    MAX_CANDIDATES,
    MAX_FEATURE_SOURCE_CHARS,
    MIN_CANDIDATES,
    RANDOM_SEED,
)
from mlagent.state import FeatureRecord, Ledger

AGENT_KEY = "feature_engineer"
AGENT_NAME = display_name(AGENT_KEY)
FEATURES_RESULT_TAG = "FEATURES"
SCREEN_RESULT_TAG = "SCREEN"
SCREENING_FOLDER_NAME = "screening"

FEATURES_RESULT_KEYS_TEXT = """\
- "module_path": the path of the features file (this script)
- "functions": list of the function names other scripts can import
- "description": one short sentence about what the file does"""

SCREEN_RESULT_KEYS_TEXT = """\
- "baseline_scores": list of numbers, the primary metric of the current setup on each \
validation split
- "candidates": list of objects {"name": candidate name, "scores": list of numbers in split \
order, or null when it failed, "error": short text or ""}"""

FEATURES_FILE_RULES_TEXT = """\
Your script IS the features file. It defines the functions other scripts will import. Under \
if __name__ == "__main__": it runs a quick check on a small sample of the real data and \
prints the result line. Importing the file must have no side effects: no printing, no file \
writing, no training.
Rules for every function:
- Anything learned from data (means, counts, category maps, vocabularies, scalers, fitted \
embeddings) must be learned by a function that receives training examples only, so scripts can \
call it inside each split. The matching apply function must work on new data.
- Never use the target or labels inside a preparation step.
- Handle missing values, unseen categories, empty inputs and corrupt files without crashing.
- Be deterministic: take a seed argument where randomness is used.
- Be efficient: work on whole columns or batches, never row by row loops on big data.
- Keep names and arguments stable. Other scripts depend on them.
Start the file with a docstring that says what each function takes and returns."""

PROPOSE_PROMPT = """\
You are the Feature Engineer of an ML team. You work like an experienced ML practitioner \
running a feature experiment: you do not try one idea at a time. You plan a batch of diverse \
ideas, the team tests them all in one screening run on the fixed validation splits, and only \
the ones with a counted gain are kept. For tables these are new columns. For images, text, \
audio and other data they are preprocessing, augmentation or embedding steps.

{run_rules}

WHAT THE TEAM KNOWS:
{digest}

CURRENT FEATURES FILE (version {feature_version}):
{features_source}

Think before you write the batch:
1. What did the EDA find that a better input could exploit: skew, missing patterns, string or \
ID structure, groups, time, interactions, noise, class imbalance, differences between train and \
test?
2. Read FEATURES TRIED. Ideas that gained show which direction works: try more of that family \
in a clearly different form. Ideas that failed or crashed show what to avoid. Never repeat an \
idea, and do not submit a renamed copy of one.
3. Read the Reviewer's instructions and the open feature ideas. They come first.
4. What would a strong practitioner try on this kind of data and task that nobody has tried yet?

Then write {min_candidates} to {max_candidates} candidates (3 to 6 when each training run is \
slow, for example deep learning). Quality matters more than count. Do not pad the list.
- Spread the batch across different families, so one test covers a lot of ground. Examples:
  - tables: ratios and differences, interactions, group aggregations and frequency counts \
(without the target), parsing of strings and IDs, date and time parts, log, rank or binning \
of skewed columns, missing-value indicators, text statistics
  - images: resizing, crops, color or normalization changes, augmentation, pretrained \
embeddings
  - text: cleaning, token or n-gram variants, length and style statistics, embeddings
  - audio: spectrogram settings, trimming or padding, augmentation, pretrained embeddings
  - time series: lags, rolling statistics, calendar parts, differences
- Mix safe, standard ideas that usually help with a few bolder ones. Put the ideas with the \
highest expected gain first.
- Each candidate must work on its own. Screening adds every candidate alone to the current \
setup, so never build one candidate on top of another.
- Each candidate is one small, complete, runnable Python function in the style of the current \
file. It uses the input data only, never the target or labels. Anything learned from data \
follows the learn-then-apply pattern of the current file, so it can be learned on training \
examples only. No placeholders, no pseudo code.
- Keep the cost low: a candidate must not make the screening much slower than the current setup.
- name: lowercase snake_case, specific (ratio_income_to_age, not feature_1).
- reason: one sentence in the form "I expect ... because ...", naming the signal you expect.
- finding_id: the exact EDA finding id that supports the idea (for example finding_003), or \
an empty string when it is a hypothesis of your own.
Return JSON: candidates (list of objects with name, code, reason, finding_id)."""


class Candidate(BaseModel):
    name: str
    code: str
    reason: str = ""
    finding_id: str = ""


class CandidateList(BaseModel):
    candidates: list[Candidate]


class ScreenedCandidate(BaseModel):
    name: str
    scores: list[float] | None = None
    error: str = ""


class ScreenResult(BaseModel):
    baseline_scores: list[float]
    candidates: list[ScreenedCandidate]


class FeatureStepResult(BaseModel):
    features: list[FeatureRecord]
    current_feature_version: int
    changed: bool


def normalize_candidate_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")


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
        cleaned_candidates.append(candidate.model_copy(update={"name": name, "code": code}))
    return cleaned_candidates[:MAX_CANDIDATES]


def read_features_source(agent_context: AgentContext, feature_version: int) -> str:
    features_path = agent_context.workspace.features_file(feature_version)
    return trim_text(features_path.read_text(encoding="utf-8"), MAX_FEATURE_SOURCE_CHARS)


def describe_candidates(candidates: list[Candidate]) -> str:
    return "\n\n".join(f"### {candidate.name}\n{candidate.code}" for candidate in candidates)


def build_basic_preparation_task(ledger: Ledger, agent_context: AgentContext) -> str:
    features_path = agent_context.workspace.features_file(1)
    return (
        f"Write the file {features_path}: the basic input preparation that makes this "
        "project's data model-ready.\n"
        "Tables: fill missing values, encode categories, split dates into parts. Images: "
        "loading, resizing, normalizing. Text: cleaning and tokenizing. Audio: loading, "
        "resampling, fixed length or spectrograms. Other data: the usual first step.\n\n"
        f"{FEATURES_FILE_RULES_TEXT}\n\n"
        f"{describe_brief(ledger)}\n\n"
        f"Project folder (read only): {agent_context.workspace.project_folder}\n"
        f"Never fully load a file larger than {BIG_FILE_MB} MB; use a sample for the check.\n"
        f"Playbook for this task: {ledger.plan.playbook}"
    )


def build_screening_task(
    candidates: list[Candidate], ledger: Ledger, agent_context: AgentContext
) -> str:
    workspace = agent_context.workspace
    direction = "higher" if ledger.plan.higher_is_better else "lower"
    current_file = workspace.features_file(ledger.current_feature_version)
    return (
        "Screen candidate input-preparation changes on the fixed validation splits.\n"
        f"1. Import the current preparation from {current_file}"
        f" (add {workspace.features_folder} to sys.path).\n"
        f"2. Score the current setup on every validation split of {workspace.splits_path} with "
        "a fast, simple model. These are the baseline_scores.\n"
        "3. For each candidate below, add it to the current setup alone and score it the same "
        "way on the same splits. If a candidate raises an error, record the error and go on.\n"
        "Use a fast proxy (a small model, a subsample, few epochs) so the whole script finishes "
        f"well inside the time limit. Seed: {RANDOM_SEED}. Learn anything from data inside "
        "each split only.\n"
        f"Primary metric: {ledger.plan.primary_metric} ({direction} is better).\n"
        f"Project folder (read only): {workspace.project_folder}\n\n"
        f"{describe_brief(ledger)}\n\n"
        f"Current features source (reference):\n"
        f"{read_features_source(agent_context, ledger.current_feature_version)}\n\n"
        f"Candidates:\n{describe_candidates(candidates)}\n\n"
        f"{SPLITS_FORMAT_TEXT}"
    )


def build_new_version_task(
    kept_candidates: list[Candidate], ledger: Ledger, agent_context: AgentContext
) -> str:
    workspace = agent_context.workspace
    new_file = workspace.features_file(ledger.current_feature_version + 1)
    current_file = workspace.features_file(ledger.current_feature_version)
    current_source = read_features_source(agent_context, ledger.current_feature_version)
    return (
        f"Write the file {new_file}. It is the current file "
        f"{current_file} plus the accepted "
        "candidates below. Every existing function must keep its name and arguments. Wire the "
        "accepted candidates in so scripts can use them.\n\n"
        f"{FEATURES_FILE_RULES_TEXT}\n\n"
        f"Current file source:\n{current_source}\n\n"
        f"Accepted candidates:\n{describe_candidates(kept_candidates)}\n\n"
        f"Project folder (read only): {workspace.project_folder}"
    )


def build_basic_preparation(ledger: Ledger, agent_context: AgentContext) -> FeatureStepResult:
    coder_result = run_coder(
        task_text=build_basic_preparation_task(ledger, agent_context),
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
    basic_record = FeatureRecord(
        feature_id=ledger.next_feature_id(),
        name="basic_preparation",
        reason=str(coder_result.result_data.get("description", "")),
        kept=True,
        feature_version=1,
    )
    return FeatureStepResult(
        features=[*ledger.features, basic_record], current_feature_version=1, changed=True
    )


def propose_candidates(ledger: Ledger, agent_context: AgentContext) -> list[Candidate]:
    common_values = build_common_prompt_values(AGENT_KEY, ledger, agent_context)
    prompt = render_prompt(
        PROPOSE_PROMPT,
        **common_values,
        feature_version=ledger.current_feature_version,
        features_source=read_features_source(agent_context, ledger.current_feature_version),
        min_candidates=MIN_CANDIDATES,
        max_candidates=MAX_CANDIDATES,
    )
    candidate_list = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, CandidateList)
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
    event_bus.emit(EventType.INFO, AGENT_NAME, f"screening {len(candidates)} candidates")

    screen_result = screen_candidates(candidates, ledger, agent_context)
    if screen_result is None:
        event_bus.emit(EventType.WARNING, AGENT_NAME, "the screening script did not work")
        return unchanged

    judged_candidates = judge_candidates(candidates, screen_result, ledger.plan.higher_is_better)
    kept_candidates = [candidate for candidate, _, kept in judged_candidates if kept]
    new_version = ledger.current_feature_version + 1
    version_written = bool(kept_candidates) and write_new_version(
        kept_candidates, ledger, agent_context
    )
    if kept_candidates and not version_written:
        event_bus.emit(EventType.WARNING, AGENT_NAME, "could not write the new features file")

    new_records = []
    for candidate, gain, kept in judged_candidates:
        is_in_new_version = kept and version_written
        new_records.append(
            FeatureRecord(
                feature_id=f"feature_{len(ledger.features) + len(new_records) + 1:03d}",
                name=candidate.name,
                reason=candidate.reason,
                finding_id=candidate.finding_id,
                score_gain=gain,
                kept=is_in_new_version,
                feature_version=new_version if is_in_new_version else 0,
            )
        )
    event_bus.emit(
        EventType.INFO,
        AGENT_NAME,
        f"{len(kept_candidates)} of {len(candidates)} candidates showed a counted gain",
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
        result = build_basic_preparation(ledger, agent_context)
    else:
        result = run_features_round(ledger, agent_context)

    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"features file version {result.current_feature_version}"
        f"{'' if result.changed else ' (unchanged)'}",
    )
    return result
