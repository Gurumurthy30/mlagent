"""The Feature Engineer: designs data preparation and iteratively screens new feature candidates.

Round 0: LLM proposes foundation preparation steps; Coder writes features_v1.py.
Later rounds: LLM proposes candidate specs; Coder screens them on fixed splits;
code evaluates with is_counted_gain and adds winning features to features_vN.py.
"""

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
You are the Feature Engineer of an ML team. You design initial input preparation and cleaning.
You do not write Python code; the Coder implements your plan.

GOAL:
Turn raw data into clean, leak-free, model-ready inputs for baseline modeling.

SITUATION:
{digest}

HOW TO THINK:
- Address raw data anomalies (missing values, inconsistent strings, datetimes).
- Encode categoricals sensibly; normalize/scale if linear models are planned.
- For images/audio/text, establish standard sizes, channels, tokenization or embeddings.
- Ensure all parameters (means, medians, encoders) are fit ONLY on train splits.

YOU MAY:
- Keep preparation minimal if tree-based models are used.

OUTPUT:
Return JSON:
- thinking: 2-3 sentences describing the preparation strategy.
- steps: list of objects {{"name": str, "what_to_do": str, "why": str}}."""

PROPOSE_PROMPT = """\
You are the Feature Engineer of an ML team. You propose candidate feature ideas to improve validation score.
You do not write Python code; candidate specs are screened by a fast proxy script.

GOAL:
Propose up to {max_candidates} candidate feature specs that target weaknesses from recent rounds and EDA findings.

SITUATION:
{digest}

EXISTING FUNCTIONS IN FEATURES FILE:
{existing_features}

HOW TO THINK:
- Target error segments from the Reviewer's instructions and high-priority EDA findings.
- Think in families: ratio, interaction, aggregation, frequency, datetime, missing_flag, text_stat, embedding.
- Name exact columns and mathematical operations in the idea field.
- If current features already capture the available signal and no new idea is likely to help, return an empty list.

YOU MAY:
- Propose an empty list of candidates (pass) when no idea is worth testing.

OUTPUT:
Return JSON:
- thinking: 2-3 sentences explaining your reasoning.
- candidates: list of up to {max_candidates} objects {{"name": str, "family": str, "reason": str, "finding_id": str, "idea": str}}."""


class FoundationStep(BaseModel):
    name: str
    what_to_do: str
    why: str = ""


class FoundationDraft(BaseModel):
    thinking: str = ""
    steps: list[FoundationStep] = []


class CandidateSpec(BaseModel):
    name: str
    family: str = "general"
    reason: str = ""
    finding_id: str = ""
    idea: str


class CandidateList(BaseModel):
    thinking: str = ""
    candidates: list[CandidateSpec] = []


def inspect_features_file(path: Path) -> str:
    """Extract top-level function names and docstrings from a features module."""
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
    return "\n".join(lines) or "No functions defined."


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

    coder_task = (
        f"Create the foundational features module at {features_v1_path}.\n"
        "Implement data preparation based on these steps:\n"
        f"{steps_text}\n\n"
        "CONTRACT INTERFACE (must be strictly followed):\n"
        "1. fit_preparation(train_data, seed=42) -> state dict/object (fits all scalers/encoders strictly on train_data).\n"
        "2. apply_preparation(data, state) -> DataFrame/numpy array (transforms data using fitted state).\n\n"
        f"Project folder (read-only): {workspace.project_folder}\n"
        f"Brief target: {ledger.brief.target or 'none'}, ID column: {ledger.brief.id_column or 'none'}\n\n"
        "Under `if __name__ == '__main__':`, test on a sample from the project data and print JSON:\n"
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
        digest=digest, existing_features=existing_funcs, max_candidates=max_candidates
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
    candidates_text = "\n".join(
        f"- Name: {c.name} | Family: {c.family} | Idea: {c.idea} (Reason: {c.reason})"
        for c in candidates
    )
    direction = "higher" if ledger.brief.higher_is_better else "lower"
    screening_folder = workspace.artifacts_folder / "screening"
    screening_folder.mkdir(parents=True, exist_ok=True)

    screen_task = (
        f"Screen these candidate features on the validation splits:\n"
        f"{candidates_text}\n\n"
        f"Current features file: {current_features_path}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Primary metric: {ledger.brief.primary_metric} ({direction} is better)\n"
        f"Project folder: {workspace.project_folder}\n"
        f"Seed: {RANDOM_SEED}\n\n"
        "Instructions:\n"
        "1. Load data and splits. Compute baseline validation scores across splits with a fast baseline model.\n"
        "2. For each candidate, implement it, transform data, train fast model, and collect scores on validation splits.\n"
        f"3. Print JSON <{SCREEN_RESULT_TAG}> with baseline_scores and candidates list (name, scores, error).\n\n"
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
        winners_text = "\n".join(f"- {w[0].name}: {w[0].idea}" for w in winners)

        incorporate_task = (
            f"Add these winning features into a new features module at {new_features_path}:\n"
            f"{winners_text}\n\n"
            f"Base features file to copy and expand: {current_features_path}\n"
            "Preserve existing fit_preparation and apply_preparation functions, integrating the new features cleanly.\n"
            f"Test on sample and output <{FOUNDATION_RESULT_TAG}>{{\"status\": \"ok\"}}</{FOUNDATION_RESULT_TAG}>."
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

    event_bus.emit(EventType.INFO, AGENT_NAME, cli_line)
    return {
        "features": all_features,
        "current_feature_version": new_version if changed else ledger.current_feature_version,
        "last_features_changed": changed,
    }
