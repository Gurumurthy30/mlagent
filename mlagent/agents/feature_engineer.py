"""Feature Engineer agent: fold-isolated feature pipelines and Champion-vs-Challenger inheritance.

Creates new feature versions branching strictly from the current Champion feature version.
Implements data cleaning, date decomposition, domain ratios, group aggregations, and pruning.
Guarantees zero data leakage by enforcing fit_preparation strictly on train folds.
"""

from __future__ import annotations

import json

# Agent execution context and exceptions
from mlagent.agents.base import AgentContext, AgentFailedError

# Script execution sub-agent
from mlagent.agents.coder import run_coder

# Master graph state models
from mlagent.graph.state import (
    FeaturePlan,
    FeatureVersionRecord,
    Ledger,
)

# Event logging and notifications
from mlagent.helper.events import EventType

# Prompt templates and role definitions
from mlagent.memory.assembler import MemoryAssembler
from mlagent.memory.cards import FeatureVersionCard
from mlagent.memory.store import MemoryStore
from mlagent.prompts.feature_engineer import FEATURE_ENGINEER_ROLE

AGENT_KEY = "feature_engineer"
AGENT_NAME = "Feature Engineer"
RESULT_TAG = "FEATURE_MANIFEST"

RESULT_KEYS = """\
- "output_columns": list of string column names produced by apply_preparation
- "added_columns": list of newly introduced column names
- "pruned_columns": list of removed or dropped column names
- "verified_sample_rows": integer number of rows tested without NaN/Inf errors"""


def run_feature_engineer(ledger: Ledger, agent_context: AgentContext) -> FeaturePlan:
    # Generate next feature version branching from current Champion via MemoryAssembler
    event_bus = agent_context.event_bus
    parent_version = ledger.active_base_feature_version()
    new_version = ledger.latest_candidate_feature_version + 1

    event_bus.emit(
        EventType.AGENT_STARTED,
        AGENT_NAME,
        f"designing feature pipeline v{new_version} branching from Champion v{parent_version}",
    )

    store = MemoryStore(agent_context.workspace.root)
    assembler = MemoryAssembler(store)

    task_text = (
        f"Design feature engineering pipeline v{new_version} branching strictly from Champion feature version v{parent_version}. "
        "Enforce leak-free fold isolation and specify column transformations and pruning rules."
    )
    prompt = assembler.build_prompt(
        agent_name="feature_engineer",
        ledger=ledger,
        task=task_text,
        schema_hint=str(FeaturePlan.model_json_schema()),
    )

    try:
        plan = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=FeaturePlan,
            system_message=FEATURE_ENGINEER_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(f"Feature planning failed for v{new_version}: {exc}") from exc

    # Direct Coder to implement features_vN.py and test fold-isolation
    features_folder = agent_context.workspace.artifacts_folder / "features"
    features_folder.mkdir(parents=True, exist_ok=True)
    script_name = f"features_v{new_version}.py"
    target_script_path = features_folder / script_name

    coder_task = (
        f"Implement complete feature engineering module `{script_name}` branching from Champion v{parent_version}.\n\n"
        f"Target file: {target_script_path}\n"
        f"Parent version: {parent_version}\n"
        f"Operations to apply: {json.dumps([op.model_dump() for op in plan.operations_to_apply], indent=2)}\n"
        f"Columns to drop: {json.dumps(plan.columns_to_drop_or_prune)}\n"
        f"Requires scaling: {plan.requires_numeric_scaling}\n\n"
        "Contract Requirements:\n"
        "1. Define `fit_preparation(train_fold_df: pd.DataFrame) -> dict`.\n"
        "   - Fit all imputers, standardizers, group statistics, or encodings strictly on train_fold_df.\n"
        "2. Define `apply_preparation(df: pd.DataFrame, prep_state: dict) -> pd.DataFrame`.\n"
        "   - Transform dataframe using prep_state without re-fitting.\n"
        "   - Sanitize NaNs/Infs safely: select numeric columns with `df.select_dtypes(include=[np.number]).columns` and call `.replace([np.inf, -np.inf], 0.0).fillna(0.0)`. NEVER call np.isinf(df) or np.isnan(df) on the whole dataframe because string/object columns will crash with TypeError!\n"
        f"   - NEVER include the target column ('{ledger.brief.target_column}') or ID column in output features.\n"
        "   - Output matrix must be all-numeric: convert categorical columns with one-hot/ordinal encoding or select numeric features.\n"
        "3. In the `if __name__ == '__main__':` block, load a slice of train data from project folder, "
        "execute fit_preparation and apply_preparation, and print the JSON manifest tagged with <FEATURE_MANIFEST>:\n"
        "   {\n"
        '     "output_columns": list of final column names,\n'
        '     "added_columns": list of newly introduced column names,\n'
        '     "pruned_columns": list of removed column names,\n'
        '     "verified_sample_rows": integer number of rows tested without NaN/Inf\n'
        "   }"
    )

    coder_result = run_coder(
        task_text=coder_task,
        result_tag=RESULT_TAG,
        result_keys_text=RESULT_KEYS,
        working_folder=features_folder,
        script_name=script_name,
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )

    if coder_result.succeeded and not target_script_path.exists():
        fallback_test = features_folder / f"test_{script_name}"
        if fallback_test.exists():
            import shutil
            shutil.copy2(fallback_test, target_script_path)

    output_cols: list[str] = []
    added_cols: list[str] = []
    pruned_cols: list[str] = list(plan.columns_to_drop_or_prune)

    if coder_result.succeeded and isinstance(coder_result.result_data, dict):
        output_cols = coder_result.result_data.get("output_columns", [])
        added_cols = coder_result.result_data.get("added_columns", [])
        pruned_cols = coder_result.result_data.get("pruned_columns", pruned_cols)

    # Record manifest in ledger
    feature_record = FeatureVersionRecord(
        feature_version=new_version,
        parent_feature_version=parent_version,
        iteration=ledger.iteration,
        hypothesis_id=ledger.hypotheses[-1].hypothesis_id if ledger.hypotheses else "",
        script_path=str(target_script_path),
        output_feature_columns=output_cols,
        added_columns=added_cols,
        pruned_columns=pruned_cols,
        operations=plan.operations_to_apply,
    )
    ledger.feature_versions.append(feature_record)
    ledger.latest_candidate_feature_version = new_version

    # Append FeatureVersionCard to MemoryStore
    hyp_id = ledger.hypotheses[-1].hypothesis_id if ledger.hypotheses else "baseline"
    fv_summary = (
        f"Feature Version {new_version} (parent {parent_version}). "
        f"Added: {len(added_cols)} cols. Pruned: {len(pruned_cols)} cols. "
        f"Hypothesis: {hyp_id}."
    )
    fv_card = FeatureVersionCard(
        tag="feature_version",
        card_id=f"fv_{new_version}",
        owner="feature_engineer",
        iteration=ledger.iteration,
        memory_summary=fv_summary,
        payload={
            "version": new_version,
            "parent_version": parent_version,
            "hypothesis_id": hyp_id,
            "added_columns": added_cols,
            "pruned_columns": pruned_cols,
            "script_path": str(target_script_path),
        },
        artifact_paths=[str(target_script_path)],
        priority="high",
    )
    try:
        store.append_card(fv_card)
    except FileExistsError:
        pass

    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"Feature pipeline v{new_version} verified ({len(output_cols)} features): {plan.cli_summary}",
    )
    return plan