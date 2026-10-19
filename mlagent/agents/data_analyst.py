"""Exploratory Data Analysis (EDA) and Preprocessing Analyst agent.

Executes a bounded empirical investigation loop:
1. run_eda_think: Plans 1 to 5 quantitative questions targeting data doubts or distributions.
2. run_coder: Executes python scripts to extract exact distributions, correlations, and anomalies.
3. run_eda_read: Interprets outputs into quantitative EdaFinding records with modeling implications.
4. run_eda_blueprint: Synthesizes findings into a comprehensive column-by-column preprocessing plan
   and feature engineering backlog.
"""

from __future__ import annotations

import json
from typing import Any

# Agent execution context and exceptions
from mlagent.agents.base import AgentContext, AgentFailedError

# Script execution sub-agent
from mlagent.agents.coder import run_coder

# Configured execution limits
from mlagent.configs.settings import BIG_FILE_MB

# Graph state schemas
from mlagent.graph.state import (
    EdaBlueprint,
    EdaFinding,
    EdaFindingsSummary,
    EdaQuestionPlan,
    Ledger,
)

# Event logging and notifications
from mlagent.helper.events import EventType

# Agent prompt templates and roles
from mlagent.memory.assembler import MemoryAssembler
from mlagent.memory.cards import EdaFindingCard, PrepSpecCard
from mlagent.memory.store import MemoryStore
from mlagent.prompts.data_analyst import (
    EDA_ANALYST_ROLE,
    EDA_READ_PROMPT,
)

AGENT_KEY = "eda_analyst"
AGENT_NAME = "EDA Analyst"
RESULT_TAG = "EDA_FINDINGS"

RESULT_KEYS = """\
- "answers": list of objects {"question": str, "result": str or dict with exact metrics/quantiles}
- "tables": list of objects {"name": str, "rows": list of dicts (up to 20 rows)}
- "problems": list of strings for any execution failures"""


def run_eda_think(ledger: Ledger, agent_context: AgentContext) -> EdaQuestionPlan:
    # Plan quantitative exploratory questions for current round via MemoryAssembler
    event_bus = agent_context.event_bus
    eda_round = ledger.eda.eda_loops_used + 1
    max_rounds = agent_context.config.limits.max_eda_loops

    event_bus.emit(
        EventType.AGENT_STARTED,
        AGENT_NAME,
        f"planning questions for EDA round {eda_round} of {max_rounds}",
        ledger=ledger,
    )

    store = MemoryStore(agent_context.workspace.root)
    assembler = MemoryAssembler(store)
    task_text = (
        f"Plan 1 to 5 quantitative exploratory questions for EDA round {eda_round} of {max_rounds}."
    )
    prompt = assembler.build_prompt(
        agent_name="eda_think",
        ledger=ledger,
        task=task_text,
        schema_hint=str(EdaQuestionPlan.model_json_schema()),
    )

    try:
        plan = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=EdaQuestionPlan,
            system_message=EDA_ANALYST_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(f"EDA planning failed in round {eda_round}: {exc}") from exc

    event_bus.emit(
        EventType.INFO,
        AGENT_NAME,
        f"EDA round {eda_round} plan: {plan.cli_summary}",
    )
    return plan


def execute_eda_script(
    plan: EdaQuestionPlan,
    ledger: Ledger,
    agent_context: AgentContext,
) -> dict[str, Any]:
    # Formulate coder task and run inspection script
    eda_round = ledger.eda.eda_loops_used + 1
    working_folder = agent_context.workspace.artifacts_folder / AGENT_KEY
    working_folder.mkdir(parents=True, exist_ok=True)

    questions_formatted = "\n".join(f"{i}. {q}" for i, q in enumerate(plan.questions_to_test, 1))
    project_folder = agent_context.workspace.project_folder

    task_text = (
        f"Execute quantitative exploratory analysis for EDA round {eda_round}.\n\n"
        f"Questions to Answer:\n{questions_formatted}\n\n"
        "Instructions:\n"
        f"- Read data from project folder: {project_folder}\n"
        f"- Files larger than {BIG_FILE_MB}MB must be sampled or chunked (max 50,000 rows).\n"
        "- Compute exact statistical metrics: quantiles, counts, missing fractions, correlations.\n"
        "- Print JSON result at the end of execution matching required schema."
    )

    coder_result = run_coder(
        task_text=task_text,
        result_tag=RESULT_TAG,
        result_keys_text=RESULT_KEYS,
        working_folder=working_folder,
        script_name=f"eda_round_{eda_round}.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )

    if not coder_result.succeeded:
        # Return fallback error dictionary
        return {
            "answers": [],
            "tables": [],
            "problems": [f"Script failed: {coder_result.error_text}"],
        }

    return coder_result.result_data


def run_eda_read(
    coder_result_data: dict[str, Any],
    plan: EdaQuestionPlan,
    ledger: Ledger,
    agent_context: AgentContext,
) -> EdaFindingsSummary:
    # Interpret coder script output into structured empirical findings
    event_bus = agent_context.event_bus
    eda_round = ledger.eda.eda_loops_used + 1

    questions_text = "\n".join(f"- {q}" for q in plan.questions_to_test)
    script_output = json.dumps(coder_result_data, indent=2)

    prompt = EDA_READ_PROMPT.format(
        eda_round=eda_round,
        questions_text=questions_text,
        script_output=script_output[:8000],
    )

    try:
        summary = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=EdaFindingsSummary,
            system_message=EDA_ANALYST_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(f"EDA result reading failed in round {eda_round}: {exc}") from exc

    store = MemoryStore(agent_context.workspace.root)

    # Store newly discovered findings with deterministic finding_ids and append EdaFindingCard
    existing_count = len(ledger.eda.findings)
    for idx, draft in enumerate(summary.new_findings, 1):
        finding_id = f"F_{existing_count + idx:03d}"
        ledger.eda.findings.append(
            EdaFinding(
                finding_id=finding_id,
                eda_round=eda_round,
                statement=draft.statement,
                quantitative_evidence=draft.quantitative_evidence,
                modeling_implication=draft.modeling_implication,
                priority=draft.priority,
            )
        )
        finding_card = EdaFindingCard(
            tag="eda_finding",
            card_id=finding_id,
            owner="data_analyst",
            iteration=eda_round,
            memory_summary=f"{draft.statement}: {draft.modeling_implication}",
            payload={
                "finding_id": finding_id,
                "statement": draft.statement,
                "quantitative_evidence": draft.quantitative_evidence,
                "modeling_implication": draft.modeling_implication,
                "priority": draft.priority,
            },
            artifact_paths=[],
            priority=draft.priority,
        )
        try:
            store.append_card(finding_card)
        except FileExistsError:
            pass

    # Update ledger tracking
    ledger.eda.eda_loops_used = eda_round
    ledger.eda.open_questions = list(summary.unresolved_questions)

    event_bus.emit(
        EventType.INFO,
        AGENT_NAME,
        f"EDA round {eda_round} complete: {summary.cli_summary}",
    )
    return summary


def run_eda_blueprint(ledger: Ledger, agent_context: AgentContext) -> EdaBlueprint:
    # Synthesize all empirical findings into master data cleaning & feature blueprint via MemoryAssembler
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "synthesizing EDA blueprint", ledger=ledger)

    store = MemoryStore(agent_context.workspace.root)
    assembler = MemoryAssembler(store)
    task_text = (
        "Synthesize all empirical findings and data distributions into the master EDA blueprint."
    )
    prompt = assembler.build_prompt(
        agent_name="eda_blueprint",
        ledger=ledger,
        task=task_text,
        schema_hint=str(EdaBlueprint.model_json_schema()),
    )

    try:
        blueprint = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=EdaBlueprint,
            system_message=EDA_ANALYST_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(f"EDA blueprint generation failed: {exc}") from exc

    # Populate master ledger EDA fields
    ledger.eda.done = True
    ledger.eda.concise_executive_summary = blueprint.concise_executive_summary
    ledger.eda.duplicate_handling_policy = blueprint.duplicate_handling_policy
    ledger.eda.target_distribution_and_imbalance = blueprint.target_distribution_and_imbalance
    ledger.eda.recommended_validation_scheme = blueprint.recommended_validation_scheme
    ledger.eda.recommended_split_column = blueprint.recommended_split_column
    ledger.eda.redundant_or_collinear_columns_to_drop = (
        blueprint.redundant_or_collinear_columns_to_drop
    )
    ledger.eda.leakage_risks_to_guard = blueprint.leakage_risks_to_guard
    ledger.eda.column_preprocessing_plan = blueprint.column_preprocessing_plan
    ledger.eda.domain_consistency_rules = blueprint.domain_consistency_rules
    ledger.eda.feature_hypotheses = blueprint.feature_hypotheses_backlog

    # Append PrepSpecCard to MemoryStore
    prep_card = PrepSpecCard(
        tag="prep_spec",
        card_id=f"prep_spec_round_{ledger.iteration}",
        owner="data_analyst",
        iteration=ledger.iteration,
        memory_summary=blueprint.concise_executive_summary or "Data preparation plan locked.",
        payload={
            "concise_executive_summary": blueprint.concise_executive_summary,
            "duplicate_policy": blueprint.duplicate_handling_policy,
            "validation_scheme": blueprint.recommended_validation_scheme,
            "redundant_columns": blueprint.redundant_or_collinear_columns_to_drop,
            "leakage_risks": blueprint.leakage_risks_to_guard,
        },
        priority="high",
    )
    try:
        store.append_card(prep_card)
    except FileExistsError:
        pass

    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"EDA blueprint ready: {blueprint.cli_summary}",
    )
    return blueprint
