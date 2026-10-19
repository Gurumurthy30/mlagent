"""Master DAG orchestration for mlagent v3 using LangGraph StateGraph.

Connects the 8 professional nodes:
1. data_engineer: Directory scan and problem scoping
2. eda_think: Bounded exploratory investigation loop
3. eda_blueprint: Data cleaning specification and feature backlog
4. ml_lead_round_0: Validation locking and Tier-1 baseline configuration
5. feature_engineer: Fold-isolated feature engineering inheriting Champion features
6. modeler: Progressive complexity model training
7. tuner: Systematic hyperparameter optimization on Champion
8. evaluator: Independent audit and Champion linear ratchet
9. ml_lead: Iterative hypothesis strategist
10. finalizer: Deliverable packaging, ensembling, and drift monitoring
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

# Base agent context and errors
from mlagent.agents.base import AgentContext, AgentFailedError

# Agents
from mlagent.agents.data_analyst import (
    execute_eda_script,
    run_eda_blueprint,
    run_eda_read,
    run_eda_think,
)
from mlagent.agents.data_engineer import run_data_engineer
from mlagent.agents.evaluator import run_evaluator
from mlagent.agents.feature_engineer import run_feature_engineer
from mlagent.agents.finalizer import run_finalizer
from mlagent.agents.ml_lead import run_ml_lead_hypothesis, run_ml_lead_round_0
from mlagent.agents.modeler import run_modeler
from mlagent.agents.summarizer import run_summarizer
from mlagent.agents.tuner import run_tuner

# Routing logic
from mlagent.graph.routing import (
    route_after_evaluator,
    route_eda_step,
    route_strategist,
)

# Graph master ledger state
from mlagent.graph.state import Ledger

# Workspace persistence
from mlagent.helper.workspace import save_ledger


def build_pipeline_graph(agent_context: AgentContext) -> CompiledStateGraph:
    # Construct LangGraph StateGraph DAG for mlagent
    builder = StateGraph(Ledger)

    # 1. Data Engineer node
    def node_data_engineer(state: Ledger) -> Ledger:
        state.phase = "data_engineer"
        run_data_engineer(state, agent_context)
        save_ledger(agent_context.workspace, state, node_name="data_engineer")
        return state

    # 2. EDA Think node (executes think -> script -> read cycle)
    def node_eda_think(state: Ledger) -> Ledger:
        state.phase = "eda_analyst"
        plan = run_eda_think(state, agent_context)
        coder_data = execute_eda_script(plan, state, agent_context)
        summary = run_eda_read(coder_data, plan, state, agent_context)
        min_eda_rounds = min(2, agent_context.config.limits.max_eda_loops)
        if summary.ready_to_build_blueprint and state.eda.eda_loops_used >= min_eda_rounds:
            state.eda.done = True
        elif state.eda.eda_loops_used >= agent_context.config.limits.max_eda_loops:
            state.eda.done = True
        save_ledger(agent_context.workspace, state, node_name="eda_think")
        return state

    # 3. EDA Blueprint node
    def node_eda_blueprint(state: Ledger) -> Ledger:
        state.phase = "eda_analyst"
        run_eda_blueprint(state, agent_context)
        save_ledger(agent_context.workspace, state, node_name="eda_blueprint")
        return state

    # 4. ML Lead Round 0 (validation locking)
    def node_ml_lead_round_0(state: Ledger) -> Ledger:
        state.phase = "ml_lead"
        run_ml_lead_round_0(state, agent_context)
        save_ledger(agent_context.workspace, state, node_name="ml_lead_round_0")
        return state

    # 5. Feature Engineer node
    def node_feature_engineer(state: Ledger) -> Ledger:
        state.phase = "feature_engineer"
        run_feature_engineer(state, agent_context)
        save_ledger(agent_context.workspace, state, node_name="feature_engineer")
        return state

    # 6. Modeler node
    def node_modeler(state: Ledger) -> Ledger:
        state.phase = "modeler"
        run_modeler(state, agent_context)
        save_ledger(agent_context.workspace, state, node_name="modeler")
        return state

    # 7. Tuner node
    def node_tuner(state: Ledger) -> Ledger:
        state.phase = "tuner"
        run_tuner(state, agent_context)
        save_ledger(agent_context.workspace, state, node_name="tuner")
        return state

    # 8. Evaluator node
    def node_evaluator(state: Ledger) -> Ledger:
        state.phase = "evaluator"
        run_evaluator(state, agent_context)
        save_ledger(agent_context.workspace, state, node_name="evaluator")
        return state

    # 9. ML Lead strategist node
    def node_ml_lead(state: Ledger) -> Ledger:
        state.phase = "ml_lead"
        run_ml_lead_hypothesis(state, agent_context)
        save_ledger(agent_context.workspace, state, node_name="ml_lead")
        return state

    # 10. Finalizer node
    def node_finalizer(state: Ledger) -> Ledger:
        state.phase = "finalizer"
        run_finalizer(state, agent_context)
        save_ledger(agent_context.workspace, state, node_name="finalizer")
        return state

    # 11. Summarizer node
    def node_summarizer(state: Ledger) -> Ledger:
        state.phase = "summarizer"
        run_summarizer(state, agent_context)
        state.phase = "done"
        state.status = "done"
        save_ledger(agent_context.workspace, state, node_name="summarizer")
        return state

    # Register nodes
    builder.add_node("data_engineer", node_data_engineer)
    builder.add_node("eda_think", node_eda_think)
    builder.add_node("eda_blueprint", node_eda_blueprint)
    builder.add_node("ml_lead_round_0", node_ml_lead_round_0)
    builder.add_node("feature_engineer", node_feature_engineer)
    builder.add_node("modeler", node_modeler)
    builder.add_node("tuner", node_tuner)
    builder.add_node("evaluator", node_evaluator)
    builder.add_node("ml_lead", node_ml_lead)
    builder.add_node("finalizer", node_finalizer)
    builder.add_node("summarizer", node_summarizer)

    # Wire static linear edges
    builder.add_edge(START, "data_engineer")
    builder.add_edge("data_engineer", "eda_think")
    builder.add_edge("eda_blueprint", "ml_lead_round_0")
    builder.add_edge("ml_lead_round_0", "feature_engineer")
    builder.add_edge("feature_engineer", "modeler")
    builder.add_edge("modeler", "evaluator")
    builder.add_edge("tuner", "evaluator")
    builder.add_edge("finalizer", "summarizer")
    builder.add_edge("summarizer", END)

    # Wire conditional edges
    def eda_step_router(state: Ledger) -> str:
        max_eda = agent_context.config.limits.max_eda_loops
        return route_eda_step(state, max_eda)

    def evaluator_router(state: Ledger) -> str:
        max_exp = agent_context.config.limits.max_experiment_loops
        return route_after_evaluator(state, max_exp)

    def strategist_router(state: Ledger) -> str:
        return route_strategist(state)

    builder.add_conditional_edges(
        "eda_think",
        eda_step_router,
        {
            "eda_think": "eda_think",
            "eda_blueprint": "eda_blueprint",
        },
    )

    builder.add_conditional_edges(
        "evaluator",
        evaluator_router,
        {
            "ml_lead": "ml_lead",
            "finalizer": "finalizer",
        },
    )

    builder.add_conditional_edges(
        "ml_lead",
        strategist_router,
        {
            "fe_and_model": "feature_engineer",
            "model_only": "modeler",
            "tune_champion": "tuner",
            "revisit_eda": "eda_think",
            "finalize": "finalizer",
        },
    )

    return builder.compile()


def run_pipeline(ledger: Ledger, agent_context: AgentContext) -> Ledger:
    # Compile and execute the full ML pipeline
    graph = build_pipeline_graph(agent_context)

    try:
        final_state: Any = graph.invoke(ledger)
        if isinstance(final_state, Ledger):
            return final_state
        if isinstance(final_state, dict):
            # If LangGraph returns dictionary state, update ledger from it
            return Ledger.model_validate(final_state)
        return ledger
    except AgentFailedError as exc:
        ledger.status = "failed"
        ledger.error_message = str(exc)
        save_ledger(agent_context.workspace, ledger)
        return ledger
    except Exception as exc:
        ledger.status = "failed"
        ledger.error_message = f"Unexpected pipeline failure: {exc}"
        save_ledger(agent_context.workspace, ledger)
        return ledger
