"""Conditional routing and termination rules for the ML workflow graph."""

from __future__ import annotations

from typing import Literal

from mlagent.graph.state import Ledger, StrategistRoute


# Routing decision after EDA READ step
def route_eda_step(ledger: Ledger, max_eda_loops: int) -> Literal["eda_think", "eda_blueprint"]:
    # Exit loop if analyst flagged ready or max EDA budget reached
    if ledger.eda.done or ledger.eda.eda_loops_used >= max_eda_loops:
        return "eda_blueprint"
    # Otherwise continue exploration loop
    return "eda_think"


# Routing decision after ML Lead Strategist planning
def route_strategist(ledger: Ledger) -> StrategistRoute:
    # Return chosen strategic route
    return ledger.current_route


# Routing decision after Evaluator validation
def route_after_evaluator(
    ledger: Ledger,
    max_experiment_loops: int,
    max_consecutive_crashes: int = 3,
) -> Literal["ml_lead", "finalizer"]:
    # Check if budget exhausted (total runs or iterations completed)
    if len(ledger.runs) >= max_experiment_loops or (ledger.iteration + 1) >= max_experiment_loops:
        ledger.stop_reason = "max_experiment_loops_reached"
        return "finalizer"

    # Check for consecutive crashes
    recent_runs = ledger.runs[-max_consecutive_crashes:]
    if len(recent_runs) >= max_consecutive_crashes and all(r.crashed for r in recent_runs):
        ledger.stop_reason = "consecutive_crashes"
        return "finalizer"

    # Check if strategist requested finalization
    if ledger.current_route == "finalize":
        ledger.stop_reason = "strategist_finalize"
        return "finalizer"

    # Continue loop to next ML Lead hypothesis
    return "ml_lead"
