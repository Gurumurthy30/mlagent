"""ML Lead strategist agent: validation design and iterative hypothesis planning.

Responsibilities:
1. run_ml_lead_round_0: Locks in leak-free validation split scheme (Stratified, Group, Time-series),
   generates verified splits.json, and initializes baseline hypothesis.
2. run_ml_lead_hypothesis: Inspects evaluation feedback, diagnoses generalization bottlenecks,
   and formulates the next hypothesis (Champion-vs-Challenger linear ratchet).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

# Agent execution context and exceptions
from mlagent.agents.base import AgentContext, AgentFailedError

# Configuration constants
from mlagent.configs.settings import RANDOM_SEED, SPLITS_FILE_NAME

# Master graph state models
from mlagent.graph.state import (
    HypothesisPlan,
    HypothesisRecord,
    Ledger,
    ValidationPlan,
)

# Event logging and notifications
from mlagent.helper.events import EventType

# Split file validation contract
from mlagent.helper.script_contract import check_splits_file

# Prompt templates and role definitions
from mlagent.memory.assembler import MemoryAssembler
from mlagent.memory.cards import (
    HypothesisCard,
    LessonLearnedCard,
    ValidationPlanCard,
)
from mlagent.memory.store import MemoryStore
from mlagent.prompts.ml_lead import ML_LEAD_ROLE

AGENT_KEY = "ml_lead"
AGENT_NAME = "ML Lead"


def generate_splits_file(
    plan: ValidationPlan,
    ledger: Ledger,
    artifacts_folder: Path,
    project_folder: Path,
) -> Path:
    # Deterministically construct splits.json honoring validation scheme
    splits_path = artifacts_folder / SPLITS_FILE_NAME
    train_file = None

    # Search for training file from project profile
    if ledger.brief.profile:
        for f in ledger.brief.profile.files:
            if f.inferred_role == "train":
                train_file = project_folder / f.path
                break

    # Fallback search if profile role was unassigned
    if not train_file or not train_file.exists():
        for p in project_folder.glob("*.csv"):
            if "train" in p.name.lower():
                train_file = p
                break

    if not train_file or not train_file.exists():
        csv_files = list(project_folder.glob("*.csv"))
        if csv_files:
            train_file = csv_files[0]

    if not train_file or not train_file.exists():
        raise AgentFailedError(
            f"Cannot generate splits: no training data found in {project_folder}"
        )

    # Read training file to extract row index and split metadata
    sep = "\t" if train_file.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(train_file, sep=sep)
    n_samples = len(df)
    example_ids = list(range(n_samples))
    fold_of_example = [-1] * n_samples

    # Compute holdout indices if holdout_fraction > 0
    holdout_frac = plan.holdout_fraction
    hold_idx: list[int] = []
    target_col = ledger.brief.target_column
    stratify_target = None

    if target_col and target_col in df.columns:
        if plan.validation_scheme in ["stratified_kfold", "stratified_group_kfold"]:
            stratify_target = df[target_col].to_numpy()

    # Isolate holdout partition
    if holdout_frac > 0.0 and n_samples >= 20:
        n_holdout = max(1, int(n_samples * holdout_frac))
        if stratify_target is not None:
            try:
                cv_idx, hold_idx = train_test_split(
                    example_ids,
                    test_size=n_holdout,
                    stratify=stratify_target,
                    random_state=RANDOM_SEED,
                )
            except Exception:
                cv_idx, hold_idx = train_test_split(
                    example_ids, test_size=n_holdout, random_state=RANDOM_SEED
                )
        else:
            cv_idx, hold_idx = train_test_split(
                example_ids, test_size=n_holdout, random_state=RANDOM_SEED
            )
        # Mark holdout entries with -1
        for idx in hold_idx:
            fold_of_example[idx] = -1
    else:
        cv_idx = example_ids.copy()

    # Partition cv_idx across folds
    n_splits = max(2, plan.n_splits)
    scheme = plan.validation_scheme

    if scheme == "group_kfold" and plan.split_column in df.columns:
        groups = df.loc[cv_idx, plan.split_column].to_numpy()
        gkf = GroupKFold(n_splits=n_splits)
        for fold, (_, val_pos) in enumerate(gkf.split(cv_idx, groups=groups)):
            for pos in val_pos:
                fold_of_example[cv_idx[pos]] = fold
    elif (
        scheme == "stratified_group_kfold"
        and plan.split_column in df.columns
        and stratify_target is not None
    ):
        groups = df.loc[cv_idx, plan.split_column].to_numpy()
        y_cv = df.loc[cv_idx, target_col].to_numpy()
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        for fold, (_, val_pos) in enumerate(sgkf.split(cv_idx, y=y_cv, groups=groups)):
            for pos in val_pos:
                fold_of_example[cv_idx[pos]] = fold
    elif scheme == "stratified_kfold" and stratify_target is not None:
        y_cv = df.loc[cv_idx, target_col].to_numpy()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        for fold, (_, val_pos) in enumerate(skf.split(cv_idx, y_cv)):
            for pos in val_pos:
                fold_of_example[cv_idx[pos]] = fold
    elif scheme in ["time_series_split", "purged_time_series_split"]:
        tss = TimeSeriesSplit(n_splits=n_splits, gap=plan.gap)
        for fold, (_, val_pos) in enumerate(tss.split(cv_idx)):
            for pos in val_pos:
                fold_of_example[cv_idx[pos]] = fold
        # Backfill any pre-window indices into fold 0
        for i in cv_idx:
            if fold_of_example[i] == -1 and i not in hold_idx:
                fold_of_example[i] = 0
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        for fold, (_, val_pos) in enumerate(kf.split(cv_idx)):
            for pos in val_pos:
                fold_of_example[cv_idx[pos]] = fold

    # Construct splits JSON object
    payload = {
        "scheme": scheme,
        "seed": RANDOM_SEED,
        "n_splits": n_splits,
        "id_description": f"Row position in {train_file.name}",
        "example_ids": example_ids,
        "fold_of_example": fold_of_example,
    }
    splits_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Contract integrity verification
    errors = check_splits_file(splits_path)
    if errors:
        raise AgentFailedError(f"Generated splits file is invalid: {'; '.join(errors)}")

    return splits_path


def run_ml_lead_round_0(ledger: Ledger, agent_context: AgentContext) -> ValidationPlan:
    # Plan validation protocol and initialize Tier-1 baseline hypothesis via MemoryAssembler
    event_bus = agent_context.event_bus
    event_bus.emit(
        EventType.AGENT_STARTED,
        AGENT_NAME,
        "locking validation protocol for Round 0",
        ledger=ledger,
    )

    store = MemoryStore(agent_context.workspace.root)
    assembler = MemoryAssembler(store)
    task_text = "Lock in leak-free validation splitting protocol, holdout fraction, and simple Tier-1 baseline algorithm."
    prompt = assembler.build_prompt(
        agent_name="ml_lead_round_0",
        ledger=ledger,
        task=task_text,
        schema_hint=str(ValidationPlan.model_json_schema()),
    )

    try:
        plan = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=ValidationPlan,
            system_message=ML_LEAD_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(f"Round 0 validation planning failed: {exc}") from exc

    # Generate and verify splits.json on disk
    splits_path = generate_splits_file(
        plan=plan,
        ledger=ledger,
        artifacts_folder=agent_context.workspace.artifacts_folder,
        project_folder=agent_context.workspace.project_folder,
    )

    # Store validation plan in master ledger
    ledger.validation_plan = plan
    ledger.splits_path = str(splits_path)

    # Seed baseline hypothesis record
    hyp_id = ledger.next_hypothesis_id()
    baseline_hyp = HypothesisRecord(
        hypothesis_id=hyp_id,
        iteration=0,
        route="fe_and_model",
        base_feature_version=0,
        hypothesis_statement=plan.baseline_hypothesis_statement,
        target_model_family=plan.baseline_model_family,
        model_complexity_tier="tier_1_simple_baseline",
        model_instructions=[
            "Train Tier-1 simple baseline model using Stage 3 preprocessed features"
        ],
    )
    ledger.hypotheses.append(baseline_hyp)

    # Append ValidationPlanCard to MemoryStore
    summary_text = (
        f"Scheme: {plan.validation_scheme}. Splits: {plan.n_splits}. Holdout: {plan.holdout_fraction:.2f}. "
        f"Baseline: {plan.baseline_model_family}."
    )
    val_card = ValidationPlanCard(
        tag="validation_plan",
        card_id="validation_plan",
        owner="ml_lead",
        iteration=0,
        memory_summary=summary_text,
        payload={
            "scheme": plan.validation_scheme,
            "n_splits": plan.n_splits,
            "holdout_fraction": plan.holdout_fraction,
            "split_column": plan.split_column,
            "time_column": plan.time_column,
            "model_progression": plan.planned_model_progression,
        },
        priority="core",
    )
    try:
        store.append_card(val_card)
    except FileExistsError:
        pass

    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"Validation locked ({plan.validation_scheme}, {plan.n_splits} folds): {plan.cli_summary}",
    )
    return plan


def run_ml_lead_hypothesis(ledger: Ledger, agent_context: AgentContext) -> HypothesisPlan:
    # Formulate next iteration hypothesis based on Champion-vs-Challenger history via MemoryAssembler
    event_bus = agent_context.event_bus
    iteration = ledger.iteration + 1
    event_bus.emit(
        EventType.AGENT_STARTED,
        AGENT_NAME,
        f"formulating hypothesis for iteration {iteration}",
        ledger=ledger,
    )

    store = MemoryStore(agent_context.workspace.root)
    assembler = MemoryAssembler(store)
    task_text = f"Formulate the next empirical hypothesis for iteration {iteration} based on Champion vs Challenger trajectory."
    prompt = assembler.build_prompt(
        agent_name="ml_lead",
        ledger=ledger,
        task=task_text,
        schema_hint=str(HypothesisPlan.model_json_schema()),
    )

    try:
        plan = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=HypothesisPlan,
            system_message=ML_LEAD_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(
            f"Hypothesis formulation failed in iteration {iteration}: {exc}"
        ) from exc

    # Record newly formulated hypothesis
    hyp_id = ledger.next_hypothesis_id()
    hyp_record = HypothesisRecord(
        hypothesis_id=hyp_id,
        iteration=iteration,
        route=plan.next_route,
        base_feature_version=ledger.active_base_feature_version(),
        hypothesis_statement=plan.hypothesis_statement,
        selected_feature_idea_ids=plan.selected_feature_idea_ids,
        custom_feature_instructions=plan.custom_feature_instructions,
        target_model_family=plan.target_model_family,
        model_complexity_tier=plan.model_complexity_tier,
        model_instructions=plan.model_instructions,
        eda_revisit_question=plan.eda_revisit_question,
        bottleneck_diagnosis=plan.bottleneck_diagnosis,
        lesson_learned=plan.lesson_learned,
    )
    ledger.hypotheses.append(hyp_record)
    ledger.current_route = plan.next_route
    ledger.iteration = iteration

    # Append HypothesisCard
    hyp_summary = (
        f"Hypothesis {hyp_id} ({plan.next_route}): {plan.hypothesis_statement} "
        f"Target: {plan.target_model_family} (Tier {plan.model_complexity_tier})."
    )
    h_card = HypothesisCard(
        tag="hypothesis",
        card_id=hyp_id,
        owner="ml_lead",
        iteration=iteration,
        memory_summary=hyp_summary,
        payload={
            "hypothesis_id": hyp_id,
            "iteration": iteration,
            "route": plan.next_route,
            "base_feature_version": hyp_record.base_feature_version,
            "statement": plan.hypothesis_statement,
            "selected_idea_ids": plan.selected_feature_idea_ids,
            "target_model_family": plan.target_model_family,
            "tier": plan.model_complexity_tier,
            "lesson_learned": plan.lesson_learned,
        },
        priority="high",
    )
    try:
        store.append_card(h_card)
    except FileExistsError:
        pass

    # Append LessonLearnedCard if a lesson was derived
    if plan.lesson_learned:
        l_summary = f"Lesson {hyp_id}: {plan.lesson_learned} (Route: {plan.next_route})."
        l_card = LessonLearnedCard(
            tag="lesson_learned",
            card_id=f"lesson_{hyp_id}",
            owner="ml_lead",
            iteration=iteration,
            memory_summary=l_summary,
            payload={
                "lesson_id": f"lesson_{hyp_id}",
                "iteration": iteration,
                "lesson_text": plan.lesson_learned,
                "applies_to_route": plan.next_route,
            },
            priority="core",
        )
        try:
            store.append_card(l_card)
        except FileExistsError:
            pass

    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"Hypothesis formulated for round {iteration} ({plan.next_route}): {plan.cli_summary}",
    )
    return plan
