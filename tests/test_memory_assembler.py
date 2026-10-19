"""Unit tests for MemoryAssembler card selection, priority tiering, and prompt delimiting."""

from pathlib import Path

from mlagent.graph.state import ExperimentRun, Ledger
from mlagent.memory.assembler import MemoryAssembler
from mlagent.memory.cards import (
    ChampionPointerCard,
    EdaFindingCard,
    EvaluationVerdictCard,
    ExperimentRunCard,
    FeatureVersionCard,
    FinalDeliverableCard,
    HypothesisCard,
    LessonLearnedCard,
    PrepSpecCard,
    ProjectBriefCard,
    ValidationPlanCard,
)
from mlagent.memory.store import MemoryStore


def _seed_basic_store(store: MemoryStore) -> None:
    # Seed standard baseline cards into the store
    brief = ProjectBriefCard(
        card_id="project_brief",
        owner="data_engineer",
        memory_summary="Classify customer churn on tabular features.",
        payload={"task": "churn"},
    )
    store.append_card(brief)

    champ = ChampionPointerCard(
        card_id="champion_001",
        owner="evaluator",
        memory_summary="Champion: experiment_001 (FV 1) CV=0.8800.",
        payload={"champ": "experiment_001"},
    )
    store.update_champion_pointer(champ)

    val = ValidationPlanCard(
        card_id="validation_plan",
        owner="ml_lead",
        memory_summary="Stratified K-Fold with 5 splits.",
        payload={"scheme": "stratified_kfold"},
    )
    store.append_card(val)

    prep = PrepSpecCard(
        card_id="prep_spec_round_0",
        owner="data_analyst",
        memory_summary="Standard imputers and zero-leakage encoders.",
        payload={"prep": "ok"},
    )
    store.append_card(prep)

    fv = FeatureVersionCard(
        card_id="fv_1",
        owner="feature_engineer",
        memory_summary="Feature version 1 baseline features.",
        payload={"v": 1},
    )
    store.append_card(fv)

    hyp = HypothesisCard(
        card_id="hyp_001",
        owner="ml_lead",
        memory_summary="Train GBDT on baseline features.",
        payload={"hyp": 1},
    )
    store.append_card(hyp)

    run = ExperimentRunCard(
        card_id="experiment_001",
        owner="modeler",
        memory_summary="LightGBM baseline CV=0.8800.",
        payload={"exp": 1},
    )
    store.append_card(run)

    verd = EvaluationVerdictCard(
        card_id="verdict_001",
        owner="evaluator",
        memory_summary="Approve experiment_001 as Champion.",
        payload={"verdict": "approve"},
    )
    store.append_card(verd)

    lesson = LessonLearnedCard(
        card_id="lesson_001",
        owner="ml_lead",
        memory_summary="Tree depth 6 prevents overfit.",
        payload={"lesson": 1},
        priority="core",
    )
    store.append_card(lesson)

    final = FinalDeliverableCard(
        card_id="final_deliverable",
        owner="finalizer",
        memory_summary="Final model packaged with predictions.",
        payload={"final": 1},
    )
    store.append_card(final)


def test_prompt_uses_required_delimiters(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    _seed_basic_store(store)
    assembler = MemoryAssembler(store)
    ledger = Ledger()

    prompt = assembler.build_prompt(
        agent_name="modeler",
        ledger=ledger,
        task="Train candidate model",
        schema_hint='{"type": "object"}',
    )

    assert "<system>" in prompt and "</system>" in prompt
    assert "<memory>" in prompt and "</memory>" in prompt
    assert "<task>" in prompt and "</task>" in prompt
    assert "<schema>" in prompt and "</schema>" in prompt


def test_project_brief_and_champion_pointer_present_in_agents(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    _seed_basic_store(store)
    assembler = MemoryAssembler(store)
    ledger = Ledger()

    agents_with_both = [
        "eda_think",
        "ml_lead",
        "feature_engineer",
        "modeler",
        "evaluator",
        "finalizer",
    ]
    for agent in agents_with_both:
        prompt = assembler.build_prompt(agent, ledger, task="Test task")
        assert "[project_brief]" in prompt, f"Missing project_brief in {agent}"
        assert "[champion_pointer]" in prompt, f"Missing champion_pointer in {agent}"


def test_core_cards_never_dropped_when_trimming(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    _seed_basic_store(store)

    # Add 50 low/medium priority cards to trigger trimming
    for i in range(1, 51):
        card = EdaFindingCard(
            card_id=f"finding_extra_{i:03d}",
            owner="data_analyst",
            iteration=1,
            memory_summary=f"Extra finding {i}",
            payload={"i": i},
            priority="medium",
        )
        store.append_card(card)

    assembler = MemoryAssembler(store)
    ledger = Ledger()

    # Request prompt for eda_blueprint (loads many findings)
    prompt = assembler.build_prompt("eda_blueprint", ledger, task="Build blueprint")

    # Protected core card must remain
    assert "[project_brief]" in prompt


def test_agent_card_selection_matches_spec(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    _seed_basic_store(store)
    assembler = MemoryAssembler(store)
    ledger = Ledger()

    # Modeler prompt must contain: project_brief, validation_plan, champion_pointer, feature_version, hypothesis, experiment_run
    modeler_prompt = assembler.build_prompt("modeler", ledger, task="Train model")
    assert "[project_brief]" in modeler_prompt
    assert "[validation_plan]" in modeler_prompt
    assert "[champion_pointer]" in modeler_prompt
    assert "[feature_version]" in modeler_prompt
    assert "[hypothesis]" in modeler_prompt
    assert "[experiment_run]" in modeler_prompt

    # Evaluator prompt must contain: project_brief, validation_plan, experiment_run, champion_pointer, evaluation_verdict
    evaluator_prompt = assembler.build_prompt("evaluator", ledger, task="Audit run")
    assert "[project_brief]" in evaluator_prompt
    assert "[validation_plan]" in evaluator_prompt
    assert "[experiment_run]" in evaluator_prompt
    assert "[champion_pointer]" in evaluator_prompt
    assert "[evaluation_verdict]" in evaluator_prompt


def test_100_runs_in_ledger_does_not_leak_raw_json(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    _seed_basic_store(store)
    assembler = MemoryAssembler(store)

    # Populate a Ledger with 100 fake experiment runs
    ledger = Ledger()
    for i in range(1, 101):
        ledger.runs.append(
            ExperimentRun(
                experiment_id=f"huge_fake_run_{i:03d}",
                iteration=i,
                model_family="RandomForest",
                cv_mean=0.8500 + (i * 0.0001),
                artifact_dir=f"/tmp/fake_dir_{i}",
                summary=f"SecretRawDataToken_{i}",
            )
        )

    prompt = assembler.build_prompt("modeler", ledger, task="Train next model")

    # The prompt should NEVER contain raw JSON dumps of all 100 runs
    assert "huge_fake_run_050" not in prompt
    assert "SecretRawDataToken_099" not in prompt
    # Modeler should have at most the 3 cards loaded from store, not 100
    assert prompt.count("[experiment_run]") <= 3
