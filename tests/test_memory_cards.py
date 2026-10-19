"""Unit tests for typed memory cards."""

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
    WorkingScratchpadCard,
)


def test_memory_summary_allows_unrestricted_length() -> None:
    # Memory cards now allow arbitrary length memory_summary without restriction
    eda = EdaFindingCard(
        tag="eda_finding",
        card_id="f1",
        owner="data_analyst",
        memory_summary="X" * 1500,
        payload={"key": "val"},
    )
    assert len(eda.memory_summary) == 1500

    brief = ProjectBriefCard(
        tag="project_brief",
        card_id="brief1",
        owner="data_engineer",
        memory_summary="Y" * 2000,
        payload={"key": "val"},
    )
    assert len(brief.memory_summary) == 2000

    val = ValidationPlanCard(
        tag="validation_plan",
        card_id="v1",
        owner="ml_lead",
        memory_summary="Z" * 3000,
        payload={"key": "val"},
    )
    assert len(val.memory_summary) == 3000


def test_card_subclasses_have_expected_tag_and_priority() -> None:
    brief = ProjectBriefCard(
        card_id="b1",
        owner="data_engineer",
        memory_summary="Project brief summary",
        payload={"task": "classification"},
    )
    assert brief.tag == "project_brief"
    assert brief.priority == "core"

    finding = EdaFindingCard(
        card_id="f1",
        owner="data_analyst",
        memory_summary="EDA finding summary",
        payload={"statement": "Clean data"},
    )
    assert finding.tag == "eda_finding"
    assert finding.priority == "medium"

    prep = PrepSpecCard(
        card_id="p1",
        owner="data_analyst",
        memory_summary="Prep spec summary",
        payload={"cleaning": "done"},
    )
    assert prep.tag == "prep_spec"
    assert prep.priority == "high"

    val = ValidationPlanCard(
        card_id="val1",
        owner="ml_lead",
        memory_summary="Validation plan summary",
        payload={"scheme": "stratified_kfold"},
    )
    assert val.tag == "validation_plan"
    assert val.priority == "core"

    fv = FeatureVersionCard(
        card_id="fv1",
        owner="feature_engineer",
        memory_summary="Feature version summary",
        payload={"version": 1},
    )
    assert fv.tag == "feature_version"
    assert fv.priority == "high"

    run = ExperimentRunCard(
        card_id="exp1",
        owner="modeler",
        memory_summary="Run summary",
        payload={"exp_id": "exp1"},
    )
    assert run.tag == "experiment_run"
    assert run.priority == "high"

    verdict = EvaluationVerdictCard(
        card_id="verd1",
        owner="evaluator",
        memory_summary="Verdict summary",
        payload={"verdict": "approve"},
    )
    assert verdict.tag == "evaluation_verdict"
    assert verdict.priority == "high"

    champ = ChampionPointerCard(
        card_id="champ1",
        owner="evaluator",
        memory_summary="Champ pointer summary",
        payload={"champ": "exp1"},
    )
    assert champ.tag == "champion_pointer"
    assert champ.priority == "core"

    hyp = HypothesisCard(
        card_id="hyp1",
        owner="ml_lead",
        memory_summary="Hypothesis summary",
        payload={"hypothesis": "try trees"},
    )
    assert hyp.tag == "hypothesis"
    assert hyp.priority == "high"

    lesson = LessonLearnedCard(
        card_id="les1",
        owner="ml_lead",
        memory_summary="Lesson summary",
        payload={"lesson": "regularize"},
    )
    assert lesson.tag == "lesson_learned"
    assert lesson.priority == "core"

    final = FinalDeliverableCard(
        card_id="fin1",
        owner="finalizer",
        memory_summary="Final deliverable summary",
        payload={"strategy": "single"},
    )
    assert final.tag == "final_deliverable"
    assert final.priority == "core"

    scratch = WorkingScratchpadCard(
        card_id="scr1",
        owner="coder",
        memory_summary="Scratchpad summary",
        payload={"temp": 1},
    )
    assert scratch.tag == "working_scratchpad"
    assert scratch.priority == "low"


def test_round_trip_json_serialization() -> None:
    card = ExperimentRunCard(
        card_id="exp_test_001",
        owner="modeler",
        iteration=2,
        memory_summary="Run exp_test_001: lightgbm with CV=0.8920±0.0120. Holdout=0.8850. Status=healthy.",
        payload={
            "experiment_id": "exp_test_001",
            "model_family": "lightgbm",
            "cv_mean": 0.8920,
            "holdout_score": 0.8850,
        },
        artifact_paths=["/path/to/train_model.py", "/path/to/artifacts"],
        priority="high",
    )

    json_str = card.model_dump_json()
    loaded_card = ExperimentRunCard.model_validate_json(json_str)

    assert loaded_card.card_id == card.card_id
    assert loaded_card.tag == "experiment_run"
    assert loaded_card.iteration == 2
    assert loaded_card.memory_summary == card.memory_summary
    assert loaded_card.payload["cv_mean"] == 0.8920
    assert loaded_card.artifact_paths == card.artifact_paths
    assert loaded_card.priority == "high"
