import pytest

from mlagent.digest import (
    BEST_RUNS_SHOWN,
    LATEST_RUNS_SHOWN,
    SECTION_BUILDERS_BY_AGENT,
    build_digest,
    build_progress_text,
    trim_text,
)
from mlagent.memory import create_new_ledger
from mlagent.settings import DIGEST_CHAR_LIMITS, MAX_ERROR_TEXT_CHARS
from mlagent.state import (
    Evaluation,
    ExperimentRun,
    FeatureIdea,
    FeatureRecord,
    Finding,
    Review,
)


def make_busy_ledger(run_count: int = 40):
    ledger = create_new_ledger("predict churn", "macro F1 mainly, also AUC")
    ledger.brief.summary = "customer table with churn label"
    ledger.brief.primary_metric = "macro_f1"
    ledger.plan.primary_metric = "macro_f1"
    ledger.plan.validation_scheme = "stratified_kfold"
    ledger.plan.playbook = "gradient boosting first. " * 200
    ledger.eda.checklist = {"missing values": "covered", "target balance": "open"}
    ledger.eda.findings = [
        Finding(finding_id="finding_001", statement="target is imbalanced", priority="high"),
        Finding(finding_id="finding_002", statement="id is unique", priority="low"),
    ]
    ledger.eda.feature_ideas = [FeatureIdea(description="tenure bucket", finding_id="finding_001")]
    ledger.eda.leakage_risks = ["column refund_date is only known after churn"]
    ledger.features = [FeatureRecord(feature_id="feature_001", name="tenure_bucket", kept=True)]
    for number in range(1, run_count + 1):
        experiment_id = f"experiment_{number:03d}"
        ledger.runs.append(
            ExperimentRun(
                experiment_id=experiment_id,
                model_family="lightgbm",
                score_mean=0.5 + number / 1000,
                description="x" * 300,
            )
        )
        ledger.evaluations.append(Evaluation(experiment_id=experiment_id, verdict="approve"))
    ledger.reviews = [Review(iteration=1, focus="model", reason="try boosting", instructions=["a"])]
    return ledger


def test_trim_text_keeps_short_text_and_cuts_long_text():
    assert trim_text("short", 100) == "short"

    trimmed = trim_text("x" * 500, 100)

    assert len(trimmed) == 100
    assert trimmed.endswith("(trimmed)")


@pytest.mark.parametrize("agent_name", list(SECTION_BUILDERS_BY_AGENT))
def test_every_agent_digest_stays_inside_its_limit(agent_name):
    digest = build_digest(agent_name, make_busy_ledger())

    assert len(digest) <= DIGEST_CHAR_LIMITS[agent_name]


@pytest.mark.parametrize("agent_name", list(SECTION_BUILDERS_BY_AGENT))
def test_every_agent_can_build_a_digest_from_an_empty_ledger(agent_name):
    digest = build_digest(agent_name, create_new_ledger("", ""))

    assert isinstance(digest, str)


def test_every_agent_with_a_limit_has_section_builders():
    assert set(SECTION_BUILDERS_BY_AGENT) == set(DIGEST_CHAR_LIMITS)


def test_run_list_is_compacted_to_best_and_latest_runs():
    ledger = make_busy_ledger(run_count=40)

    digest = build_digest("reviewer", ledger)

    shown_run_lines = [line for line in digest.splitlines() if line.startswith("- experiment_")]
    assert len(shown_run_lines) <= BEST_RUNS_SHOWN + LATEST_RUNS_SHOWN
    assert "showing" in digest and "of 40" in digest
    assert "BEST APPROVED RUN: experiment_040" in digest
    assert "experiment_020 |" not in digest


def test_lower_is_better_picks_the_lowest_score_as_best():
    ledger = make_busy_ledger(run_count=20)
    ledger.plan.higher_is_better = False

    digest = build_digest("reviewer", ledger)

    assert "BEST APPROVED RUN: experiment_001" in digest


def test_run_status_words_are_shown():
    ledger = create_new_ledger("task", "metric")
    ledger.runs = [
        ExperimentRun(experiment_id="experiment_001", score_mean=0.7),
        ExperimentRun(experiment_id="experiment_002", score_mean=0.9),
        ExperimentRun(experiment_id="experiment_003", error="MemoryError"),
        ExperimentRun(experiment_id="experiment_004", score_mean=0.8),
    ]
    ledger.evaluations = [
        Evaluation(experiment_id="experiment_001", verdict="approve"),
        Evaluation(experiment_id="experiment_002", verdict="reject", reasons=["leakage"]),
    ]

    digest = build_digest("reviewer", ledger)

    assert "experiment_001 | start |  | score 0.7000 | approved" in digest
    assert "| rejected |" in digest
    assert "| crashed |" in digest
    assert "| not evaluated |" in digest
    assert "experiment_002: leakage" in digest


def test_crash_error_text_is_cut_to_its_limit():
    ledger = create_new_ledger("task", "metric")
    ledger.runs = [ExperimentRun(experiment_id="experiment_001", error="Z" * 5000)]

    digest = build_digest("coder", ledger)

    assert digest.count("Z") == MAX_ERROR_TEXT_CHARS


def test_only_the_data_analyst_sees_open_checklist_items():
    ledger = make_busy_ledger(run_count=2)

    assert "target balance" in build_digest("data_analyst", ledger)
    assert "target balance" not in build_digest("modeler", ledger)


def test_data_engineer_sees_only_the_users_task():
    digest = build_digest("data_engineer", make_busy_ledger(run_count=2))

    assert "macro F1 mainly, also AUC" in digest
    assert "EDA FINDINGS" not in digest
    assert "EXPERIMENT RUNS" not in digest


def test_high_priority_findings_come_before_low_priority_ones():
    digest = build_digest("ml_lead", make_busy_ledger(run_count=2))

    assert digest.index("finding_001") < digest.index("finding_002")


def test_playbook_is_shortened_inside_the_plan():
    digest = build_digest("modeler", make_busy_ledger(run_count=2))

    assert digest.count("gradient boosting first.") < 200


def test_progress_text_names_the_best_run_and_the_task():
    ledger = make_busy_ledger(run_count=3)

    progress_text = build_progress_text(ledger)

    assert "best approved run: experiment_003" in progress_text
    assert "predict churn" in progress_text
    assert "stop reason: not stopped yet" in progress_text


def test_plan_section_skips_fields_that_are_still_empty():
    ledger = create_new_ledger("task", "metric")
    ledger.plan.primary_metric = "rmse"
    ledger.plan.higher_is_better = False

    digest = build_digest("modeler", ledger)

    assert "- primary metric: rmse (lower is better)" in digest
    assert "baseline" not in digest
    assert "playbook" not in digest
    assert "splits file" not in digest
    assert "validation:" in digest and "()" not in digest
