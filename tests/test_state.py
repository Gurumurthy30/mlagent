from mlagent.state import (
    Evaluation,
    ExperimentRun,
    Finding,
    Ledger,
    ProjectBrief,
)


def make_run(experiment_id: str, score_mean: float | None, error: str = "") -> ExperimentRun:
    return ExperimentRun(experiment_id=experiment_id, score_mean=score_mean, error=error)


def test_ledger_round_trips_through_json():
    ledger = Ledger(user_task="predict churn", user_metrics_text="macro F1 mainly, also AUC")
    ledger.brief = ProjectBrief(target="churn", extra_metrics=["auc"])
    ledger.eda.findings.append(Finding(finding_id="finding_001", statement="Target is imbalanced"))
    ledger.runs.append(make_run("experiment_001", 0.71))

    restored = Ledger.model_validate_json(ledger.model_dump_json())

    assert restored == ledger


def test_new_ledgers_do_not_share_lists():
    first_ledger = Ledger()
    second_ledger = Ledger()

    first_ledger.runs.append(make_run("experiment_001", 0.5))
    first_ledger.eda.findings.append(Finding(finding_id="finding_001", statement="x"))

    assert second_ledger.runs == []
    assert second_ledger.eda.findings == []


def test_ids_come_from_code_and_count_up():
    ledger = Ledger()
    assert ledger.next_experiment_id() == "experiment_001"
    assert ledger.next_feature_id() == "feature_001"
    assert ledger.next_finding_id() == "finding_001"

    ledger.runs.append(make_run("experiment_001", 0.5))
    assert ledger.next_experiment_id() == "experiment_002"


def test_best_approved_run_when_higher_is_better():
    ledger = Ledger()
    ledger.plan.higher_is_better = True
    ledger.runs = [make_run("experiment_001", 0.70), make_run("experiment_002", 0.80)]
    ledger.evaluations = [
        Evaluation(experiment_id="experiment_001", verdict="approve"),
        Evaluation(experiment_id="experiment_002", verdict="approve"),
    ]

    assert ledger.best_approved_run().experiment_id == "experiment_002"


def test_best_approved_run_when_lower_is_better():
    ledger = Ledger()
    ledger.plan.higher_is_better = False  # for example RMSE
    ledger.runs = [make_run("experiment_001", 12.0), make_run("experiment_002", 9.5)]
    ledger.evaluations = [
        Evaluation(experiment_id="experiment_001", verdict="approve"),
        Evaluation(experiment_id="experiment_002", verdict="approve"),
    ]

    assert ledger.best_approved_run().experiment_id == "experiment_002"


def test_rejected_crashed_and_unevaluated_runs_are_never_best():
    ledger = Ledger()
    ledger.runs = [
        make_run("experiment_001", 0.60),  # approved, the only valid one
        make_run("experiment_002", 0.99),  # rejected by the Evaluator
        make_run("experiment_003", None, error="MemoryError"),  # crashed
        make_run("experiment_004", 0.95),  # never evaluated
    ]
    ledger.evaluations = [
        Evaluation(experiment_id="experiment_001", verdict="approve"),
        Evaluation(experiment_id="experiment_002", verdict="reject", reasons=["leakage"]),
        Evaluation(experiment_id="experiment_003", verdict="approve"),
    ]

    assert [run.experiment_id for run in ledger.approved_runs()] == ["experiment_001"]
    assert ledger.best_approved_run().experiment_id == "experiment_001"


def test_best_approved_run_is_none_without_approved_runs():
    assert Ledger().best_approved_run() is None


def test_recent_crashes_returns_latest_crashes_only():
    ledger = Ledger()
    ledger.runs = [
        make_run("experiment_001", None, error="first"),
        make_run("experiment_002", 0.7),
        make_run("experiment_003", None, error="second"),
        make_run("experiment_004", None, error="third"),
    ]

    crashes = ledger.recent_crashes(2)

    assert [run.error for run in crashes] == ["second", "third"]


def test_find_run_and_latest_evaluation():
    ledger = Ledger()
    ledger.runs = [make_run("experiment_001", 0.7)]
    ledger.evaluations = [
        Evaluation(experiment_id="experiment_001", verdict="reject"),
        Evaluation(experiment_id="experiment_001", verdict="approve"),
    ]

    assert ledger.find_run("experiment_001") is not None
    assert ledger.find_run("experiment_999") is None
    assert ledger.find_evaluation("experiment_001").verdict == "approve"
