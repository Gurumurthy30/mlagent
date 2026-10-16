"""Global pytest configuration, hooks, fixtures, and Hypothesis strategies for mlagent."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, settings

from mlagent.config import AppConfig
from mlagent.events import EventBus
from mlagent.memory import Workspace
from mlagent.state import (
    EdaReport,
    ExperimentPlan,
    ExperimentRun,
    FeatureRecord,
    FileInfo,
    FinalResult,
    Finding,
    Ledger,
    ProjectBrief,
    Review,
    Verdict,
)

# --- Seed & Hypothesis Configuration ---

SESSION_SEED: int = 42


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--mlagent-seed",
        action="store",
        default=None,
        type=int,
        help="Session seed for reproducible randomness in tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    global SESSION_SEED
    cli_seed = config.getoption("--mlagent-seed")
    env_seed = os.environ.get("MLAGENT_TEST_SEED")

    if cli_seed is not None:
        SESSION_SEED = cli_seed
    elif env_seed:
        try:
            SESSION_SEED = int(env_seed)
        except ValueError:
            SESSION_SEED = random.randint(1, 1_000_000)
    else:
        SESSION_SEED = random.randint(1, 1_000_000)

    random.seed(SESSION_SEED)
    try:
        import numpy as np
        np.random.seed(SESSION_SEED % (2**32 - 1))
    except ImportError:
        pass

    # Hypothesis profiles: dev (50), ci (100), thorough (500)
    settings.register_profile("dev", max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    settings.register_profile("ci", max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    settings.register_profile("thorough", max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])

    profile_name = os.environ.get("HYPOTHESIS_PROFILE", "dev")
    settings.load_profile(profile_name)


def pytest_report_header(config: pytest.Config) -> list[str]:
    profile_name = os.environ.get("HYPOTHESIS_PROFILE", "dev")
    return [
        f"mlagent session seed: {SESSION_SEED} (reproduce with --mlagent-seed={SESSION_SEED})",
        f"Hypothesis profile: {profile_name}",
    ]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Any:
    outcome = yield
    report = outcome.get_result()
    if report.failed:
        report.sections.append((
            "mlagent Reproducibility Info",
            f"Seed: {SESSION_SEED}\nReproduce with: pytest --mlagent-seed={SESSION_SEED} {item.nodeid}",
        ))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("MLAGENT_LIVE") != "1":
        skip_live = pytest.mark.skip(reason="Live Ollama tests skipped (set MLAGENT_LIVE=1 to run)")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


# --- Shared Fixtures ---

@pytest.fixture
def session_seed() -> int:
    return SESSION_SEED


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def default_config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path)
    ws.create_folders()
    return ws


# --- Fake LLM for Integration & Pipeline Tests ---

class FakeLlmClient:
    """Deterministic, schema-aware fake LLM client for tests."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    def check_ready(self) -> None:
        pass

    def ask_text(self, agent_name: str, prompt: str) -> str:
        self.calls.append({"type": "text", "agent": agent_name, "prompt": prompt})
        if agent_name in self.responses:
            res = self.responses[agent_name]
            if isinstance(res, str):
                return res
            if callable(res):
                return str(res(prompt))
        return (
            "## In brief\n"
            "Baseline LightGBM trained with 5-fold CV achieved 0.8520 F1 score.\n\n"
            "## Exploration\n"
            "Dataset has 1000 rows and 10 features.\n\n"
            "## Validation\n"
            "Stratified 5-fold cross validation.\n\n"
            "## What worked\n"
            "Feature scaling and tree depth tuning.\n\n"
            "## Dead ends\n"
            "Target encoding showed signs of leakage.\n\n"
            "## Verification\n"
            "Validation checks passed with zero overlap.\n\n"
            "## Future work\n"
            "Explore ensemble with neural net.\n"
        )

    def ask_structured(self, agent_name: str, prompt: str, schema: type[Any]) -> Any:
        self.calls.append({"type": "structured", "agent": agent_name, "prompt": prompt, "schema": schema})
        if agent_name in self.responses:
            val = self.responses[agent_name]
            if isinstance(val, schema):
                return val
            if isinstance(val, dict):
                return schema.model_validate(val)
            if callable(val):
                import inspect
                sig = inspect.signature(val)
                res = val(prompt, schema) if len(sig.parameters) >= 2 else val(prompt)
                if isinstance(res, schema):
                    return res
                return schema.model_validate(res)

        # Generate a minimal valid schema instance dynamically
        return self._make_default_instance(schema)

    def _make_default_instance(self, schema: type[Any]) -> Any:
        fields = getattr(schema, "model_fields", {})
        data: dict[str, Any] = {}
        for fname, finfo in fields.items():
            if finfo.default is not None and not str(finfo.default).startswith("PydanticUndefined"):
                data[fname] = finfo.default
            elif finfo.default_factory is not None:
                data[fname] = finfo.default_factory()
            else:
                ann = finfo.annotation
                data[fname] = self._sample_value_for_type(ann, fname)
        try:
            return schema.model_validate(data)
        except Exception:
            return schema.model_construct(**data)

    def _sample_value_for_type(self, t: Any, name: str) -> Any:
        s = str(t)
        if "str" in s:
            return f"sample_{name}"
        if "int" in s:
            return 1
        if "float" in s:
            return 0.5
        if "bool" in s:
            return True if "done" in name else False
        if "list" in s or "Sequence" in s:
            return []
        if "dict" in s or "Mapping" in s:
            return {}
        return None


# --- Hypothesis Strategies for Generating Valid State Objects ---

@st.composite
def st_project_brief(draw: Any) -> ProjectBrief:
    return ProjectBrief(
        summary=draw(st.text(min_size=0, max_size=100)),
        modality=draw(st.sampled_from(["tabular", "image", "text", "audio"])),
        task_family=draw(st.sampled_from(["classification", "regression", "clustering"])),
        target=draw(st.text(min_size=1, max_size=20)),
        id_column=draw(st.text(min_size=0, max_size=20)),
        primary_metric=draw(st.sampled_from(["f1", "auc", "rmse", "accuracy", "logloss"])),
        higher_is_better=draw(st.booleans()),
        files=[
            FileInfo(path="train.csv", size_mb=1.5, kind="table", role="train", columns=["id", "feat", "target"]),
            FileInfo(path="test.csv", size_mb=0.8, kind="table", role="test", columns=["id", "feat"]),
        ],
    )


@st.composite
def st_finding(draw: Any) -> Finding:
    fid = f"find_{draw(st.integers(min_value=1, max_value=999)):03d}"
    return Finding(
        finding_id=fid,
        statement=draw(st.text(min_size=1, max_size=50)),
        evidence=draw(st.text(min_size=0, max_size=50)),
        implication=draw(st.text(min_size=0, max_size=50)),
        priority=draw(st.sampled_from(["high", "medium", "low"])),
    )


@st.composite
def st_feature_record(draw: Any) -> FeatureRecord:
    fid = f"feature_{draw(st.integers(min_value=1, max_value=999)):03d}"
    return FeatureRecord(
        feature_id=fid,
        name=draw(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_")),
        family=draw(st.sampled_from(["foundation", "ratio", "interaction", "datetime", "aggregation"])),
        reason=draw(st.text(min_size=0, max_size=40)),
        finding_id=f"find_{draw(st.integers(min_value=1, max_value=20)):03d}",
        kept=draw(st.booleans()),
        score_gain=draw(st.one_of(st.none(), st.floats(min_value=-0.5, max_value=0.5, allow_nan=False))),
    )


@st.composite
def st_experiment_run(draw: Any) -> ExperimentRun:
    eid = f"run_{draw(st.integers(min_value=1, max_value=999)):03d}"
    is_crashed = draw(st.booleans())
    verdict: Verdict | None = draw(st.one_of(st.none(), st.sampled_from(["approve", "reject"]))) if not is_crashed else None
    scores = (
        draw(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=2, max_size=5))
        if not is_crashed
        else []
    )
    mean_val = sum(scores) / len(scores) if scores else None
    return ExperimentRun(
        experiment_id=eid,
        iteration=draw(st.integers(min_value=1, max_value=20)),
        model_family=draw(st.sampled_from(["lightgbm", "xgboost", "catboost", "random_forest", "linear"])),
        feature_version=draw(st.integers(min_value=1, max_value=5)),
        setup_signature=f"sig_{eid}",
        summary=draw(st.text(min_size=0, max_size=30)),
        score_mean=mean_val,
        score_std=0.01 if mean_val is not None else None,
        split_scores=scores,
        holdout_score=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False))) if not is_crashed else None,
        verdict=verdict,
        verdict_reasons=[draw(st.text(min_size=1, max_size=30))] if verdict == "reject" else [],
        error=draw(st.text(min_size=1, max_size=50)) if is_crashed else "",
        gain_vs_best=draw(st.one_of(st.none(), st.floats(min_value=-0.2, max_value=0.2, allow_nan=False))),
        counted_gain=draw(st.booleans()) if verdict == "approve" else False,
    )


@st.composite
def st_ledger(draw: Any) -> Ledger:
    brief = draw(st_project_brief())
    runs = draw(st.lists(st_experiment_run(), min_size=0, max_size=10))
    features = draw(st.lists(st_feature_record(), min_size=0, max_size=10))
    findings = draw(st.lists(st_finding(), min_size=0, max_size=5))

    reviews = [
        Review(
            iteration=i,
            focus=draw(st.sampled_from(["features", "model", "tune", "finish"])),
            reviewed_focus=draw(st.sampled_from(["start", "features", "model", "tune"])),
            reason="test reason",
            lesson="test lesson",
            instructions=["instruction 1"],
        )
        for i in range(1, draw(st.integers(min_value=1, max_value=4)))
    ]

    from typing import get_args

    from mlagent.state import Phase

    return Ledger(
        user_task=draw(st.text(min_size=1, max_size=100)),
        user_metrics_text="f1",
        status=draw(st.sampled_from(["running", "done", "failed"])),
        phase=draw(st.sampled_from(list(get_args(Phase)))),
        iteration=draw(st.integers(min_value=1, max_value=10)),
        brief=brief,
        eda=EdaReport(
            summary="test EDA summary",
            findings=findings,
            open_questions=["question 1"],
            eda_loops_used=draw(st.integers(min_value=1, max_value=4)),
        ),
        plan=ExperimentPlan(
            target_metric=brief.primary_metric,
            higher_is_better=brief.higher_is_better,
            split_strategy="kfold",
            n_splits=5,
        ),
        features=features,
        current_feature_version=draw(st.integers(min_value=0, max_value=3)),
        runs=runs,
        reviews=reviews,
        final=FinalResult(headline="test headline", best_experiment_id="run_001") if runs else FinalResult(),
    )
