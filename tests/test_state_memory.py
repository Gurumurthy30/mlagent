"""Tests for state.py and memory.py: invariants, serialization, queries, and crash safety."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hypothesis import given

from mlagent.memory import (
    LedgerLoadError,
    Workspace,
    create_new_ledger,
    load_ledger,
    save_ledger,
    write_progress_file,
)
from mlagent.state import (
    FeatureIdea,
    Ledger,
)
from tests.conftest import st_ledger


@pytest.mark.unit
@given(ledger=st_ledger())
def test_ledger_json_roundtrip(ledger: Ledger) -> None:
    """Any valid random ledger round-trips through JSON serialization unchanged."""
    raw_json = ledger.model_dump_json()
    loaded = Ledger.model_validate_json(raw_json)
    assert loaded == ledger


@pytest.mark.unit
@given(ledger=st_ledger())
def test_save_and_load_ledger_roundtrip(ledger: Ledger) -> None:
    """Any valid random ledger round-trips through save_ledger / load_ledger unchanged."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Workspace(Path(tmpdir) / "ws")
        ws.create_folders()

        save_ledger(ws, ledger)
        reloaded = load_ledger(ws)
        assert reloaded == ledger


@pytest.mark.unit
def test_create_new_ledger_initial_invariants() -> None:
    """New ledger starts in valid initial state."""
    ledger = create_new_ledger("Predict churn", "f1 and auc")
    assert ledger.user_task == "Predict churn"
    assert ledger.user_metrics_text == "f1 and auc"
    assert ledger.status == "running"
    assert ledger.phase == "data_engineer"
    assert ledger.iteration == 0
    assert ledger.current_feature_version == 0
    assert ledger.runs == []
    assert ledger.features == []
    assert ledger.reviews == []


@pytest.mark.unit
@given(ledger=st_ledger())
def test_best_approved_run_invariants(ledger: Ledger) -> None:
    """best_approved_run respects metric direction, ignores crashed/rejected runs."""
    approved = [r for r in ledger.runs if r.verdict == "approve" and r.score_mean is not None and not r.crashed]
    best = ledger.best_approved_run()

    if not approved:
        assert best is None
    else:
        assert best is not None
        assert best.verdict == "approve"
        assert not best.crashed
        assert best.score_mean is not None

        higher = ledger.brief.higher_is_better
        if higher:
            for r in approved:
                assert r.score_mean <= best.score_mean
        else:
            for r in approved:
                assert r.score_mean >= best.score_mean


@pytest.mark.unit
@given(ledger=st_ledger())
def test_ledger_queries_invariants(ledger: Ledger) -> None:
    """Verify definitions of unevaluated_runs, runs_this_round, round_had_gain, dead_ends, round_cards."""
    # unevaluated_runs
    uneval = ledger.unevaluated_runs()
    assert all(r.iteration == ledger.iteration and not r.crashed and r.verdict is None for r in uneval)

    # runs_this_round
    this_round = ledger.runs_this_round()
    assert all(r.iteration == ledger.iteration for r in this_round)

    # round_had_gain
    for n in range(1, ledger.iteration + 1):
        has_gain = ledger.round_had_gain(n)
        expected = any(r.iteration == n and r.counted_gain and r.verdict == "approve" for r in ledger.runs)
        assert has_gain == expected

    # dead_ends: at most 15, no duplicates
    dead = ledger.dead_ends()
    assert len(dead) <= 15
    assert len(dead) == len(set(dead))

    # round_cards: one per finished review
    cards = ledger.round_cards()
    assert len(cards) == len(ledger.reviews)

    # open_feature_ideas
    open_ideas = ledger.open_feature_ideas()
    assert all(isinstance(idea, FeatureIdea) for idea in open_ideas)


@pytest.mark.unit
def test_crash_safe_ledger_saving(tmp_path: Path) -> None:
    """Interrupted or corrupt write leaves a loadable previous ledger."""
    ws = Workspace(tmp_path / "ws")
    ws.create_folders()

    # Save initial version 1
    ledger1 = create_new_ledger("Initial task", "f1")
    ledger1.iteration = 1
    save_ledger(ws, ledger1)
    assert load_ledger(ws).iteration == 1

    # Save updated version 2
    ledger2 = create_new_ledger("Updated task", "f1")
    ledger2.iteration = 2
    save_ledger(ws, ledger2)
    assert load_ledger(ws).iteration == 2

    # Verify backup exists
    assert ws.previous_ledger_path.exists()

    # Simulate catastrophic corruption of the active ledger.json (e.g. truncated JSON)
    ws.ledger_path.write_text('{"user_task": "Corrupted partial write...', encoding="utf-8")

    # load_ledger should safely fall back to previous_ledger_path
    recovered = load_ledger(ws)
    assert recovered.iteration == 1
    assert recovered.user_task == "Initial task"


@pytest.mark.unit
def test_load_ledger_raises_when_no_ledger_exists(tmp_path: Path) -> None:
    """load_ledger raises LedgerLoadError when neither primary nor backup exists."""
    ws = Workspace(tmp_path / "ws")
    ws.create_folders()
    with pytest.raises(LedgerLoadError, match="No usable saved ledger"):
        load_ledger(ws)


@pytest.mark.unit
def test_write_progress_file(tmp_path: Path) -> None:
    """write_progress_file atomically updates progress.md."""
    ws = Workspace(tmp_path / "ws")
    ws.create_folders()
    progress_text = "# Progress\nRound 1 complete."
    write_progress_file(ws, progress_text)
    assert ws.progress_path.exists()
    content = ws.progress_path.read_text(encoding="utf-8")
    assert "Round 1 complete." in content


@pytest.mark.unit
def test_archive_existing_normal_rename(tmp_path: Path) -> None:
    """archive_existing renames the workspace folder to an archive path."""
    ws = Workspace(tmp_path)
    ws.create_folders()
    ws.config_path.write_text("dummy_config: true\n", encoding="utf-8")
    save_ledger(ws, create_new_ledger("Test task", "acc"))

    archive_path = ws.archive_existing()
    assert archive_path.exists()
    assert (archive_path / "config.yaml").exists()
    assert (archive_path / "memory" / "ledger.json").exists()


@pytest.mark.unit
def test_archive_existing_permission_error_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When rename raises PermissionError (Windows directory lock), archive_existing copies and clears."""
    ws = Workspace(tmp_path)
    ws.create_folders()
    ws.config_path.write_text("important_old_config: true\n", encoding="utf-8")
    save_ledger(ws, create_new_ledger("Test task", "acc"))

    # Simulate Windows PermissionError: [WinError 5] Access is denied on rename
    def mock_rename(self: Any, target: Any) -> None:
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "rename", mock_rename)

    archive_path = ws.archive_existing()
    assert archive_path.exists()
    # Archive has the old files preserved
    assert (archive_path / "config.yaml").exists()
    assert (archive_path / "memory" / "ledger.json").exists()
    # Workspace root has been cleared for the fresh run
    assert not ws.has_saved_ledger()


