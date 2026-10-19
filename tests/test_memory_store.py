"""Unit tests for MemoryStore persistence, atomicity, and archival."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mlagent.memory.cards import (
    ChampionPointerCard,
    EdaFindingCard,
    ExperimentRunCard,
    LessonLearnedCard,
)
from mlagent.memory.store import MemoryStore


def test_append_card_collision_raises(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    card = EdaFindingCard(
        card_id="finding_001",
        owner="data_analyst",
        iteration=1,
        memory_summary="Finding 1",
        payload={"data": 123},
    )

    path1 = store.append_card(card)
    assert Path(path1).exists()

    # Second write with same card_id must raise FileExistsError
    with pytest.raises(FileExistsError):
        store.append_card(card)


def test_load_cards_newest_first(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)

    for i in range(1, 6):
        card = ExperimentRunCard(
            card_id=f"exp_{i:03d}",
            owner="modeler",
            iteration=i,
            memory_summary=f"Run iteration {i}",
            payload={"run_index": i},
        )
        store.append_card(card)

    loaded = store.load_cards("experiment_run", limit=3)
    assert len(loaded) == 3
    # Newest first: iteration 5, 4, 3
    assert [c.iteration for c in loaded] == [5, 4, 3]
    assert [c.card_id for c in loaded] == ["exp_005", "exp_004", "exp_003"]


def test_update_champion_pointer_is_atomic(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)

    initial_champ = ChampionPointerCard(
        card_id="champ_init",
        owner="evaluator",
        iteration=1,
        memory_summary="Initial champion",
        payload={"score": 0.85},
    )
    store.update_champion_pointer(initial_champ)

    current_file = tmp_path / "memory" / "cards" / "champion_pointer" / "current.json"
    assert current_file.exists()
    assert "champ_init" in current_file.read_text(encoding="utf-8")

    # Simulate crash/failure during atomic replace of next champion
    new_champ = ChampionPointerCard(
        card_id="champ_v2",
        owner="evaluator",
        iteration=2,
        memory_summary="New candidate champion",
        payload={"score": 0.92},
    )

    with patch("os.replace", side_effect=OSError("Disk write simulated failure")):
        with patch("shutil.move", side_effect=OSError("Disk write simulated failure")):
            with pytest.raises(OSError):
                store.update_champion_pointer(new_champ)

    # Verify old champion pointer survives intact
    surviving_text = current_file.read_text(encoding="utf-8")
    assert "champ_init" in surviving_text


def test_archive_old_cards_moves_excess(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)

    # Write 10 cards
    for i in range(1, 11):
        card = EdaFindingCard(
            card_id=f"f_{i:03d}",
            owner="data_analyst",
            iteration=i,
            memory_summary=f"Summary for finding {i}",
            payload={"finding_idx": i},
        )
        store.append_card(card)

    tag_dir = tmp_path / "memory" / "cards" / "eda_finding"
    archive_dir = tmp_path / "memory" / "archive" / "eda_finding"
    assert len(list(tag_dir.glob("*.json"))) == 10

    # Archive keeping 6
    moved = store.archive_old_cards("eda_finding", keep=6)
    assert moved == 4
    assert len(list(tag_dir.glob("*.json"))) == 6
    assert len(list(archive_dir.glob("*.json"))) == 4


def test_promote_lesson_to_core(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    lesson = LessonLearnedCard(
        card_id="lesson_1",
        owner="ml_lead",
        iteration=1,
        memory_summary="Regularization prevents tree overfit",
        payload={"text": "Regularization"},
        priority="high",
    )
    store.append_card(lesson)

    # Reference count < 2: remains high
    store.promote_lesson_to_core(lesson, reference_count=1)
    assert lesson.priority == "high"

    # Reference count >= 2: promoted to core
    store.promote_lesson_to_core(lesson, reference_count=2)
    assert lesson.priority == "core"

    # Verify saved file on disk reflects core priority
    reloaded = store.load_latest("lesson_learned")
    assert reloaded is not None
    assert reloaded.priority == "core"
