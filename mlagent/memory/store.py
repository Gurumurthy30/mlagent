"""Persistent disk-backed store for typed memory cards.

Provides append-only persistence under workspace/memory/cards/{tag}/{card_id}.json.
Enforces the linear Champion ratchet by making ChampionPointerCard the only mutable record.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from mlagent.memory.cards import (
    TAG_TO_CARD_CLS,
    ChampionPointerCard,
    LessonLearnedCard,
    MemoryCard,
)


class MemoryStore:
    """Manages append-only memory card persistence and archival."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.memory_root = self.workspace_root / "memory"
        self.cards_dir = self.memory_root / "cards"
        self.archive_dir = self.memory_root / "archive"

        # Initialize base folders
        self.cards_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def _tag_dir(self, tag: str) -> Path:
        d = self.cards_dir / tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _archive_tag_dir(self, tag: str) -> Path:
        d = self.archive_dir / tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _atomic_write_json(self, target_path: Path, data_str: str) -> None:
        # Atomic file write with Windows file-locking retry
        temp_path = target_path.with_suffix(f".tmp_{os.getpid()}_{time.time_ns()}")
        temp_path.write_text(data_str, encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(temp_path, target_path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        # Final fallback
        shutil.move(str(temp_path), str(target_path))

    def append_card(self, card: MemoryCard) -> str:
        # Append new immutable card to disk. Raises if card_id exists.
        tag_dir = self._tag_dir(card.tag)
        target_file = tag_dir / f"{card.card_id}.json"

        if target_file.exists():
            raise FileExistsError(
                f"Memory card '{card.card_id}' already exists under tag '{card.tag}'. "
                "Cards are strictly append-only."
            )

        payload_json = card.model_dump_json(indent=2)
        self._atomic_write_json(target_file, payload_json)
        return str(target_file)

    def update_champion_pointer(self, card: ChampionPointerCard) -> str:
        # Single mutable record: atomically updates current.json
        tag_dir = self._tag_dir("champion_pointer")
        current_file = tag_dir / "current.json"

        # Also store permanent historical snapshot
        history_file = tag_dir / f"{card.card_id}.json"
        card_json = card.model_dump_json(indent=2)

        if not history_file.exists():
            self._atomic_write_json(history_file, card_json)

        # In-place atomic update of active champion pointer
        self._atomic_write_json(current_file, card_json)
        return str(current_file)

    def load_cards(
        self,
        tag: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryCard]:
        # Load cards sorted newest first
        tag_dir = self.cards_dir / tag
        if not tag_dir.exists():
            return []

        card_cls = TAG_TO_CARD_CLS.get(tag, MemoryCard)
        cards: list[MemoryCard] = []

        # Find all JSON card files, excluding current.json pointer
        file_paths = [p for p in tag_dir.glob("*.json") if p.name != "current.json"]

        for file_path in file_paths:
            try:
                raw_json = file_path.read_text(encoding="utf-8")
                card = card_cls.model_validate_json(raw_json)

                # Apply matching filters if provided
                if filters:
                    matched = True
                    for key, val in filters.items():
                        if getattr(card, key, None) != val:
                            matched = False
                            break
                    if not matched:
                        continue

                cards.append(card)
            except Exception:
                continue

        # Sort newest-first (by iteration desc, then created_at desc)
        cards.sort(key=lambda c: (c.iteration, c.created_at), reverse=True)
        return cards[:limit]

    def load_latest(self, tag: str) -> MemoryCard | None:
        # Special case for champion pointer
        if tag == "champion_pointer":
            current_file = self.cards_dir / "champion_pointer" / "current.json"
            if current_file.exists():
                try:
                    return ChampionPointerCard.model_validate_json(
                        current_file.read_text(encoding="utf-8")
                    )
                except Exception:
                    pass

        cards = self.load_cards(tag=tag, limit=1)
        return cards[0] if cards else None

    def archive_old_cards(self, tag: str, keep: int = 50) -> int:
        # Move cards exceeding retention limit to archive directory
        tag_dir = self.cards_dir / tag
        if not tag_dir.exists():
            return 0

        file_paths = [p for p in tag_dir.glob("*.json") if p.name != "current.json"]
        if len(file_paths) <= keep:
            return 0

        # Sort by file modification time (oldest first)
        file_paths.sort(key=lambda p: p.stat().st_mtime)
        to_move = file_paths[: len(file_paths) - keep]
        archive_tag_dir = self._archive_tag_dir(tag)

        moved_count = 0
        for p in to_move:
            dest = archive_tag_dir / p.name
            try:
                shutil.move(str(p), str(dest))
                moved_count += 1
            except Exception:
                pass

        return moved_count

    def promote_lesson_to_core(self, lesson_card: LessonLearnedCard, reference_count: int) -> None:
        # Elevate priority of validated repeated empirical lessons
        if reference_count >= 2 and lesson_card.priority != "core":
            lesson_card.priority = "core"
            tag_dir = self._tag_dir(lesson_card.tag)
            target_file = tag_dir / f"{lesson_card.card_id}.json"
            self._atomic_write_json(target_file, lesson_card.model_dump_json(indent=2))
