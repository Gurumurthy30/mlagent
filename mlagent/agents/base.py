"""Shared pieces for every agent.

AgentContext: the tools an agent uses (LLM, code runner, events, config)
AgentFailedError: raised when an agent cannot do its job
facts_to_text: formats script result JSON for prompts, cutting only at item boundaries
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlagent.code_runner import CodeRunner
    from mlagent.config import AppConfig
    from mlagent.environment_probe import EnvironmentInfo
    from mlagent.events import EventBus
    from mlagent.llm_client import LlmClient
    from mlagent.memory import Workspace


class AgentFailedError(Exception):
    """Raised when an agent cannot do its job. The run stops with this message."""


@dataclass
class AgentContext:
    config: AppConfig
    workspace: Workspace
    event_bus: EventBus
    llm_client: LlmClient
    code_runner: CodeRunner
    environment: EnvironmentInfo


def facts_to_text(result_data: dict, max_chars: int = 5000) -> str:
    """Format dictionary into text for prompts, cutting only at item/key boundaries."""
    if not isinstance(result_data, dict):
        text = str(result_data)
        if len(text) <= max_chars:
            return text
        last_nl = text[:max_chars].rfind("\n")
        cutoff = last_nl if last_nl > max_chars // 2 else max_chars
        return text[:cutoff] + "\n...(remaining omitted)"

    lines: list[str] = []
    current_chars = 0
    for key, val in result_data.items():
        val_str = json.dumps(val, indent=1, ensure_ascii=False, default=str)
        item_text = f"{key}: {val_str}"
        item_len = len(item_text) + 1
        if lines and current_chars + item_len > max_chars:
            lines.append("... [remaining items omitted]")
            break
        lines.append(item_text)
        current_chars += item_len

    return "\n".join(lines)
