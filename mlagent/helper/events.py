"""Events: how agents report progress.

An agent calls event_bus.emit(...). Listeners format or log events as needed.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class EventType(StrEnum):
    AGENT_STARTED = "agent_started"
    AGENT_FINISHED = "agent_finished"
    INFO = "info"
    DECISION = "decision"
    CODE_RUN = "code_run"
    WARNING = "warning"
    ERROR = "error"
    RESULT = "result"
    STATUS = "status"


@dataclass
class AgentEvent:
    event_type: EventType
    agent_name: str
    message: str
    data: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


EventListener = Callable[[AgentEvent], None]


class EventBus:
    def __init__(self) -> None:
        self._listeners: list[EventListener] = []

    def subscribe(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    def emit(
        self,
        event_type: EventType,
        agent_name: str,
        message: str,
        data: dict | None = None,
        **kwargs,
    ) -> None:
        payload = dict(data) if data is not None else {}
        if kwargs:
            payload.update(kwargs)
        event = AgentEvent(event_type, agent_name, message, payload)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as listener_error:
                # A broken display must never stop the ML run.
                print(f"Event listener failed: {listener_error}", file=sys.stderr)
