"""Tests for the event system: EventBus, AgentEvent, and ConsoleUI rendering invariants."""

from __future__ import annotations

import pytest
from rich.console import Console

from mlagent.console_ui import ConsoleUI
from mlagent.events import AgentEvent, EventBus, EventType


@pytest.mark.unit
@pytest.mark.parametrize("event_type", list(EventType))
def test_event_bus_emits_every_event_type(event_type: EventType) -> None:
    """EventBus emits every EventType successfully to subscribers."""
    bus = EventBus()
    received: list[AgentEvent] = []

    bus.subscribe(received.append)
    test_message = f"Testing event: {event_type.value}"
    bus.emit(event_type, "TestAgent", test_message, key="val", count=42)

    assert len(received) == 1
    event = received[0]
    assert event.event_type == event_type
    assert event.agent_name == "TestAgent"
    assert event.message == test_message
    assert event.data == {"key": "val", "count": 42}


@pytest.mark.unit
@pytest.mark.parametrize("event_type", list(EventType))
def test_console_ui_renders_markup_characters_without_raising_or_losing_text(
    event_type: EventType,
) -> None:
    """Messages containing Rich markup like [red], [bold], [/] do not crash and preserve text."""
    console = Console(record=True, width=120)
    ui = ConsoleUI(console=console)
    bus = EventBus()
    bus.subscribe(ui.handle_event)

    tricky_messages = [
        "Normal message without brackets.",
        "Message with [red]red tag[/red] and [bold]bold text[/bold].",
        "Mismatched [brackets [and] nested [unclosed tags",
        "Path like C:\\Users\\[test_dir]\\data.csv [dim]status[/dim]",
        "[cyan]Leading tag with [yellow]nested tags[/yellow] and trailing [[double]]",
    ]

    for msg in tricky_messages:
        bus.emit(event_type, "Modeler", msg)

    exported_text = console.export_text()
    for msg in tricky_messages:
        # In rich.text.Text.append(msg), markup is not parsed, so the raw text is preserved verbatim
        assert msg in exported_text


@pytest.mark.unit
def test_event_bus_broken_listener_does_not_crash_emit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing event listener logs to stderr but never crashes or interrupts emit."""
    bus = EventBus()
    call_log: list[str] = []

    def failing_listener(event: AgentEvent) -> None:
        raise RuntimeError("Simulated UI crash!")

    def working_listener(event: AgentEvent) -> None:
        call_log.append(event.message)

    bus.subscribe(failing_listener)
    bus.subscribe(working_listener)

    # emit should not raise despite the first listener raising
    bus.emit(EventType.INFO, "SafeAgent", "Safe message")

    assert call_log == ["Safe message"]
