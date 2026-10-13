import io

from rich.console import Console

from mlagent.console_ui import ConsoleUI
from mlagent.events import EventBus, EventType


def test_emit_reaches_every_listener():
    received_events = []
    event_bus = EventBus()
    event_bus.subscribe(received_events.append)
    event_bus.subscribe(received_events.append)

    event_bus.emit(EventType.INFO, "Data Analyst", "EDA round 1 started", round_number=1)

    assert len(received_events) == 2
    assert received_events[0].agent_name == "Data Analyst"
    assert received_events[0].data == {"round_number": 1}


def test_failing_listener_does_not_stop_other_listeners(capsys):
    received_events = []
    event_bus = EventBus()

    def broken_listener(event):
        raise RuntimeError("display crashed")

    event_bus.subscribe(broken_listener)
    event_bus.subscribe(received_events.append)

    event_bus.emit(EventType.INFO, "Modeler", "training")

    assert len(received_events) == 1
    assert "display crashed" in capsys.readouterr().err


def test_console_ui_prints_agent_name_and_message_without_reading_markup():
    output_buffer = io.StringIO()
    console_ui = ConsoleUI(Console(file=output_buffer, width=120, force_terminal=False))
    event_bus = EventBus()
    event_bus.subscribe(console_ui.handle_event)

    event_bus.emit(EventType.DECISION, "Reviewer", "next focus [features] scores [0.7, 0.8]")

    printed_text = output_buffer.getvalue()
    assert "Reviewer" in printed_text
    assert "DECIDE" in printed_text
    assert "[features] scores [0.7, 0.8]" in printed_text


def test_console_ui_has_a_label_for_every_event_type():
    output_buffer = io.StringIO()
    console_ui = ConsoleUI(Console(file=output_buffer, width=120, force_terminal=False))
    event_bus = EventBus()
    event_bus.subscribe(console_ui.handle_event)

    for event_type in EventType:
        event_bus.emit(event_type, "Agent", "message")

    assert output_buffer.getvalue().count("Agent") == len(EventType)
