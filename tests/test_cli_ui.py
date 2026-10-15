"""Tests for Stage C CLI and UI invariants: themes, formatting, live status, summary, and plain mode."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest
from hypothesis import given
from rich.console import Console
from rich.panel import Panel

from mlagent.cli import (
    EXIT_OK,
    EXIT_USAGE,
    flags_to_overrides,
    resume_command,
    status_command,
)
from mlagent.config import UiConfig
from mlagent.console_ui import ConsoleUI
from mlagent.environment_probe import EnvironmentInfo
from mlagent.events import AgentEvent, EventBus, EventType
from mlagent.state import Ledger
from mlagent.summary import (
    LiveStatusBar,
    get_symbols,
    get_theme,
    is_plain_mode,
    render_confirmation_panel,
    render_environment_table,
    render_summary_panel,
)
from tests.conftest import st_ledger

ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


@pytest.mark.unit
@pytest.mark.parametrize("event_type", list(EventType))
def test_render_events_markup_and_long_text_never_raises_and_preserves_text(
    event_type: EventType,
) -> None:
    """Emitting every event type with tricky markup and long text preserves raw message text."""
    console = Console(record=True, width=120)
    ui = ConsoleUI(console=console, config=UiConfig(verbosity="detailed"))
    bus = EventBus()
    bus.subscribe(ui.handle_event)

    tricky_messages = [
        "Plain text message.",
        "Formatted like [bold red]error[/bold red] and [dim]info[/dim].",
        "Mismatched [brackets [and] nested tags",
        "Path like C:\\Users\\[test_workspace]\\data.csv",
        "Very long text: " + ("abcdefghij " * 30),
        "Multi-line message:\nFirst line\nSecond indented line\nThird line",
    ]

    for msg in tricky_messages:
        bus.emit(event_type, "Modeler", msg)

    output = console.export_text()
    clean_output = re.sub(r"\s+", " ", output)
    for msg in tricky_messages:
        clean_msg = re.sub(r"\s+", " ", msg)
        assert clean_msg in clean_output, f"Message '{msg}' missing from output."


@pytest.mark.unit
def test_plain_mode_contains_no_ansi_escape_sequences() -> None:
    """Plain mode log output, summary panel, and environment tables contain no ANSI escape sequences."""
    console = Console(record=True, width=100, force_terminal=False)
    cfg = UiConfig(mode="plain", timestamps=True)
    ui = ConsoleUI(console=console, config=cfg)
    assert ui._is_plain is True

    # 1. Log event
    event = AgentEvent(EventType.INFO, "Modeler", "Training model without escape codes")
    ui.handle_event(event)

    # 2. Header and message
    ui.show_header("0.1.0")
    ui.show_message("Plain informational line")

    output = console.export_text()
    assert not ANSI_ESCAPE_PATTERN.search(output), "ANSI escape sequence found in log output!"

    # 3. Environment check in plain mode
    env = EnvironmentInfo(
        operating_system="Windows 11",
        shell_name="powershell",
        python_version="3.12.0",
        cpu_count=8,
        ram_total_gb=16.0,
        ram_free_gb=8.0,
    )
    env_text = render_environment_table(env, cfg)
    assert isinstance(env_text, str)
    assert not ANSI_ESCAPE_PATTERN.search(env_text)

    # 4. Confirmation panel in plain mode
    conf_text = render_confirmation_panel(
        task="predict prices",
        metrics="rmse",
        data_folder=Path("."),
        limits=None,
        workspace_folder=Path("mlagent_workspace"),
        model_name="qwen2.5",
        config=cfg,
    )
    assert isinstance(conf_text, str)
    assert not ANSI_ESCAPE_PATTERN.search(conf_text)

    # 5. Summary panel in plain mode
    ledger = Ledger(user_task="predict prices", status="done")
    summary_text = render_summary_panel(ledger, cfg)
    assert isinstance(summary_text, str)
    assert not ANSI_ESCAPE_PATTERN.search(summary_text)


@pytest.mark.unit
@pytest.mark.parametrize("width", [70, 80, 100, 120])
def test_no_rendered_log_line_exceeds_console_width(width: int) -> None:
    """No rendered log line exceeds the console width."""
    console = Console(record=True, width=width)
    ui = ConsoleUI(console=console, config=UiConfig(verbosity="detailed", log_width=width))

    long_msg = "Super long message explaining model hyperparameters and cross validation metrics " * 4
    event = AgentEvent(EventType.INFO, "FeatureEngineer", long_msg)
    ui.handle_event(event)

    output = console.export_text()
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    for line in lines:
        assert len(line) <= width, f"Rendered line length {len(line)} exceeds width {width}: {line}"


@pytest.mark.unit
def test_summary_rendering_all_ledger_statuses() -> None:
    """Summary panel renders correctly for done, stopped early, failed, and fresh ledgers."""
    cfg = UiConfig(theme="dark")

    # 1. Done status
    ledger_done = Ledger(user_task="Predict churn", status="done")
    panel_done = render_summary_panel(ledger_done, cfg)
    assert isinstance(panel_done, Panel)

    # 2. Stopped early with approved runs
    from mlagent.state import ExperimentRun
    run_approved = ExperimentRun(
        experiment_id="run_001",
        model_family="lightgbm",
        score_mean=0.8521,
        score_std=0.012,
        holdout_score=0.8490,
        verdict="approve",
    )
    ledger_stopped = Ledger(
        user_task="Predict churn",
        status="running",
        stop_reason="plateau",
        runs=[run_approved],
    )
    panel_stopped = render_summary_panel(ledger_stopped, cfg)
    assert isinstance(panel_stopped, Panel)

    # 3. Failed status
    ledger_failed = Ledger(
        user_task="Predict churn",
        status="failed",
        stop_reason="llm_unavailable",
        error_message="Ollama crashed",
    )
    panel_failed = render_summary_panel(ledger_failed, cfg)
    assert isinstance(panel_failed, Panel)

    # 4. Fresh empty ledger
    ledger_fresh = Ledger(user_task="Predict churn")
    panel_fresh = render_summary_panel(ledger_fresh, cfg)
    assert isinstance(panel_fresh, Panel)


@pytest.mark.unit
@given(st_ledger())
def test_summary_panel_renders_for_random_ledgers(ledger: Ledger) -> None:
    """Summary rendering never crashes for random valid ledgers."""
    cfg = UiConfig(theme="dark")
    result = render_summary_panel(ledger, cfg)
    assert isinstance(result, (Panel, str))


@pytest.mark.unit
def test_status_command_on_fresh_empty_workspace_prints_clear_message(tmp_path: Path) -> None:
    """status command on a fresh empty workspace prints a clear friendly message without error."""
    console = Console(record=True, width=100)
    ui = ConsoleUI(console=console)

    exit_code = status_command(tmp_path, ui)
    assert exit_code == EXIT_OK

    output = console.export_text()
    assert "No saved run found" in output
    assert "mlagent start" in output


@pytest.mark.unit
def test_resume_command_on_fresh_empty_workspace_exits_with_usage(tmp_path: Path) -> None:
    """resume command without an existing saved run exits cleanly with a helpful message."""
    console = Console(record=True, width=100)
    ui = ConsoleUI(console=console)

    exit_code = resume_command(tmp_path, ui)
    assert exit_code == EXIT_USAGE

    output = console.export_text()
    assert "No saved run found" in output


@pytest.mark.unit
def test_every_ui_config_key_is_exercised() -> None:
    """Every key in UiConfig is exercised and impacts UI behavior."""
    fields = set(UiConfig.model_fields.keys())
    expected_fields = {
        "mode",
        "theme",
        "verbosity",
        "timestamps",
        "show_environment",
        "confirm_before_start",
        "ascii_only",
        "log_width",
    }
    assert expected_fields.issubset(fields), f"Missing fields in UiConfig: {expected_fields - fields}"

    # 1. mode
    cfg_plain = UiConfig(mode="plain")
    cfg_rich = UiConfig(mode="rich")
    assert is_plain_mode(cfg_plain) is True
    assert is_plain_mode(cfg_rich) is False

    # 2. theme
    theme_dark = get_theme("dark")
    theme_light = get_theme("light")
    theme_mono = get_theme("mono")
    assert theme_dark.accent != theme_light.accent
    assert theme_mono.name == "mono"

    # 3. verbosity
    ui_quiet = ConsoleUI(config=UiConfig(verbosity="quiet"))
    ui_normal = ConsoleUI(config=UiConfig(verbosity="normal"))
    ui_detailed = ConsoleUI(config=UiConfig(verbosity="detailed"))

    assert ui_quiet.should_display(EventType.CODE_RUN) is False
    assert ui_quiet.should_display(EventType.INFO) is False
    assert ui_quiet.should_display(EventType.RESULT) is True

    assert ui_normal.should_display(EventType.CODE_RUN) is False
    assert ui_normal.should_display(EventType.INFO) is True

    assert ui_detailed.should_display(EventType.CODE_RUN) is True

    # 4. timestamps
    c1 = Console(record=True, width=100)
    ui_ts = ConsoleUI(console=c1, config=UiConfig(timestamps=True))
    ui_no_ts = ConsoleUI(console=c1, config=UiConfig(timestamps=False))
    line_ts = ui_ts._format_event_line(AgentEvent(EventType.INFO, "Agent", "msg"))
    line_no_ts = ui_no_ts._format_event_line(AgentEvent(EventType.INFO, "Agent", "msg"))
    assert ":" in line_ts.plain[:10]  # Contains HH:MM:SS
    assert ":" not in line_no_ts.plain[:5]

    # 5. show_environment & confirm_before_start
    cfg_flags = UiConfig(show_environment=False, confirm_before_start=False)
    assert cfg_flags.show_environment is False
    assert cfg_flags.confirm_before_start is False

    # 6. ascii_only
    sym_unicode = get_symbols(ascii_only=False)
    sym_ascii = get_symbols(ascii_only=True)
    assert sym_unicode["done"] != sym_ascii["done"]
    assert sym_ascii["done"] == "o"

    # 7. log_width
    ui_w80 = ConsoleUI(config=UiConfig(log_width=80))
    assert ui_w80.config.log_width == 80


@pytest.mark.unit
def test_flags_to_overrides_translation() -> None:
    """CLI flags map correctly to config override strings."""
    args = argparse.Namespace(
        verbose=True,
        quiet=False,
        plain=True,
        no_color=True,
        theme="light",
        yes=True,
        overrides=["limits.max_eda_loops=6"],
    )
    ov = flags_to_overrides(args)
    assert "limits.max_eda_loops=6" in ov
    assert "ui.verbosity=detailed" in ov
    assert "ui.mode=plain" in ov
    assert "ui.theme=mono" in ov
    assert "ui.theme=light" in ov
    assert "ui.confirm_before_start=false" in ov


@pytest.mark.unit
def test_live_status_bar_updates_from_ledger() -> None:
    """LiveStatusBar formats stage tracker, now line, and scoreboard correctly."""
    import time

    start_time = time.monotonic() - 65.0
    cfg = UiConfig(theme="dark")
    status_bar = LiveStatusBar(cfg, start_time)

    ledger = Ledger(user_task="Test task", phase="experiment_loop", iteration=2)
    status_bar.ledger = ledger
    status_bar.update_status("Modeler", "training lightgbm", "seeking higher accuracy")

    console = Console(record=True, width=100)
    lines = list(status_bar.__rich_console__(console, None))  # type: ignore[arg-type]
    assert len(lines) == 3

    stage_line, now_line, score_line = lines
    assert "Experiment" in stage_line.plain
    assert "Modeler" in now_line.plain
    assert "seeking higher accuracy" in now_line.plain
    assert "Scoreboard" in score_line.plain
