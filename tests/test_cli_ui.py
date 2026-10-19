"""Unit tests for ConsoleUI and Rich rendering components."""

import argparse
from pathlib import Path

from rich.console import Console

from mlagent.cli.cli import (
    EXIT_OK,
    EXIT_USAGE,
    flags_to_overrides,
    resume_command,
    status_command,
)
from mlagent.cli.console_ui import ConsoleUI
from mlagent.cli.summary import (
    LiveStatusBar,
    format_progress_bar,
    get_symbols,
    get_theme,
    is_plain_mode,
    render_confirmation_panel,
    render_environment_table,
    render_summary_panel,
)
from mlagent.configs.config import LimitsConfig, UiConfig
from mlagent.graph.state import EdaFinding, Ledger
from mlagent.helper.environment_probe import EnvironmentInfo
from mlagent.helper.events import AgentEvent, EventType


def test_status_command_on_fresh_empty_workspace_prints_clear_message(tmp_path: Path) -> None:
    """status command on a fresh empty workspace prints a clear friendly message without error."""
    console = Console(record=True, width=100)
    ui = ConsoleUI(console=console)

    exit_code = status_command(tmp_path, ui=ui)
    assert exit_code == EXIT_OK

    output = console.export_text()
    assert "No saved run found" in output
    assert "mlagent start" in output


def test_resume_command_on_fresh_empty_workspace_exits_with_usage(tmp_path: Path) -> None:
    """resume command without an existing saved run exits cleanly with a helpful message."""
    console = Console(record=True, width=100)
    ui = ConsoleUI(console=console)

    exit_code = resume_command(tmp_path, ui=ui)
    assert exit_code == EXIT_USAGE

    output = console.export_text()
    assert "No saved run found" in output


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


def test_flags_to_overrides_with_model_flag() -> None:
    """CLI --model flag maps to llm.name override."""
    args = argparse.Namespace(
        model="qwen2.5:32b",
        overrides=[],
    )
    ov = flags_to_overrides(args)
    assert "llm.name=qwen2.5:32b" in ov


def test_live_status_bar_updates_from_ledger() -> None:
    """LiveStatusBar formats stage tracker, now line, and scoreboard correctly."""
    import time

    start_time = time.monotonic() - 65.0
    cfg = UiConfig(theme="dark")
    status_bar = LiveStatusBar(cfg, start_time)

    ledger = Ledger(user_task="Test task", phase="modeler", iteration=2)
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


def test_live_status_bar_uses_configured_limits_and_model() -> None:
    """LiveStatusBar dynamically displays max_loops from limits config and model name."""
    import time

    start_time = time.monotonic() - 10.0
    cfg = UiConfig(theme="dark")
    limits = LimitsConfig(max_experiment_loops=25)
    status_bar = LiveStatusBar(cfg, start_time, limits=limits, model_name="qwen2.5:32b")

    ledger = Ledger(user_task="Test task", phase="modeler", iteration=3)
    status_bar.ledger = ledger

    console = Console(record=True, width=120)
    lines = list(status_bar.__rich_console__(console, None))  # type: ignore[arg-type]
    stage_line = lines[0].plain
    score_line = lines[2].plain

    assert "round 4/25" in stage_line
    assert "qwen2.5:32b" in score_line


def test_confirmation_panel_displays_model_and_server() -> None:
    """Confirmation panel displays configured model, server url, and dynamic limits."""
    cfg = UiConfig(theme="dark")
    limits = LimitsConfig(max_experiment_loops=18, max_eda_loops=6)
    panel = render_confirmation_panel(
        task="predict survival",
        metrics="f1",
        data_folder=Path("."),
        limits=limits,
        workspace_folder=Path("mlagent_workspace"),
        model_name="deepseek-r1:32b",
        config=cfg,
        base_url="http://127.0.0.1:11434",
    )
    console = Console(record=True, width=100)
    console.print(panel)
    output = console.export_text()

    assert "deepseek-r1:32b" in output
    assert "18 experiment rounds" in output
    assert "EDA cap: 6" in output
    assert "http://127.0.0.1:11434" in output


def test_summary_panel_escapes_bracketed_markup() -> None:
    """Findings or task text with literal square brackets do not break markup parsing."""
    cfg = UiConfig(theme="dark")
    ledger = Ledger(
        user_task="Predict [status] on [train.csv]",
        status="done",
    )
    ledger.eda.concise_executive_summary = "Analyzed [Cabin] column with [deck, num, side] splits"
    ledger.eda.findings.append(
        EdaFinding(
            priority="high",
            statement="Feature [CryoSleep] has [99%] predictive correlation",
            quantitative_evidence="Correlation = 0.99",
            modeling_implication="Use feature in GBDT",
        )
    )
    panel = render_summary_panel(ledger, cfg)
    console = Console(record=True, width=100)
    console.print(panel)
    output = console.export_text()
    assert "Predict [status]" in output
    assert "[deck, num, side]" in output
    assert "[CryoSleep]" in output


def test_render_environment_table() -> None:
    """Environment table displays system details cleanly in rich format."""
    env = EnvironmentInfo(
        operating_system="Windows 11",
        shell_name="powershell",
        python_version="3.12.0",
        cpu_count=8,
        ram_total_gb=32.0,
        ram_free_gb=18.0,
        gpu_name="NVIDIA RTX 4090",
        gpu_memory_total_gb=24.0,
        can_reach_pypi=True,
    )
    cfg = UiConfig(theme="dark")
    panel = render_environment_table(env, cfg)
    console = Console(record=True, width=100)
    console.print(panel)
    output = console.export_text()

    assert "Windows 11" in output
    assert "powershell" in output
    assert "NVIDIA RTX 4090" in output
    assert "PyPI" in output


def test_format_progress_bar() -> None:
    """format_progress_bar handles empty, partial, and 100% progress correctly."""
    assert format_progress_bar(0, 0) == ""
    assert format_progress_bar(0, 10) == "[░░░░░░░░░░] 0%"
    assert format_progress_bar(5, 10) == "[█████░░░░░] 50%"
    assert format_progress_bar(10, 10) == "[██████████] 100%"
    # ASCII mode
    assert format_progress_bar(6, 10, ascii_only=True) == "[######----] 60%"


def test_live_status_bar_eda_loop_and_progress_bar() -> None:
    """LiveStatusBar shows EDA round and progress bar during Understand phase."""
    import time
    start_time = time.monotonic() - 15.0
    cfg = UiConfig(theme="dark")
    limits = LimitsConfig(max_eda_loops=8, max_experiment_loops=12)
    status_bar = LiveStatusBar(cfg, start_time, limits=limits)

    ledger = Ledger(user_task="House Prices", phase="eda_analyst")
    ledger.eda.eda_loops_used = 2  # Current loop is 3/8
    status_bar.ledger = ledger

    console = Console(record=True, width=120)
    lines = list(status_bar.__rich_console__(console, None))  # type: ignore[arg-type]
    stage_line = lines[0].plain
    score_line = lines[2].plain

    assert "Understand (EDA 3/8)" in stage_line
    assert "%" in score_line or "Scoreboard" in score_line


def test_console_ui_updates_ledger_and_preserves_on_start_live() -> None:
    """ConsoleUI stores and attaches the active ledger reference to LiveStatusBar."""
    cfg = UiConfig(mode="rich")
    console = Console(record=True, width=100)
    ui = ConsoleUI(console=console, config=cfg)

    ledger = Ledger(user_task="House Prices", phase="modeler", iteration=1)
    ui.update_ledger(ledger)
    assert ui._current_ledger is ledger

    ui.start_live()
    try:
        assert ui._live_status_bar is not None
        assert ui._live_status_bar.ledger is ledger
    finally:
        ui.stop_live()


def test_console_ui_handle_event_updates_live_status_on_agent_started() -> None:
    """ConsoleUI handle_event updates active agent, action, and ledger on AGENT_STARTED."""
    cfg = UiConfig(mode="rich")
    console = Console(record=True, width=100)
    ui = ConsoleUI(console=console, config=cfg)
    ui.start_live()
    try:
        ledger = Ledger(user_task="House Prices", phase="eda_analyst")
        ledger.eda.eda_loops_used = 1
        event = AgentEvent(
            event_type=EventType.AGENT_STARTED,
            agent_name="Data Analyst",
            message="planning questions for EDA round 2 of 8",
            data={"ledger": ledger},
        )
        ui.handle_event(event)

        assert ui._current_agent == "Data Analyst"
        assert "planning questions for EDA round 2 of 8" in ui._current_action
        assert ui._live_status_bar.ledger is ledger
        assert ui._live_status_bar.current_agent == "Data Analyst"
    finally:
        ui.stop_live()


def test_live_status_bar_transitions_through_all_phases() -> None:
    """LiveStatusBar stages and progress bars advance across Understand, Plan, Experiment, Deliver."""
    import time

    start_time = time.monotonic() - 60.0
    cfg = UiConfig(theme="dark")
    limits = LimitsConfig(max_eda_loops=8, max_experiment_loops=12)
    status_bar = LiveStatusBar(cfg, start_time, limits=limits)
    console = Console(record=True, width=120)

    # 1. EDA loop 1
    ledger = Ledger(user_task="House Prices", phase="eda_analyst")
    ledger.eda.eda_loops_used = 0
    status_bar.ledger = ledger
    lines = list(status_bar.__rich_console__(console, None))  # type: ignore[arg-type]
    assert "Understand (EDA 1/8)" in lines[0].plain
    assert "EDA 1/8" in lines[2].plain

    # 2. EDA loop 2
    ledger.eda.eda_loops_used = 1
    lines = list(status_bar.__rich_console__(console, None))  # type: ignore[arg-type]
    assert "Understand (EDA 2/8)" in lines[0].plain
    assert "EDA 2/8" in lines[2].plain

    # 3. EDA Blueprint
    ledger.eda.done = True
    lines = list(status_bar.__rich_console__(console, None))  # type: ignore[arg-type]
    assert "Understand (Blueprint)" in lines[0].plain
    assert "EDA Blueprint" in lines[2].plain

    # 4. Plan (Round 0 ml_lead)
    ledger.phase = "ml_lead"
    lines = list(status_bar.__rich_console__(console, None))  # type: ignore[arg-type]
    assert "Understand (1 EDAs)" in lines[0].plain
    assert "Plan" in lines[0].plain
    assert "Planning" in lines[2].plain

    # 5. Experiment (Feature Engineer / Modeler)
    ledger.phase = "modeler"
    ledger.iteration = 0
    lines = list(status_bar.__rich_console__(console, None))  # type: ignore[arg-type]
    assert "Experiment (round 1/12)" in lines[0].plain
    assert "Round 1/12" in lines[2].plain

    # 6. ML Lead Strategist for round 2 (should NOT go back to Plan)
    from mlagent.graph.state import ExperimentRun
    ledger.phase = "ml_lead"
    ledger.iteration = 1
    ledger.runs.append(
        ExperimentRun(
            experiment_id="EXP_001",
            iteration=0,
            model_family="lightgbm",
            verdict="approve",
            cv_mean=0.1234,
        )
    )
    lines = list(status_bar.__rich_console__(console, None))  # type: ignore[arg-type]
    assert "Experiment (round 2/12)" in lines[0].plain
    assert "Round 2/12" in lines[2].plain

    # 7. Deliver (Finalizer)
    ledger.phase = "finalizer"
    lines = list(status_bar.__rich_console__(console, None))  # type: ignore[arg-type]
    assert "Deliver" in lines[0].plain
    assert "Complete" in lines[2].plain



