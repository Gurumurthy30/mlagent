"""The command line. Run `mlagent start` inside the folder of an ML project.

This file only talks to the user and wires the parts together. All the work happens in the graph.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from mlagent import __version__
from mlagent.agents.base import AgentContext
from mlagent.code_runner import CodeRunner
from mlagent.config import (
    DEFAULT_CONFIG_TEXT,
    AppConfig,
    ConfigError,
    UiConfig,
    get_effective_config_details,
    load_or_create_config,
)
from mlagent.console_ui import ConsoleUI
from mlagent.environment_probe import (
    EnvironmentInfo,
    probe_environment,
    save_environment,
)
from mlagent.events import EventBus
from mlagent.graph import run_pipeline
from mlagent.llm_client import LlmClient, LlmUnavailableError
from mlagent.memory import (
    LedgerLoadError,
    RunLogWriter,
    Workspace,
    create_new_ledger,
    load_ledger,
    save_ledger,
)
from mlagent.state import Ledger
from mlagent.summary import (
    get_symbols,
    get_theme,
    is_plain_mode,
    render_confirmation_panel,
    render_environment_table,
    render_summary_panel,
)

EXIT_OK = 0
EXIT_PROBLEM = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

RESUME_QUESTION = "A saved run exists in this folder. Resume it or start fresh?"
TASK_QUESTION = "What should the agents do? Describe the ML task"
METRICS_QUESTION = (
    "Which metrics matter? Free text, you may list several (press Enter to let the agents decide)"
)

LlmClientFactory = Callable[..., LlmClient]


def ask_required_text(ui: ConsoleUI, question: str) -> str:
    """Prompt user for required text and reject empty answers."""
    return ui.ask_required_task(question)


def show_final_summary(
    ui: ConsoleUI,
    ledger: Ledger,
    config: AppConfig | None = None,
    workspace: Workspace | None = None,
) -> None:
    """Render the final summary panel and headline without LLM calls."""
    ui_config = config.ui if config is not None else ui.config
    summary_panel = render_summary_panel(ledger, ui_config)
    ui.console.print(summary_panel)
    if ledger.final.headline:
        ui.show_message(ledger.final.headline, style=f"bold {ui.theme.accent}")


def decide_exit_code(ledger: Ledger) -> int:
    ended_early = ledger.stop_reason in ("llm_unavailable", "agent_failed")
    return EXIT_PROBLEM if ledger.status == "failed" or ended_early else EXIT_OK


def prepare_workspace(ui: ConsoleUI, workspace: Workspace) -> bool:
    """Create the workspace. Returns True to resume a saved run, False for a new one."""
    resuming = False
    old_config_text = ""
    if workspace.exists() and workspace.has_saved_ledger():
        if ui.ask_choice(RESUME_QUESTION, ["resume", "fresh"]) == "resume":
            resuming = True
        else:
            if workspace.config_path.exists():
                old_config_text = workspace.config_path.read_text(encoding="utf-8")
            archive_path = workspace.archive_existing()
            ui.show_message(f"The old workspace was kept as {archive_path.name}.")

    workspace.create_folders()
    if old_config_text:
        workspace.config_path.write_text(old_config_text, encoding="utf-8")
    return resuming


def load_or_start_ledger(ui: ConsoleUI, workspace: Workspace, resuming: bool) -> Ledger:
    if resuming:
        ledger = load_ledger(workspace)
        if ledger.status == "failed":
            ledger = ledger.model_copy(
                update={"status": "running", "stop_reason": None, "error_message": ""}
            )
        return ledger

    user_task = ui.ask_required_task(TASK_QUESTION)
    user_metrics_text = ui.ask_text(METRICS_QUESTION).strip()
    ledger = create_new_ledger(user_task, user_metrics_text)
    save_ledger(workspace, ledger)
    return ledger


def init_config_command(output: str | None, force: bool, ui: ConsoleUI) -> int:
    target_path = Path(output) if output else Path("config.yaml")
    if target_path.exists() and not force:
        ui.show_message(
            f"Config file {target_path} already exists. Use --force to overwrite.",
            style="bold red",
        )
        return EXIT_USAGE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    ui.show_message(f"Created default configuration at {target_path}", style="bold green")
    return EXIT_OK


def config_command(config_path_str: str | None, overrides: list[str], ui: ConsoleUI) -> int:
    cfg_path = (
        Path(config_path_str)
        if config_path_str
        else (
            Path("mlagent_workspace/config.yaml")
            if Path("mlagent_workspace/config.yaml").exists()
            else Path("config.yaml")
        )
    )
    try:
        details = get_effective_config_details(cfg_path if cfg_path.exists() else None, overrides)
    except ConfigError as error:
        ui.show_message(str(error), style="bold red")
        return EXIT_USAGE

    from rich.table import Table

    table = Table(title="mlagent Effective Configuration")
    table.add_column("Section", style="cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value", style="green")
    table.add_column("Tier", style="dim")
    table.add_column("Source", style="yellow")
    for item in details:
        val_str = str(item["value"])
        if len(val_str) > 40:
            val_str = val_str[:37] + "..."
        table.add_row(item["section"], item["key"], val_str, item["tier"], item["source"])
    ui.console.print(table)
    return EXIT_OK


def status_command(
    project_folder: Path,
    ui: ConsoleUI,
    config_path: Path | None = None,
    overrides: list[str] | None = None,
) -> int:
    """Render progress and summary view from a run's saved ledger without disturbing a running process."""
    workspace = Workspace(project_folder)
    if not workspace.exists() or not workspace.has_saved_ledger():
        ui.show_message(
            f"No saved run found in {workspace.root}. Start a project with `mlagent start`.",
            style=ui.theme.warn,
        )
        return EXIT_OK

    cfg_file = config_path if config_path is not None else workspace.config_path
    try:
        config = load_or_create_config(cfg_file, overrides=overrides)
    except ConfigError as error:
        ui.show_message(str(error), style="bold red")
        return EXIT_USAGE

    ui.config = config.ui
    ui.theme = get_theme(config.ui.theme)
    ui.symbols = get_symbols(config.ui.ascii_only)
    ui._is_plain = is_plain_mode(config.ui, ui.console)

    ledger = load_ledger(workspace)
    show_final_summary(ui, ledger, config, workspace)
    return decide_exit_code(ledger)


def resume_command(
    project_folder: Path,
    ui: ConsoleUI,
    create_llm_client: LlmClientFactory = LlmClient,
    probe: Callable[[], EnvironmentInfo] = probe_environment,
    config_path: Path | None = None,
    overrides: list[str] | None = None,
    debug: bool = False,
) -> int:
    """Resume a saved run directly without prompting."""
    workspace = Workspace(project_folder)
    if not workspace.exists() or not workspace.has_saved_ledger():
        ui.show_message(
            f"No saved run found in {workspace.root}. Use `mlagent start` to start a new project.",
            style="bold red",
        )
        return EXIT_USAGE

    return start_command(
        project_folder=project_folder,
        ui=ui,
        create_llm_client=create_llm_client,
        probe=probe,
        config_path=config_path,
        overrides=overrides,
        force_resume=True,
        debug=debug,
    )


def start_command(
    project_folder: Path,
    ui: ConsoleUI,
    create_llm_client: LlmClientFactory = LlmClient,
    probe: Callable[[], EnvironmentInfo] = probe_environment,
    config_path: Path | None = None,
    overrides: list[str] | None = None,
    force_resume: bool = False,
    debug: bool = False,
) -> int:
    workspace = Workspace(project_folder)
    ui.show_header(__version__)

    resuming = False
    if force_resume:
        resuming = True
        workspace.create_folders()
    else:
        resuming = prepare_workspace(ui, workspace)

    cfg_file = config_path if config_path is not None else workspace.config_path
    config_is_new = not cfg_file.exists()
    try:
        config = load_or_create_config(cfg_file, overrides=overrides)
    except ConfigError as error:
        ui.show_message(f"Config error: {error}", style="bold red")
        return EXIT_USAGE

    # Update UI with loaded config
    ui.config = config.ui
    ui.theme = get_theme(config.ui.theme)
    ui.symbols = get_symbols(config.ui.ascii_only)
    ui._is_plain = is_plain_mode(config.ui, ui.console)

    if config_is_new:
        ui.show_message(f"Created {cfg_file}. Edit it to change the model or limits.")

    event_bus = EventBus()
    event_bus.subscribe(ui.handle_event)
    event_bus.subscribe(RunLogWriter(workspace.run_log_path).handle_event)

    # 1. Environment Check
    environment = probe()
    save_environment(environment, workspace.environment_path)
    if config.ui.show_environment:
        env_table = render_environment_table(
            environment, config.ui, low_ram_warning_gb=config.machine.low_ram_warning_gb
        )
        ui.console.print(env_table)

    # 2. LLM Readiness Check
    llm_client = create_llm_client(config.llm, workspace, event_bus)
    try:
        llm_client.check_ready()
    except LlmUnavailableError as error:
        err_text = str(error)
        model_name = config.llm.name
        base_url = config.llm.base_url
        if "not pulled" in err_text or "not available" in err_text:
            advice = f"Pull the model first: ollama pull {model_name}"
        else:
            advice = f"Make sure Ollama is running at {base_url} (start it with `ollama serve`)."
        ui.show_message(f"LLM Unavailable: {err_text}", style="bold red")
        ui.show_message(f"  {advice}")
        ui.show_message(f"  Log file: {workspace.logs_folder}")
        return EXIT_USAGE

    # 3. Ledger Setup
    try:
        ledger = load_or_start_ledger(ui, workspace, resuming)
    except LedgerLoadError as error:
        ui.show_message(f"{error} Start again and choose fresh.", style="bold red")
        return EXIT_USAGE
    except EOFError:
        ui.show_message("No task was given, so there is nothing to do.", style="bold red")
        return EXIT_USAGE

    if ledger.status == "done":
        ui.show_message("This run already finished.")
        show_final_summary(ui, ledger, config, workspace)
        return decide_exit_code(ledger)

    # 4. Confirmation Panel ("What I will do")
    if not resuming and config.ui.confirm_before_start:
        conf_panel = render_confirmation_panel(
            task=ledger.user_task,
            metrics=ledger.user_metrics_text,
            data_folder=workspace.project_folder,
            limits=config.limits,
            workspace_folder=workspace.root,
            model_name=config.llm.name,
            config=config.ui,
        )
        ui.console.print(conf_panel)
        if not ui.ask_confirm("Start?"):
            ui.show_message("Aborted by user.")
            return EXIT_OK

    if resuming:
        ui.show_message(f"Resuming the saved run at: {ledger.phase}.")

    agent_context = AgentContext(
        config=config,
        workspace=workspace,
        event_bus=event_bus,
        llm_client=llm_client,
        code_runner=CodeRunner(workspace, config.limits.code_timeout_minutes * 60),
        environment=environment,
    )

    # 5. Run Pipeline with Live Status Display
    ui.start_live()
    ui.update_ledger(ledger)
    try:
        final_ledger = run_pipeline(ledger, agent_context)
    except KeyboardInterrupt:
        ui.stop_live()
        ui.show_message(
            "Stopped. State is saved; run `mlagent resume` to continue.",
            style="yellow",
        )
        return EXIT_INTERRUPTED
    except Exception as err:
        ui.stop_live()
        if debug:
            import traceback
            traceback.print_exc()
        ui.show_message(
            f"The run stopped on an unexpected error: {err}. See {workspace.logs_folder}.",
            style="bold red",
        )
        return EXIT_PROBLEM
    finally:
        ui.stop_live()

    show_final_summary(ui, final_ledger, config, workspace)
    return decide_exit_code(final_ledger)


def flags_to_overrides(args: argparse.Namespace) -> list[str]:
    """Map command line flags to config override key paths."""
    ov: list[str] = list(getattr(args, "overrides", []) or [])
    if getattr(args, "verbose", False):
        ov.append("ui.verbosity=detailed")
    elif getattr(args, "quiet", False):
        ov.append("ui.verbosity=quiet")
    if getattr(args, "plain", False):
        ov.append("ui.mode=plain")
    if getattr(args, "no_color", False):
        ov.append("ui.theme=mono")
    if getattr(args, "theme", None):
        ov.append(f"ui.theme={args.theme}")
    if getattr(args, "yes", False):
        ov.append("ui.confirm_before_start=false")
    return ov


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlagent",
        description="A multi-agent system that runs a machine learning project end to end.",
    )
    parser.add_argument("--version", action="version", version=f"mlagent {__version__}")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--config", type=str, default=None, help="Path to config.yaml file")
    common_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config setting, e.g. --set limits.max_eda_loops=6",
    )
    common_parser.add_argument("-v", "--verbose", action="store_true", help="Detailed verbosity including code runs")
    common_parser.add_argument("-q", "--quiet", action="store_true", help="Quiet verbosity")
    common_parser.add_argument("--plain", action="store_true", help="Force plain terminal output mode")
    common_parser.add_argument("--no-color", action="store_true", help="Monochrome theme without ANSI colors")
    common_parser.add_argument("--theme", choices=["dark", "light", "mono"], default=None, help="Color theme")
    common_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    common_parser.add_argument("--debug", action="store_true", help="Show full stack traces on unexpected errors")

    subparsers = parser.add_subparsers(dest="command")

    # start
    start_parser = subparsers.add_parser(
        "start",
        parents=[common_parser],
        help="Run the agents on the ML project in this folder.",
    )
    start_parser.add_argument("folder", nargs="?", default=".", help="Project folder path")

    # resume
    resume_parser = subparsers.add_parser(
        "resume",
        parents=[common_parser],
        help="Resume a saved run directly without prompting.",
    )
    resume_parser.add_argument("folder", nargs="?", default=".", help="Project folder path")

    # status
    status_parser = subparsers.add_parser(
        "status",
        parents=[common_parser],
        help="Display progress and summary view from a run's saved ledger.",
    )
    status_parser.add_argument("folder", nargs="?", default=".", help="Project folder path")

    # init-config
    init_parser = subparsers.add_parser("init-config", help="Write a default config.yaml with documentation.")
    init_parser.add_argument("--output", type=str, default=None, help="Output config path (default: config.yaml)")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config file if present")

    # config
    subparsers.add_parser(
        "config",
        parents=[common_parser],
        help="Display effective configuration and sources.",
    )

    arguments = parser.parse_args(argv)

    # Pre-configure UI with CLI flags
    init_ui_cfg = UiConfig(
        mode="plain" if getattr(arguments, "plain", False) else "auto",
        theme="mono" if getattr(arguments, "no_color", False) else (getattr(arguments, "theme", None) or "dark"),
        verbosity="detailed" if getattr(arguments, "verbose", False) else ("quiet" if getattr(arguments, "quiet", False) else "normal"),
    )
    ui = ConsoleUI(config=init_ui_cfg)
    overrides = flags_to_overrides(arguments)

    if arguments.command == "start":
        cfg_path = Path(arguments.config) if arguments.config else None
        target_folder = Path(arguments.folder).resolve()
        return start_command(
            target_folder,
            ui,
            config_path=cfg_path,
            overrides=overrides,
            debug=arguments.debug,
        )
    if arguments.command == "resume":
        cfg_path = Path(arguments.config) if arguments.config else None
        target_folder = Path(arguments.folder).resolve()
        return resume_command(
            target_folder,
            ui,
            config_path=cfg_path,
            overrides=overrides,
            debug=arguments.debug,
        )
    if arguments.command == "status":
        cfg_path = Path(arguments.config) if arguments.config else None
        target_folder = Path(arguments.folder).resolve()
        return status_command(
            target_folder,
            ui,
            config_path=cfg_path,
            overrides=overrides,
        )
    if arguments.command == "init-config":
        return init_config_command(arguments.output, arguments.force, ui)
    if arguments.command == "config":
        return config_command(arguments.config, overrides, ui)

    parser.print_help()
    return EXIT_USAGE
