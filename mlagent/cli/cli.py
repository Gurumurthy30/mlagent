"""The command line. Run `mlagent start` inside the folder of an ML project.

This file talks to the user and wires the components together. All execution happens in the graph.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from rich.table import Table

from mlagent import __version__
from mlagent.agents.base import AgentContext
from mlagent.cli.console_ui import ConsoleUI
from mlagent.cli.summary import (
    render_confirmation_panel,
    render_environment_table,
    render_summary_panel,
)
from mlagent.configs.config import (
    DEFAULT_CONFIG_TEXT,
    ConfigError,
    UiConfig,
    get_effective_config_details,
    load_or_create_config,
)
from mlagent.graph.graph import run_pipeline
from mlagent.graph.state import Ledger
from mlagent.helper.code_runner import CodeRunner
from mlagent.helper.environment_probe import (
    EnvironmentInfo,
    probe_environment,
    save_environment,
)
from mlagent.helper.events import AgentEvent, EventBus
from mlagent.helper.llm_client import LlmClient, LlmUnavailableError
from mlagent.helper.workspace import (
    LedgerLoadError,
    RunLogWriter,
    Workspace,
    load_ledger,
    save_ledger,
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


def print_event(event: AgentEvent) -> None:
    """Print an agent event cleanly to stdout."""
    timestamp = event.created_at.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{event.agent_name:<16}] {event.message}", flush=True)


def ask_text(prompt: str) -> str:
    """Ask a free-text question."""
    try:
        return input(f"{prompt}: ").strip()
    except (KeyboardInterrupt, EOFError):
        return ""


def ask_required_task(prompt: str) -> str:
    """Prompt user for required task text and reject empty answers."""
    while True:
        try:
            ans = input(f"{prompt}: ").strip()
        except (KeyboardInterrupt, EOFError):
            raise EOFError("No task was provided.") from None
        if ans:
            return ans
        print("Task description cannot be empty.")


def ask_choice(prompt: str, choices: list[str]) -> str:
    """Ask user to pick from a list of choices."""
    choices_str = "/".join(choices)
    while True:
        try:
            ans = input(f"{prompt} [{choices_str}]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return choices[0]
        if ans in choices:
            return ans
        print(f"Please choose one of: {choices_str}")


def ask_confirm(prompt: str, default: bool = True) -> bool:
    """Ask user for yes/no confirmation."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        ans = input(f"{prompt} {suffix}: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return default
    if not ans:
        return default
    return ans in ("y", "yes")


def load_dotenv_if_present(folder: Path | None = None) -> None:
    """Load variables from .env files into os.environ without overwriting existing environment."""
    candidates = [Path.cwd() / ".env"]
    if folder is not None:
        folder_path = Path(folder).resolve()
        candidates.extend([
            folder_path / ".env",
            folder_path.parent / ".env",
            folder_path.parent.parent / ".env",
        ])

    seen_paths: set[Path] = set()
    for dotenv_path in candidates:
        if dotenv_path in seen_paths or not dotenv_path.is_file():
            continue
        seen_paths.add(dotenv_path)
        try:
            content = dotenv_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:
            pass


def show_final_summary(
    ledger: Ledger,
    config: UiConfig | None = None,
    ui: ConsoleUI | None = None,
) -> None:
    """Render the final summary panel from ledger state."""
    if ui is None:
        ui = ConsoleUI(config=config)
    ui_cfg = config or ui.config
    panel = render_summary_panel(ledger, ui_cfg)
    ui.console.print(panel)


def decide_exit_code(ledger: Ledger) -> int:
    ended_early = ledger.stop_reason in ("llm_unavailable", "agent_failed")
    return EXIT_PROBLEM if ledger.status == "failed" or ended_early else EXIT_OK


def prepare_workspace(workspace: Workspace, ui: ConsoleUI | None = None) -> bool:
    """Create the workspace. Returns True to resume a saved run, False for a new one."""
    if ui is None:
        ui = ConsoleUI()
    resuming = False
    old_config_text = ""
    if workspace.exists() and workspace.has_saved_ledger():
        if ui.ask_choice(RESUME_QUESTION, ["resume", "fresh"]) == "resume":
            resuming = True
        else:
            if workspace.config_path.exists():
                try:
                    old_config_text = workspace.config_path.read_text(encoding="utf-8")
                except OSError:
                    old_config_text = ""
            try:
                archive_path = workspace.archive_existing()
                ui.show_message(
                    f"The old workspace was kept as {archive_path.name}.",
                    style=ui.theme.info,
                )
            except Exception as err:
                ui.show_message(
                    f"Notice: Could not archive previous workspace ({err}). Continuing with fresh workspace.",
                    style=ui.theme.warn,
                )

    workspace.create_folders()
    if old_config_text:
        try:
            workspace.config_path.write_text(old_config_text, encoding="utf-8")
        except OSError:
            pass
    return resuming


def load_or_start_ledger(
    workspace: Workspace,
    resuming: bool,
    ui: ConsoleUI | None = None,
) -> Ledger:
    if ui is None:
        ui = ConsoleUI()
    if resuming:
        ledger = load_ledger(workspace)
        if ledger.status == "failed":
            ledger = ledger.model_copy(
                update={"status": "running", "stop_reason": None, "error_message": ""}
            )
        return ledger

    user_task = ui.ask_required_task(TASK_QUESTION)
    user_metrics_text = ui.ask_text(METRICS_QUESTION).strip()
    ledger = Ledger(user_task=user_task, user_metrics_text=user_metrics_text)
    save_ledger(workspace, ledger)
    return ledger


def init_config_command(
    output: str | None,
    force: bool,
    ui: ConsoleUI | None = None,
) -> int:
    if ui is None:
        ui = ConsoleUI()
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


def config_command(
    config_path_str: str | None = None,
    overrides: list[str] | None = None,
    ui: ConsoleUI | None = None,
) -> int:
    if ui is None:
        ui = ConsoleUI()
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
        details = get_effective_config_details(
            cfg_path if cfg_path.exists() else None, overrides or []
        )
    except ConfigError as error:
        ui.show_message(f"Config error: {error}", style="bold red")
        return EXIT_USAGE

    if ui._is_plain:
        ui.show_message("\nmlagent Effective Configuration:")
        ui.show_message(f"{'Section':<15} {'Key':<30} {'Value':<30} {'Source':<10}")
        ui.show_message("-" * 85)
        for item in details:
            val_str = str(item["value"])
            if len(val_str) > 30:
                val_str = val_str[:27] + "..."
            ui.show_message(
                f"{item['section']:<15} {item['key']:<30} {val_str:<30} {item['source']:<10}"
            )
        ui.show_message("")
        return EXIT_OK

    table = Table(title="mlagent Effective Configuration", box=None)
    table.add_column("Section", style=ui.theme.muted, width=15)
    table.add_column("Key", style=ui.theme.accent, width=30)
    table.add_column("Value", style=ui.theme.text, width=32)
    table.add_column("Source", style=ui.theme.info, width=10)

    for item in details:
        val_str = str(item["value"])
        if len(val_str) > 32:
            val_str = val_str[:29] + "..."
        src = item["source"]
        src_style = (
            ui.theme.ok
            if src == "file"
            else (ui.theme.accent if src == "cli" else ui.theme.muted)
        )
        table.add_row(item["section"], item["key"], val_str, f"[{src_style}]{src}[/]")

    ui.console.print(table)
    return EXIT_OK


def status_command(
    project_folder: Path,
    ui: ConsoleUI | None = None,
    config_path: Path | None = None,
    overrides: list[str] | None = None,
) -> int:
    """Display progress and summary view from a run's saved ledger."""
    if ui is None:
        ui = ConsoleUI()
    load_dotenv_if_present(project_folder)
    workspace = Workspace(project_folder)
    if not workspace.exists() or not workspace.has_saved_ledger():
        ui.show_message(
            f"No saved run found in {workspace.root}. Start a project with `mlagent start`.",
            style=ui.theme.warn,
        )
        return EXIT_OK

    ledger = load_ledger(workspace)
    cfg_file = config_path if config_path is not None else workspace.config_path
    try:
        config = load_or_create_config(cfg_file, overrides=overrides)
        ui_cfg = config.ui
    except Exception:
        ui_cfg = ui.config
    show_final_summary(ledger, ui_cfg, ui)
    return decide_exit_code(ledger)


def resume_command(
    project_folder: Path,
    ui: ConsoleUI | None = None,
    create_llm_client: LlmClientFactory = LlmClient,
    probe: Callable[[], EnvironmentInfo] = probe_environment,
    config_path: Path | None = None,
    overrides: list[str] | None = None,
    debug: bool = False,
) -> int:
    """Resume a saved run directly without prompting."""
    if ui is None:
        ui = ConsoleUI()
    workspace = Workspace(project_folder)
    if not workspace.exists() or not workspace.has_saved_ledger():
        ui.show_message(
            f"No saved run found in {workspace.root}. Use `mlagent start` to start a new project.",
            style=ui.theme.warn,
        )
        return EXIT_USAGE
    return start_command(
        project_folder,
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
    ui: ConsoleUI | None = None,
    create_llm_client: LlmClientFactory = LlmClient,
    probe: Callable[[], EnvironmentInfo] = probe_environment,
    config_path: Path | None = None,
    overrides: list[str] | None = None,
    force_resume: bool = False,
    debug: bool = False,
    skip_confirm: bool = False,
) -> int:
    load_dotenv_if_present(project_folder)
    workspace = Workspace(project_folder)
    if ui is None:
        ui = ConsoleUI()

    ui.show_header(__version__)

    resuming = False
    if force_resume:
        resuming = True
        if not workspace.has_saved_ledger():
            ui.show_message(
                f"No saved run found in {workspace.root}. Use `mlagent start` to start fresh.",
                style=ui.theme.warn,
            )
            return EXIT_USAGE
    else:
        resuming = prepare_workspace(workspace, ui)

    cfg_file = config_path if config_path is not None else workspace.config_path
    config_is_new = not cfg_file.exists()
    try:
        config = load_or_create_config(cfg_file, overrides=overrides)
    except ConfigError as error:
        ui.show_message(f"Config error: {error}", style="bold red")
        return EXIT_USAGE

    if config_is_new:
        ui.show_message(
            f"Created {cfg_file}. Edit it to change model or limits.",
            style=ui.theme.info,
        )

    # 1. Environment Check
    environment = probe()
    save_environment(environment, workspace.environment_path)
    if config.ui.show_environment:
        env_table = render_environment_table(
            environment, config.ui, low_ram_warning_gb=config.machine.low_ram_warning_gb
        )
        ui.console.print(env_table)

    event_bus = EventBus()
    event_bus.subscribe(ui.handle_event)
    event_bus.subscribe(RunLogWriter(workspace.run_log_path).handle_event)

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
        ui.show_message(f"  {advice}", style=ui.theme.warn)
        ui.show_message(f"  Log file: {workspace.logs_folder}", style=ui.theme.muted)
        return EXIT_USAGE

    # 3. Ledger Setup
    try:
        ledger = load_or_start_ledger(workspace, resuming, ui)
    except LedgerLoadError as error:
        ui.show_message(f"{error} Start again and choose fresh.", style="bold red")
        return EXIT_USAGE
    except EOFError:
        ui.show_message("No task was given, so there is nothing to do.", style=ui.theme.warn)
        return EXIT_USAGE

    if ledger.status == "done":
        ui.show_message("This run already finished.", style=ui.theme.ok)
        show_final_summary(ledger, config.ui, ui)
        return decide_exit_code(ledger)

    # 4. Confirmation
    if not resuming and not skip_confirm and config.ui.confirm_before_start:
        confirm_panel = render_confirmation_panel(
            task=ledger.user_task,
            metrics=ledger.user_metrics_text,
            data_folder=project_folder,
            limits=config.limits,
            workspace_folder=workspace.root,
            model_name=config.llm.name,
            config=config.ui,
            base_url=config.llm.base_url,
        )
        ui.console.print(confirm_panel)
        if not ui.ask_confirm("Start?"):
            ui.show_message("Aborted by user.", style=ui.theme.warn)
            return EXIT_OK

    if resuming:
        ui.show_message(f"Resuming saved run at phase: {ledger.phase}", style=ui.theme.info)

    agent_context = AgentContext(
        config=config,
        workspace=workspace,
        event_bus=event_bus,
        llm_client=llm_client,
        code_runner=CodeRunner(workspace, config.limits.code_timeout_minutes * 60),
        environment=environment,
    )

    # 5. Run Pipeline with Live Status
    ui.set_run_context(limits=config.limits, model_name=config.llm.name)
    ui.update_ledger(ledger)
    ui.start_live()
    try:
        final_ledger = run_pipeline(ledger, agent_context)
    except KeyboardInterrupt:
        ui.stop_live()
        ui.show_message(
            "\nStopped. State is saved; run `mlagent resume` to continue.",
            style=ui.theme.warn,
        )
        return EXIT_INTERRUPTED
    except Exception as err:
        ui.stop_live()
        if debug:
            import traceback

            traceback.print_exc()
        ui.show_message(
            f"\nThe run stopped on an unexpected error: {err}. See {workspace.logs_folder}.",
            style="bold red",
        )
        return EXIT_PROBLEM
    finally:
        ui.stop_live()

    show_final_summary(final_ledger, config.ui, ui)
    return decide_exit_code(final_ledger)


def flags_to_overrides(args: argparse.Namespace) -> list[str]:
    """Map command line flags to config override key paths."""
    ov: list[str] = list(getattr(args, "overrides", []) or [])
    if getattr(args, "model", None):
        ov.append(f"llm.name={args.model}")
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
    load_dotenv_if_present()
    parser = argparse.ArgumentParser(
        prog="mlagent",
        description="A multi-agent system that runs a machine learning project end to end.",
    )
    parser.add_argument("--version", action="version", version=f"mlagent {__version__}")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--config", type=str, default=None, help="Path to config.yaml file")
    common_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Ollama model name to use (e.g. gemma4:31b, qwen2.5:32b, llama3:8b)",
    )
    common_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config setting, e.g. --set limits.max_eda_loops=6",
    )
    common_parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )
    common_parser.add_argument(
        "--debug", action="store_true", help="Show full stack traces on unexpected errors"
    )
    common_parser.add_argument(
        "--plain",
        action="store_true",
        help="Plain text output without color or live updates",
    )
    common_parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colors (uses monochrome theme)",
    )
    common_parser.add_argument(
        "--theme",
        choices=["dark", "light", "mono"],
        default=None,
        help="Color theme (default: dark)",
    )
    common_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Detailed output (includes code executions)",
    )
    common_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet output (warnings and errors only)",
    )

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
    init_parser = subparsers.add_parser(
        "init-config", help="Write a default config.yaml with documentation."
    )
    init_parser.add_argument(
        "--output", type=str, default=None, help="Output config path (default: config.yaml)"
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing config file if present"
    )

    # config
    subparsers.add_parser(
        "config",
        parents=[common_parser],
        help="Display effective configuration and sources.",
    )

    arguments = parser.parse_args(argv)
    init_ui_cfg = UiConfig(
        mode="plain" if getattr(arguments, "plain", False) else "auto",
        theme="mono"
        if getattr(arguments, "no_color", False)
        else (getattr(arguments, "theme", None) or "dark"),
        verbosity="detailed"
        if getattr(arguments, "verbose", False)
        else ("quiet" if getattr(arguments, "quiet", False) else "normal"),
    )
    ui = ConsoleUI(config=init_ui_cfg)
    overrides = flags_to_overrides(arguments)

    if arguments.command == "start":
        cfg_path = Path(arguments.config) if arguments.config else None
        target_folder = Path(arguments.folder).resolve()
        return start_command(
            target_folder,
            ui=ui,
            config_path=cfg_path,
            overrides=overrides,
            debug=arguments.debug,
            skip_confirm=arguments.yes,
        )
    if arguments.command == "resume":
        cfg_path = Path(arguments.config) if arguments.config else None
        target_folder = Path(arguments.folder).resolve()
        return resume_command(
            target_folder,
            ui=ui,
            config_path=cfg_path,
            overrides=overrides,
            debug=arguments.debug,
        )
    if arguments.command == "status":
        cfg_path = Path(arguments.config) if arguments.config else None
        target_folder = Path(arguments.folder).resolve()
        return status_command(
            target_folder,
            ui=ui,
            config_path=cfg_path,
            overrides=overrides,
        )
    if arguments.command == "init-config":
        return init_config_command(arguments.output, arguments.force, ui=ui)
    if arguments.command == "config":
        return config_command(arguments.config, overrides, ui=ui)

    parser.print_help()
    return EXIT_USAGE
