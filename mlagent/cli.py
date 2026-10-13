"""The command line. Run `mlagent start` inside the folder of an ML project.

This file only talks to the user and wires the parts together. All the work happens in the graph.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from mlagent import __version__
from mlagent.agents.base import AgentContext
from mlagent.code_runner import CodeRunner
from mlagent.config import ConfigError, load_or_create_config
from mlagent.console_ui import ConsoleUI
from mlagent.environment_probe import (
    EnvironmentInfo,
    environment_to_text,
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
    while True:
        answer = ui.ask_text(question).strip()
        if answer:
            return answer
        ui.show_message("Please type something.", style="yellow")


def show_final_summary(ui: ConsoleUI, ledger: Ledger, workspace: Workspace) -> None:
    if ledger.status == "failed":
        ui.show_message("The run failed.", style="bold red")
    else:
        ui.show_message(f"Finished. Stop reason: {ledger.stop_reason}.", style="bold green")
        if ledger.error_message:
            ui.show_message(f"The run ended early: {ledger.error_message}", style="yellow")

    best_run = ledger.best_approved_run()
    if best_run is not None:
        ui.show_message(
            f"Best result: {best_run.experiment_id} ({best_run.model_family}), "
            f"{ledger.plan.primary_metric} {best_run.score_mean:.4f}"
        )
    if ledger.final.predictions_path:
        ui.show_message(f"Predictions: {ledger.final.predictions_path}")
        for message in ledger.final.check_messages:
            ui.show_message(f"  {message}")
    if ledger.final.report_path:
        ui.show_message(f"Report: {ledger.final.report_path}")
    ui.show_message(f"Memory, logs and artifacts are in {workspace.root}")


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

    user_task = ask_required_text(ui, TASK_QUESTION)
    user_metrics_text = ui.ask_text(METRICS_QUESTION).strip()
    ledger = create_new_ledger(user_task, user_metrics_text)
    save_ledger(workspace, ledger)
    return ledger


def start_command(
    project_folder: Path,
    ui: ConsoleUI,
    create_llm_client: LlmClientFactory = LlmClient,
    probe: Callable[[], EnvironmentInfo] = probe_environment,
) -> int:
    workspace = Workspace(project_folder)
    resuming = prepare_workspace(ui, workspace)

    config_is_new = not workspace.config_path.exists()
    try:
        config = load_or_create_config(workspace.config_path)
    except ConfigError as error:
        ui.show_message(str(error), style="bold red")
        return EXIT_USAGE
    if config_is_new:
        ui.show_message(f"Created {workspace.config_path}. Edit it to change the model or limits.")

    event_bus = EventBus()
    event_bus.subscribe(ui.handle_event)
    event_bus.subscribe(RunLogWriter(workspace.run_log_path).handle_event)

    ui.show_message("Checking this machine...")
    environment = probe()
    save_environment(environment, workspace.environment_path)
    ui.show_message(environment_to_text(environment))

    llm_client = create_llm_client(config.model, workspace, event_bus)
    try:
        llm_client.check_ready()
    except LlmUnavailableError as error:
        ui.show_message(str(error), style="bold red")
        return EXIT_USAGE

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
        show_final_summary(ui, ledger, workspace)
        return decide_exit_code(ledger)
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
    try:
        final_ledger = run_pipeline(ledger, agent_context)
    except KeyboardInterrupt:
        ui.show_message(
            "Stopped. Run `mlagent start` again and choose resume to continue.", style="yellow"
        )
        return EXIT_INTERRUPTED
    except Exception:
        ui.show_message(
            f"The run stopped on an unexpected error. See {workspace.logs_folder}.",
            style="bold red",
        )
        return EXIT_PROBLEM

    show_final_summary(ui, final_ledger, workspace)
    return decide_exit_code(final_ledger)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlagent",
        description="A multi-agent system that runs a machine learning project end to end.",
    )
    parser.add_argument("--version", action="version", version=f"mlagent {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("start", help="Run the agents on the ML project in this folder.")
    arguments = parser.parse_args(argv)

    if arguments.command == "start":
        return start_command(Path.cwd(), ConsoleUI())
    parser.print_help()
    return EXIT_USAGE
