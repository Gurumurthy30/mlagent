"""Pure rendering components and summary views for the terminal UI.

Transforms ledgers, environment info, and runtime state into Rich renderables and plain text,
supporting dark, light, and mono themes, ascii fallbacks, and plain terminal degradation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mlagent.config import UiConfig
from mlagent.environment_probe import EnvironmentInfo
from mlagent.state import Ledger


@dataclass(frozen=True)
class ThemeTokens:
    name: str
    accent: str
    ok: str
    warn: str
    error: str
    muted: str
    info: str
    text: str
    border: str


DARK_THEME = ThemeTokens(
    name="dark",
    accent="#60A5FA",
    ok="#34D399",
    warn="#FBBF24",
    error="#F87171",
    muted="#94A3B8",
    info="#38BDF8",
    text="#D7E0EA",
    border="#29445B",
)

LIGHT_THEME = ThemeTokens(
    name="light",
    accent="#2563EB",
    ok="#059669",
    warn="#D97706",
    error="#DC2626",
    muted="#64748B",
    info="#0284C7",
    text="#1E293B",
    border="#94A3B8",
)

MONO_THEME = ThemeTokens(
    name="mono",
    accent="bold default",
    ok="default",
    warn="bold default",
    error="bold default",
    muted="dim default",
    info="default",
    text="default",
    border="default",
)

THEMES: dict[str, ThemeTokens] = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
    "mono": MONO_THEME,
}


def get_theme(theme_name: str) -> ThemeTokens:
    return THEMES.get(theme_name.lower(), DARK_THEME)


def is_plain_mode(config: UiConfig, console: Console | None = None) -> bool:
    """Determine whether to use plain mode based on config, terminal caps, and environment."""
    if config.mode == "plain":
        return True
    if config.mode == "rich":
        return False
    # Auto mode detection
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return True
    if console is not None and getattr(console, "record", False):
        return False
    if console is not None and not console.is_terminal:
        return True
    if console is not None and console.width < 70:
        return True
    return False


def get_symbols(ascii_only: bool) -> dict[str, str]:
    if ascii_only:
        return {
            "done": "o",
            "active": "*",
            "pending": ".",
            "arrow": "->",
            "check": "[OK]",
            "warn": "[!]",
            "error": "[X]",
            "bullet": "*",
        }
    return {
        "done": "●",
        "active": "◉",
        "pending": "○",
        "arrow": "→",
        "check": "✓",
        "warn": "⚠",
        "error": "✗",
        "bullet": "•",
    }


def render_environment_table(
    env: EnvironmentInfo,
    config: UiConfig,
    low_ram_warning_gb: float = 4.0,
) -> Panel | Table | str:
    """Compact two-column environment check table with green/amber/red status."""
    theme = get_theme(config.theme)
    symbols = get_symbols(config.ascii_only)

    os_str = getattr(env, "operating_system", getattr(env, "os_name", "Unknown"))
    has_internet = getattr(env, "can_reach_pypi", False) or getattr(env, "can_reach_huggingface", False)
    net_parts = []
    if getattr(env, "can_reach_pypi", False):
        net_parts.append("PyPI")
    if getattr(env, "can_reach_huggingface", False):
        net_parts.append("HuggingFace")
    net_desc = f"Online ({', '.join(net_parts)})" if net_parts else "Offline"

    installed = list(getattr(env, "installed_packages", {}).keys())
    lib_str = f"{', '.join(installed[:6])}..." if installed else "none detected"

    if is_plain_mode(config):
        lines = [
            "Environment Check:",
            f"  OS / Shell: {os_str} ({env.shell_name})",
            f"  Python: {env.python_version}",
            f"  CPU / RAM: {env.cpu_count} cores, {env.ram_total_gb:.1f} GB ({env.ram_free_gb:.1f} GB free)",
            f"  GPU: {'Available' if env.can_train_on_gpu else 'None / CPU only'}",
            f"  Internet: {net_desc}",
            f"  Libraries: {lib_str}",
        ]
        return "\n".join(lines)

    table = Table(box=None, expand=True, show_header=False, pad_edge=False)
    table.add_column("Property", style=theme.muted, width=24)
    table.add_column("Status", style=theme.text)

    # OS & Shell
    table.add_row("OS & Shell", f"{os_str} ({env.shell_name})")

    # Python
    table.add_row("Python", env.python_version)

    # CPU & RAM
    ram_style = theme.ok if env.ram_free_gb >= low_ram_warning_gb else theme.warn
    ram_sym = symbols["check"] if env.ram_free_gb >= low_ram_warning_gb else symbols["warn"]
    ram_text = f"{env.cpu_count} cores | {env.ram_total_gb:.1f} GB total, [{ram_style}]{env.ram_free_gb:.1f} GB free {ram_sym}[/]"
    table.add_row("CPU & RAM", ram_text)

    # GPU
    gpu_style = theme.ok if env.can_train_on_gpu else theme.muted
    gpu_sym = symbols["check"] if env.can_train_on_gpu else symbols["pending"]
    gpu_label = env.gpu_name if env.gpu_name else ("Available " + gpu_sym if env.can_train_on_gpu else "CPU only")
    table.add_row("GPU Acceleration", f"[{gpu_style}]{gpu_label}[/]")

    # Internet
    net_style = theme.ok if has_internet else theme.error
    net_sym = symbols["check"] if has_internet else symbols["error"]
    table.add_row("Internet Access", f"[{net_style}]{net_desc} {net_sym}[/]")

    # Key packages
    table.add_row("Libraries", lib_str)

    return Panel(
        table,
        title="Environment Check",
        title_align="left",
        border_style=theme.border,
        padding=(0, 1),
    )


def render_confirmation_panel(
    task: str,
    metrics: str,
    data_folder: Path,
    limits: Any,
    workspace_folder: Path,
    model_name: str,
    config: UiConfig,
    base_url: str | None = None,
) -> Panel | str:
    """Panel summarizing 'What I will do' before run confirmation."""
    theme = get_theme(config.theme)
    max_exp = getattr(limits, "max_experiment_loops", 12) if limits is not None else 12
    max_eda = getattr(limits, "max_eda_loops", 8) if limits is not None else 8

    if is_plain_mode(config):
        plain_lines = [
            "What I will do:",
            f"  Task: {task}",
            f"  Metrics: {metrics or 'Agents will decide'}",
            f"  Data folder: {data_folder}",
            f"  Max experiment loops: {max_exp}",
            f"  Workspace: {workspace_folder}",
        ]
        if base_url:
            plain_lines.append(f"  Ollama server: {base_url}")
        plain_lines.append(f"  Model: {model_name}")
        return "\n".join(plain_lines)

    grid = Table.grid(padding=(0, 2))
    grid.add_column("Label", style=theme.muted, justify="right")
    grid.add_column("Value", style=theme.text)

    grid.add_row("Task:", f"[{theme.accent}]{escape(task)}[/]")
    grid.add_row("Metrics:", escape(metrics) if metrics else f"[{theme.muted}]Agents will decide[/]")
    grid.add_row("Data folder:", str(data_folder))
    grid.add_row("Loop limits:", f"{max_exp} experiment rounds (EDA cap: {max_eda})")
    grid.add_row("Workspace:", str(workspace_folder))
    if base_url:
        grid.add_row("Ollama server:", str(base_url))
    grid.add_row("Model:", f"[{theme.info}]{escape(model_name)}[/]")

    return Panel(
        grid,
        title="What I will do",
        title_align="left",
        border_style=theme.border,
        padding=(0, 1),
    )


class LiveStatusBar:
    """3-line Live status area: Stage Tracker, Current Agent ('Now'), and Scoreboard."""

    def __init__(
        self,
        config: UiConfig,
        start_time: float,
        limits: Any | None = None,
        model_name: str | None = None,
    ) -> None:
        self.config = config
        self.start_time = start_time
        self.limits = limits
        self.model_name = model_name
        self.theme = get_theme(config.theme)
        self.symbols = get_symbols(config.ascii_only)
        self.current_agent = "Data Engineer"
        self.current_action = "inspecting project directory"
        self.current_reason = ""
        self.action_start_time = start_time
        self.ledger: Ledger | None = None

    def update_status(self, agent_name: str, action: str, reason: str = "") -> None:
        self.current_agent = agent_name
        clean_action = action.replace("\r", " ").replace("\n", " ").strip()
        self.current_action = (
            clean_action if len(clean_action) <= 85 else clean_action[:82].rstrip() + "..."
        )
        clean_reason = reason.replace("\r", " ").replace("\n", " ").strip()
        self.current_reason = (
            clean_reason if len(clean_reason) <= 45 else clean_reason[:42].rstrip() + "..."
        )
        import time
        self.action_start_time = time.monotonic()

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        import time
        if self.ledger is not None:
            action_elapsed = time.monotonic() - self.action_start_time
            total_elapsed = time.monotonic() - self.start_time
            act_text = self.current_action + (f", because {self.current_reason}" if self.current_reason else "")
            lines = build_live_status_renderable(
                self.ledger,
                self.config,
                self.current_agent,
                act_text,
                action_elapsed,
                total_elapsed,
                limits=self.limits,
                model_name=self.model_name,
            )
            for line in lines:
                yield line
            return

        theme = self.theme
        sym = self.symbols

        # Line 1: Stage Tracker
        # Stages: Understand (data_engineer, data_analyst), Plan (ml_lead), Experiment (experiment_loop), Deliver (submit, report, done)
        # Note: caller will supply ledger or we render current known stage
        # Default placeholder if rendered standalone:
        stage_text = Text()
        stage_text.append(f"{sym['active']} Understand  ", style=theme.accent)
        stage_text.append(f"{sym['arrow']}  ", style=theme.muted)
        stage_text.append(f"{sym['pending']} Plan  ", style=theme.muted)
        stage_text.append(f"{sym['arrow']}  ", style=theme.muted)
        stage_text.append(f"{sym['pending']} Experiment  ", style=theme.muted)
        stage_text.append(f"{sym['arrow']}  ", style=theme.muted)
        stage_text.append(f"{sym['pending']} Deliver", style=theme.muted)

        # Line 2: Now
        elapsed_action = int(time.monotonic() - self.action_start_time)
        m, s = divmod(elapsed_action, 60)
        dur_str = f"{m}m {s:02d}s" if m > 0 else f"{s}s"
        now_line = Text()
        now_line.append("Now: ", style=theme.muted)
        now_line.append(f"{self.current_agent}: ", style=f"bold {theme.accent}")
        now_line.append(self.current_action, style=theme.text)
        if self.current_reason:
            now_line.append(f", because {self.current_reason}", style=theme.muted)
        now_line.append(f" ({dur_str})", style="dim")

        # Line 3: Scoreboard placeholder
        elapsed_total = int(time.monotonic() - self.start_time)
        tm, ts = divmod(elapsed_total, 60)
        tot_str = f"{tm}m {ts:02d}s" if tm > 0 else f"{ts}s"
        score_line = Text()
        score_line.append("Scoreboard: ", style=theme.muted)
        score_line.append("Pending baseline", style=theme.text)
        score_line.append(f" | Elapsed: {tot_str}", style="dim")

        yield stage_text
        yield now_line
        yield score_line


def build_live_status_renderable(
    ledger: Ledger,
    config: UiConfig,
    current_agent: str,
    current_action: str,
    action_elapsed_seconds: float,
    total_elapsed_seconds: float,
    limits: Any | None = None,
    model_name: str | None = None,
) -> RenderResult:
    """Construct the 3-line live status block from current state."""
    theme = get_theme(config.theme)
    sym = get_symbols(config.ascii_only)

    phase = ledger.phase

    # 1. Stage Tracker
    # Map from ledger.phase
    # Understand: data_engineer, data_analyst
    # Plan: ml_lead
    # Experiment: experiment_loop
    # Deliver: submitter, submit, report, done
    understand_state = "pending"
    plan_state = "pending"
    exp_state = "pending"
    deliver_state = "pending"

    if phase in ("data_engineer", "data_analyst"):
        understand_state = "active"
    elif phase == "ml_lead":
        understand_state = "done"
        plan_state = "active"
    elif phase == "experiment_loop":
        understand_state = "done"
        plan_state = "done"
        exp_state = "active"
    elif phase in ("submitter", "submit", "report"):
        understand_state = "done"
        plan_state = "done"
        exp_state = "done"
        deliver_state = "active"
    elif phase == "done":
        understand_state = "done"
        plan_state = "done"
        exp_state = "done"
        deliver_state = "done"

    def stage_marker(state: str, name: str) -> tuple[str, str]:
        if state == "done":
            return f"{sym['done']} {name}", theme.ok
        if state == "active":
            return f"{sym['active']} {name}", theme.accent
        return f"{sym['pending']} {name}", theme.muted

    t_under, s_under = stage_marker(understand_state, "Understand")
    t_plan, s_plan = stage_marker(plan_state, "Plan")
    max_loops = getattr(limits, "max_experiment_loops", 12) if limits is not None else 12
    exp_name = f"Experiment (round {ledger.iteration + 1}/{max_loops})" if phase == "experiment_loop" else "Experiment"
    t_exp, s_exp = stage_marker(exp_state, exp_name)
    t_del, s_del = stage_marker(deliver_state, "Deliver")

    line1 = Text()
    line1.append(t_under, style=s_under)
    line1.append(f"  {sym['arrow']}  ", style=theme.muted)
    line1.append(t_plan, style=s_plan)
    line1.append(f"  {sym['arrow']}  ", style=theme.muted)
    line1.append(t_exp, style=s_exp)
    line1.append(f"  {sym['arrow']}  ", style=theme.muted)
    line1.append(t_del, style=s_del)

    # 2. Now
    m, s = divmod(int(action_elapsed_seconds), 60)
    dur_str = f"{m}m {s:02d}s" if m > 0 else f"{s}s"
    line2 = Text()
    line2.append("Now: ", style=theme.muted)
    line2.append(f"{current_agent}: ", style=f"bold {theme.accent}")
    line2.append(current_action, style=theme.text)
    line2.append(f" ({dur_str})", style="dim")

    # 3. Scoreboard
    tm, ts = divmod(int(total_elapsed_seconds), 60)
    tot_str = f"{tm}m {ts:02d}s" if tm > 0 else f"{ts}s"
    best_run = ledger.best_approved_run()
    approved_count = len(ledger.approved_runs())
    total_runs = len(ledger.runs)

    line3 = Text()
    line3.append("Scoreboard: ", style=theme.muted)
    if best_run and best_run.score_mean is not None:
        metric = ledger.brief.primary_metric or "score"
        direction = "higher is better" if ledger.brief.higher_is_better else "lower is better"
        line3.append(f"Best: {best_run.score_mean:.4f} {metric} ({direction})", style=f"bold {theme.ok}")
    else:
        line3.append("No approved baseline yet", style=theme.muted)

    line3.append(f" | Runs: {approved_count}/{total_runs} approved", style=theme.text)
    if model_name:
        line3.append(f" | Model: {model_name}", style="dim")
    line3.append(f" | Elapsed: {tot_str}", style="dim")

    return [line1, line2, line3]


def render_summary_panel(ledger: Ledger, config: UiConfig) -> Panel | str:
    """Build the final end-of-run summary panel from ledger state without LLM calls."""
    theme = get_theme(config.theme)
    symbols = get_symbols(config.ascii_only)

    # Determine status border and headline
    if ledger.status == "done":
        border_color = theme.ok
        status_label = "Completed"
    elif ledger.approved_runs():
        border_color = theme.warn
        status_label = f"Stopped early: {ledger.stop_reason or 'user stopped'}"
    else:
        border_color = theme.error
        status_label = f"Failed: {ledger.stop_reason or ledger.error_message or 'no runs approved'}"

    best_run = ledger.best_approved_run()
    primary_metric = ledger.brief.primary_metric or "score"

    if is_plain_mode(config):
        lines = [
            f"=== {status_label} ===",
            f"Headline: {ledger.final.headline or 'Machine learning project finished.'}",
        ]
        if best_run and best_run.score_mean is not None:
            lines.append(f"Best Model: {best_run.model_family} ({best_run.score_mean:.4f} {primary_metric})")
        if ledger.eda.summary:
            lines.append(f"EDA Summary: {ledger.eda.summary}")
        if ledger.final.predictions_path:
            lines.append(f"Predictions: {ledger.final.predictions_path}")
        if ledger.final.report_path:
            lines.append(f"Report: {ledger.final.report_path}")
        return "\n".join(lines)

    grid = Table.grid(padding=(0, 2))
    grid.add_column("Section", style=theme.muted, justify="right", width=22)
    grid.add_column("Details", style=theme.text)

    # Status line
    grid.add_row("Status:", f"[{border_color}]{status_label}[/]")

    # Best result
    if best_run and best_run.score_mean is not None:
        spread_str = f" (±{best_run.score_std:.4f})" if best_run.score_std else ""
        holdout_str = f", holdout: {best_run.holdout_score:.4f}" if best_run.holdout_score is not None else ""
        res_text = (
            f"[{theme.accent}]{best_run.model_family}[/] achieving "
            f"[{theme.ok}]{best_run.score_mean:.4f}{spread_str}[/] {primary_metric}{holdout_str}"
        )
        grid.add_row("Best Result:", res_text)
    else:
        grid.add_row("Best Result:", f"[{theme.muted}]No model was approved[/]")

    # What the agents learned (EDA & findings)
    if ledger.eda.summary:
        grid.add_row("EDA Summary:", escape(ledger.eda.summary))
    top_findings = ledger.eda.findings[:3]
    if top_findings:
        findings_text = "\n".join(f"{symbols['bullet']} [{escape(f.priority)}] {escape(f.statement)}" for f in top_findings)
        grid.add_row("Top Findings:", findings_text)

    # What helped and what did not
    kept_features = [f for f in ledger.features if f.kept]
    dropped_features = [f for f in ledger.features if not f.kept]
    if kept_features:
        feat_str = ", ".join(f"{escape(f.name)} ({f.score_gain:+.4f})" if f.score_gain else escape(f.name) for f in kept_features[:4])
        grid.add_row("Kept Features:", feat_str)
    if dropped_features:
        grid.add_row("Dead Ends:", f"{len(dropped_features)} feature idea(s) rejected by validation screening")

    # Trust & checks
    approved_count = len(ledger.approved_runs())
    total_runs = len(ledger.runs)
    grid.add_row("Trust & Validation:", f"{approved_count} of {total_runs} model runs approved under cross-validation checks")

    # Files
    file_lines = []
    if ledger.final.predictions_path:
        file_lines.append(f"Predictions: {ledger.final.predictions_path}")
    if ledger.final.report_path:
        file_lines.append(f"Report: {ledger.final.report_path}")
    if file_lines:
        grid.add_row("Output Deliverables:", "\n".join(file_lines))

    # Next steps
    next_cmds = [
        f"View report: {ledger.final.report_path or 'report.md'}",
        "Resume more rounds: mlagent resume",
    ]
    grid.add_row("Next Steps:", "\n".join(next_cmds))

    return Panel(
        grid,
        title=f"Project Summary - {escape(ledger.user_task[:60]) if ledger.user_task else 'ML Task'}",
        title_align="left",
        border_style=border_color,
        padding=(0, 1),
    )
