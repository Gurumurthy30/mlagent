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

from mlagent.configs.config import UiConfig
from mlagent.graph.state import Ledger
from mlagent.helper.environment_probe import EnvironmentInfo


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

    os_str = getattr(env, "operating_system", "Unknown")
    has_internet = getattr(env, "can_reach_pypi", False) or getattr(env, "can_reach_huggingface", False)
    net_parts = []
    if getattr(env, "can_reach_pypi", False):
        net_parts.append("PyPI")
    if getattr(env, "can_reach_huggingface", False):
        net_parts.append("HuggingFace")
    net_desc = f"Online ({', '.join(net_parts)})" if net_parts else "Offline"

    installed = list(getattr(env, "installed_packages", {}).keys())
    lib_str = f"{', '.join(installed[:6])}..." if installed else "none detected"

    # GPU description
    if env.gpu_name:
        gpu_label = f"{env.gpu_name} ({env.gpu_memory_total_gb:.1f} GB)"
        gpu_has_accel = True
    elif env.torch_mps_available:
        gpu_label = "Apple Silicon (MPS)"
        gpu_has_accel = True
    else:
        gpu_label = "None / CPU only"
        gpu_has_accel = False

    if is_plain_mode(config):
        lines = [
            "Environment Check:",
            f"  OS / Shell: {os_str} ({env.shell_name})",
            f"  Python: {env.python_version}",
            f"  CPU / RAM: {env.cpu_count} cores, {env.ram_total_gb:.1f} GB ({env.ram_free_gb:.1f} GB free)",
            f"  GPU: {gpu_label}",
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
    gpu_style = theme.ok if gpu_has_accel else theme.muted
    gpu_sym = symbols["check"] if gpu_has_accel else symbols["pending"]
    table.add_row("GPU Acceleration", f"[{gpu_style}]{gpu_label} {gpu_sym}[/]")

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


def format_progress_bar(current: int, total: int, width: int = 10, ascii_only: bool = False) -> str:
    """Render a crisp visual progress bar e.g. [██████░░░░] 60%."""
    if total <= 0:
        return ""
    ratio = max(0.0, min(1.0, current / total))
    filled = int(round(ratio * width))
    unfilled = max(0, width - filled)
    fill_char = "#" if ascii_only else "█"
    empty_char = "-" if ascii_only else "░"
    pct = int(ratio * 100)
    return f"[{fill_char * filled}{empty_char * unfilled}] {pct}%"


class LiveStatusBar:
    """3-line Live status area: Stage Tracker, Current Agent ('Now'), and Scoreboard."""

    def __init__(
        self,
        config: UiConfig,
        start_time: float,
        limits: Any | None = None,
        model_name: str | None = None,
        ledger: Ledger | None = None,
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
        self.ledger: Ledger | None = ledger

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
        max_eda = getattr(self.limits, "max_eda_loops", 8) if self.limits is not None else 8

        # Line 1: Stage Tracker
        stage_text = Text()
        stage_text.append(f"{sym['active']} Understand (EDA 1/{max_eda})  ", style=theme.accent)
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

        # Line 3: Scoreboard placeholder with initial progress bar
        elapsed_total = int(time.monotonic() - self.start_time)
        tm, ts = divmod(elapsed_total, 60)
        tot_str = f"{tm}m {ts:02d}s" if tm > 0 else f"{ts}s"
        pbar_init = format_progress_bar(0, max_eda, width=10, ascii_only=self.config.ascii_only)
        score_line = Text()
        score_line.append("Scoreboard: ", style=theme.muted)
        score_line.append("Pending baseline", style=theme.text)
        if pbar_init:
            score_line.append(f" | {pbar_init}", style=f"bold {theme.accent}")
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
    understand_state = "pending"
    plan_state = "pending"
    exp_state = "pending"
    deliver_state = "pending"

    if phase in ("data_engineer", "eda_think", "eda_analyst", "eda_blueprint"):
        understand_state = "active"
    elif phase in ("ml_lead_round_0", "ml_lead"):
        understand_state = "done"
        plan_state = "active"
    elif phase in ("feature_engineer", "modeler", "tuner", "evaluator", "experiment_loop"):
        understand_state = "done"
        plan_state = "done"
        exp_state = "active"
    elif phase in ("finalizer", "summarizer", "submitter", "submit", "report"):
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

    max_eda = getattr(limits, "max_eda_loops", 8) if limits is not None else 8
    max_loops = getattr(limits, "max_experiment_loops", 12) if limits is not None else 12

    eda_count = ledger.eda.eda_loops_used if (ledger.eda and ledger.eda.eda_loops_used) else 0
    if understand_state == "active":
        under_round = min(eda_count + 1, max_eda)
        under_name = f"Understand (EDA {under_round}/{max_eda})"
    elif understand_state == "done" and eda_count > 0:
        under_name = f"Understand ({eda_count} EDAs)"
    else:
        under_name = "Understand"

    t_under, s_under = stage_marker(understand_state, under_name)
    t_plan, s_plan = stage_marker(plan_state, "Plan")

    curr_exp_round = min(ledger.iteration + 1, max_loops)
    exp_name = (
        f"Experiment (round {curr_exp_round}/{max_loops})"
        if exp_state == "active"
        else "Experiment"
    )
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

    # 3. Scoreboard & Progress Bar
    tm, ts = divmod(int(total_elapsed_seconds), 60)
    tot_str = f"{tm}m {ts:02d}s" if tm > 0 else f"{ts}s"
    champ = ledger.champion_run()
    approved_runs = [r for r in ledger.runs if r.promoted_to_champion or r.verdict == "approve"]
    approved_count = len(approved_runs)
    total_runs = len(ledger.runs)

    # Compute progress bar depending on active phase
    if understand_state == "active":
        curr_eda = min(eda_count + 1, max_eda)
        pbar = format_progress_bar(curr_eda, max_eda, width=10, ascii_only=config.ascii_only)
        pbar_label = f"EDA {curr_eda}/{max_eda}"
    elif exp_state == "active":
        curr_round = min(ledger.iteration + 1, max_loops)
        pbar = format_progress_bar(curr_round, max_loops, width=10, ascii_only=config.ascii_only)
        pbar_label = f"Round {curr_round}/{max_loops}"
    elif deliver_state in ("active", "done"):
        pbar = format_progress_bar(max_loops, max_loops, width=10, ascii_only=config.ascii_only)
        pbar_label = "Complete"
    else:
        pbar = format_progress_bar(1, 1, width=10, ascii_only=config.ascii_only)
        pbar_label = "Planning"

    line3 = Text()
    line3.append("Scoreboard: ", style=theme.muted)
    if champ and champ.cv_mean is not None:
        metric = ledger.brief.primary_metric or "score"
        direction = "higher is better" if ledger.brief.higher_is_better else "lower is better"
        cv_err = f"±{champ.cv_std:.4f}" if champ.cv_std is not None else ""
        line3.append(
            f"Best: {champ.cv_mean:.4f}{cv_err} {metric} ({direction})",
            style=f"bold {theme.ok}",
        )
    else:
        line3.append("Pending baseline", style=theme.muted)

    if pbar:
        line3.append(f" | {pbar} ({pbar_label})", style=f"bold {theme.accent}")

    line3.append(f" | Runs: {approved_count}/{total_runs} approved", style=theme.text)
    if model_name:
        line3.append(f" | Model: {model_name}", style="dim")
    line3.append(f" | Elapsed: {tot_str}", style="dim")

    return [line1, line2, line3]


def render_summary_panel(ledger: Ledger, config: UiConfig) -> Panel | str:
    """Build the final end-of-run summary panel from ledger state without LLM calls."""
    theme = get_theme(config.theme)
    symbols = get_symbols(config.ascii_only)

    champ = ledger.champion_run()
    approved_runs = [r for r in ledger.runs if r.promoted_to_champion or r.verdict == "approve"]

    # Determine status border and headline
    if ledger.status == "done":
        border_color = theme.ok
        status_label = "Completed"
    elif approved_runs or champ:
        border_color = theme.warn
        status_label = f"Stopped early: {ledger.stop_reason or 'user stopped'}"
    else:
        border_color = theme.error
        status_label = f"Failed: {ledger.stop_reason or ledger.error_message or 'no runs approved'}"

    primary_metric = ledger.brief.primary_metric or "score"

    if is_plain_mode(config):
        lines = [
            f"=== {status_label} ===",
            f"Status: {ledger.status.upper()} (Phase: {ledger.phase})",
        ]
        if champ and champ.cv_mean is not None:
            ho_text = f", Holdout: {champ.holdout_score:.4f}" if champ.holdout_score is not None else ""
            lines.append(f"Best Model: {champ.experiment_id} ({champ.model_family}, FV {champ.feature_version}) CV={champ.cv_mean:.4f}{ho_text}")
        if ledger.eda.concise_executive_summary:
            lines.append(f"EDA Summary: {ledger.eda.concise_executive_summary}")
        if ledger.final.predictions_path:
            lines.append(f"Predictions: {ledger.final.predictions_path}")
        if ledger.final.serialized_pipeline_path:
            lines.append(f"Pipeline: {ledger.final.serialized_pipeline_path}")
        if ledger.final.report_path:
            lines.append(f"Report: {ledger.final.report_path}")
        return "\n".join(lines)

    grid = Table.grid(padding=(0, 2))
    grid.add_column("Section", style=theme.muted, justify="right", width=22)
    grid.add_column("Details", style=theme.text)

    # Status line
    grid.add_row("Status:", f"[{border_color}]{status_label}[/]")

    # Best result / Champion
    if champ and champ.cv_mean is not None:
        cv_std_str = f" (±{champ.cv_std:.4f})" if champ.cv_std is not None else ""
        holdout_str = f", holdout: {champ.holdout_score:.4f}" if champ.holdout_score is not None else ""
        res_text = (
            f"[{theme.accent}]{champ.experiment_id}[/] ([bold]{champ.model_family}[/], FV {champ.feature_version}) achieving "
            f"[{theme.ok}]{champ.cv_mean:.4f}{cv_std_str}[/] {primary_metric}{holdout_str}"
        )
        grid.add_row("Champion Model:", res_text)
    else:
        grid.add_row("Champion Model:", f"[{theme.muted}]No model was approved[/]")

    # What the agents learned (EDA & findings)
    if ledger.eda.concise_executive_summary:
        grid.add_row("EDA Summary:", escape(ledger.eda.concise_executive_summary))
    top_findings = ledger.eda.findings[:3]
    if top_findings:
        findings_text = "\n".join(
            f"{symbols['bullet']} [{escape(f.priority)}] {escape(f.statement)}" for f in top_findings
        )
        grid.add_row("Top Findings:", findings_text)

    # Hypotheses & Lessons learned
    lessons = [h.lesson_learned for h in ledger.hypotheses if h.lesson_learned]
    if lessons:
        lesson_text = "\n".join(f"{symbols['bullet']} {escape(lesson)}" for lesson in lessons[:3])
        grid.add_row("Lessons Learned:", lesson_text)

    # Trust & validation check stats
    total_runs = len(ledger.runs)
    grid.add_row("Validation Audit:", f"{len(approved_runs)} of {total_runs} model runs approved under generalization audit")

    # Files & Deliverables
    file_lines = []
    if ledger.final.predictions_path:
        file_lines.append(f"Predictions: {ledger.final.predictions_path}")
    if ledger.final.serialized_pipeline_path:
        file_lines.append(f"Pipeline: {ledger.final.serialized_pipeline_path}")
    if ledger.final.report_path:
        file_lines.append(f"Report: {ledger.final.report_path}")
    if file_lines:
        grid.add_row("Deliverables:", "\n".join(file_lines))

    # Next steps
    next_cmds = [
        f"View report: {ledger.final.report_path or 'README.md'}",
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
