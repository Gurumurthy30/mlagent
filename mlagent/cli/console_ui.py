"""Terminal UI: scrolling event log, Live status area, styled prompts, and plain fallback.

This module is the only component that knows about the terminal, Rich rendering, and ANSI controls.
It provides thread safety, theme token styling, verbosity filtering, and clean plain terminal degradation.
"""

from __future__ import annotations

import threading
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.prompt import Confirm, Prompt
from rich.text import Text

from mlagent.cli.summary import (
    LiveStatusBar,
    get_symbols,
    get_theme,
    is_plain_mode,
)
from mlagent.configs.config import UiConfig
from mlagent.helper.events import AgentEvent, EventType

# EventType -> (Short uppercase label, descriptive word, rich style key)
LABEL_AND_STYLE_BY_EVENT_TYPE: dict[EventType, tuple[str, str, str]] = {
    EventType.AGENT_STARTED: ("START", "start", "accent"),
    EventType.AGENT_FINISHED: ("DONE", "done", "ok"),
    EventType.INFO: ("INFO", "info", "text"),
    EventType.DECISION: ("DECIDE", "decision", "warn"),
    EventType.CODE_RUN: ("CODE", "code", "info"),
    EventType.WARNING: ("WARN", "warn", "warn"),
    EventType.ERROR: ("ERROR", "error", "error"),
    EventType.RESULT: ("RESULT", "result", "ok"),
    EventType.STATUS: ("STATUS", "status", "muted"),
}


class ConsoleUI:
    """Multi-mode console UI supporting Rich live status, scrolling logs, and plain fallbacks."""

    def __init__(self, console: Console | None = None, config: UiConfig | None = None) -> None:
        self.config = config if config is not None else UiConfig(verbosity="detailed")
        self.console = console or Console(width=self.config.log_width)
        self.theme = get_theme(self.config.theme)
        self.symbols = get_symbols(self.config.ascii_only)
        self._lock = threading.RLock()
        self._is_plain = is_plain_mode(self.config, self.console)
        self._live: Live | None = None
        self._live_status_bar: LiveStatusBar | None = None
        self._start_time: float | None = None
        self._current_agent: str = "System"
        self._current_action: str = "starting up"
        self._current_reason: str = ""
        self._current_ledger: Any | None = None
        self.limits: Any | None = None
        self.model_name: str | None = None

    def set_run_context(self, limits: Any | None = None, model_name: str | None = None) -> None:
        """Store run limits and model name dynamically from config to display in status and summaries."""
        with self._lock:
            self.limits = limits
            self.model_name = model_name
            if self._live_status_bar is not None:
                self._live_status_bar.limits = limits
                self._live_status_bar.model_name = model_name

    def start_live(self) -> None:
        """Start the Rich Live status area (if not in plain mode)."""
        if self._is_plain or self._live is not None:
            return
        import time

        self._start_time = time.monotonic()
        self._live_status_bar = LiveStatusBar(
            self.config,
            self._start_time,
            limits=self.limits,
            model_name=self.model_name,
            ledger=self._current_ledger,
        )
        try:
            self._live = Live(
                self._live_status_bar,
                console=self.console,
                refresh_per_second=4,
                transient=False,
            )
            self._live.start()
        except Exception:
            # Fallback cleanly to plain mode if Live cannot start
            self._live = None
            self._is_plain = True

    def stop_live(self) -> None:
        """Stop the Live status display cleanly."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def update_ledger(self, ledger: Any) -> None:
        """Update active ledger reference on the live status bar."""
        with self._lock:
            self._current_ledger = ledger
            if self._live_status_bar is not None:
                self._live_status_bar.ledger = ledger

    def should_display(self, event_type: EventType) -> bool:
        """Filter events based on verbosity setting."""
        verbosity = self.config.verbosity
        if verbosity == "quiet":
            return event_type in (
                EventType.AGENT_STARTED,
                EventType.AGENT_FINISHED,
                EventType.RESULT,
                EventType.WARNING,
                EventType.ERROR,
            )
        if verbosity == "normal":
            return event_type != EventType.CODE_RUN
        # 'detailed' shows everything
        return True

    def _truncate_status_message(self, message: str, max_length: int | None = None) -> str:
        """Truncate active status message cleanly with ellipses to avoid terminal frame distortion."""
        if max_length is None:
            width = (
                self.console.width if self.console and self.console.width else self.config.log_width
            )
            # Reserve space for Stage/Now prefix, agent name, and elapsed duration
            max_length = max(35, width - 40)
        clean = message.replace("\r", " ").replace("\n", " ").strip()
        if len(clean) > max_length:
            return clean[: max_length - 3].rstrip() + "..."
        return clean

    def update_status(self, agent_name: str, action: str, reason: str = "") -> None:
        """Update active agent status, cleanly truncating long messages with ellipses."""
        with self._lock:
            self._current_agent = agent_name
            self._current_action = self._truncate_status_message(action)
            self._current_reason = (
                self._truncate_status_message(reason, max_length=40) if reason else ""
            )
            if self._live_status_bar is not None and self._live is not None:
                self._live_status_bar.update_status(
                    self._current_agent, self._current_action, self._current_reason
                )

    def handle_event(self, event: AgentEvent) -> None:
        """Format and print an event to the terminal."""
        with self._lock:
            # Sync ledger if event payload carries an active ledger
            if "ledger" in event.data and hasattr(event.data["ledger"], "phase"):
                self.update_ledger(event.data["ledger"])

            # Handle STATUS events specially: updates live line when Live is active
            if event.event_type == EventType.STATUS:
                self._current_agent = event.agent_name
                self._current_action = self._truncate_status_message(event.message)
                raw_reason = str(event.data.get("reason", ""))
                self._current_reason = (
                    self._truncate_status_message(raw_reason, max_length=40) if raw_reason else ""
                )
                if self._live_status_bar is not None and self._live is not None:
                    self._live_status_bar.update_status(
                        self._current_agent, self._current_action, self._current_reason
                    )
                    return

            # When an agent starts, update active live status bar agent and action immediately
            if event.event_type == EventType.AGENT_STARTED:
                self._current_agent = event.agent_name
                self._current_action = self._truncate_status_message(event.message)
                raw_reason = str(event.data.get("reason", ""))
                self._current_reason = (
                    self._truncate_status_message(raw_reason, max_length=40) if raw_reason else ""
                )
                if self._live_status_bar is not None and self._live is not None:
                    self._live_status_bar.update_status(
                        self._current_agent, self._current_action, self._current_reason
                    )

            if not self.should_display(event.event_type):
                return

            # Construct log line
            line = self._format_event_line(event)
            if self._is_plain:
                # In plain mode: strip any ANSI styling, plain print via console
                self.console.print(line.plain, highlight=False)
            else:
                self.console.print(line)

    def _format_event_line(self, event: AgentEvent) -> Text:
        """Construct fixed-column log line with hanging indent and theme styling."""
        label_info = LABEL_AND_STYLE_BY_EVENT_TYPE.get(event.event_type, ("INFO", "info", "text"))
        label_text, _, style_key = label_info

        line = Text()

        # Column 1: Time
        prefix_len = 0
        if self.config.timestamps:
            time_str = event.created_at.strftime("%H:%M:%S") + "  "
            line.append(time_str, style=self.theme.muted)
            prefix_len += len(time_str)

        # Column 2: Label
        label_col = f"{label_text:<8}"
        style_color = getattr(self.theme, style_key, self.theme.text)
        label_style = style_color if style_color.startswith("bold") else f"bold {style_color}"
        line.append(label_col, style=label_style)
        prefix_len += len(label_col)

        # Column 3: Agent
        agent_col = f"{event.agent_name:<18} "
        agent_style = (
            self.theme.text if self.theme.text.startswith("bold") else f"bold {self.theme.text}"
        )
        line.append(agent_col, style=agent_style)
        prefix_len += len(agent_col)

        # Column 4: Message (with hanging indent and semantic theme coloring)
        msg_style = ""
        if event.event_type == EventType.ERROR:
            msg_style = self.theme.error
        elif event.event_type == EventType.WARNING:
            msg_style = self.theme.warn
        elif event.event_type == EventType.RESULT:
            msg_style = self.theme.ok
        elif event.event_type == EventType.DECISION:
            msg_style = self.theme.warn
        elif event.event_type == EventType.AGENT_STARTED:
            msg_style = self.theme.accent
        elif event.event_type == EventType.AGENT_FINISHED:
            msg_style = self.theme.ok

        effective_msg_style = msg_style or self.theme.text
        msg_text = event.message.replace("\r", "")
        lines = msg_text.splitlines() or [""]
        indent_spaces = " " * prefix_len

        line.append(lines[0], style=effective_msg_style)
        for extra_line in lines[1:]:
            line.append("\n" + indent_spaces + extra_line, style=effective_msg_style)

        return line

    def show_message(self, text: str, style: str = "") -> None:
        """Print a message to the console."""
        with self._lock:
            if self._is_plain:
                self.console.print(text, highlight=False)
            else:
                self.console.print(Text(text, style=style))

    def show_header(self, version: str) -> None:
        """Compact startup header line."""
        title = f"mlagent v{version} — autonomous multi-agent machine learning"
        if self._is_plain:
            self.console.print(f"=== {title} ===", highlight=False)
        else:
            self.console.print(f"[bold {self.theme.accent}]{title}[/]")

    def ask_text(self, question: str) -> str:
        """Prompt user for free-form input."""
        with self._lock:
            return Prompt.ask(
                question, default="", show_default=False, console=self.console
            ).strip()

    def ask_required_task(self, question: str) -> str:
        """Prompt for task and reject empty with a friendly message."""
        while True:
            ans = self.ask_text(question)
            if ans:
                return ans
            self.show_message(
                "Please describe the ML task (e.g. 'predict churn using train.csv').",
                style=self.theme.warn,
            )

    def ask_choice(self, question: str, choices: list[str]) -> str:
        with self._lock:
            return Prompt.ask(question, choices=choices, console=self.console)

    def ask_confirm(self, question: str, default: bool = True) -> bool:
        """Prompt user for confirmation (e.g. 'Start? [Y/n]')."""
        with self._lock:
            return Confirm.ask(question, default=default, console=self.console)
