"""The terminal UI. It prints events and asks the user questions.

This is the only file that knows about the terminal. Agents never import it.
"""

from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from mlagent.events import AgentEvent, EventType

# event type -> (short label, rich style)
LABEL_AND_STYLE_BY_EVENT_TYPE = {
    EventType.AGENT_STARTED: ("START", "bold cyan"),
    EventType.AGENT_FINISHED: ("DONE", "bold green"),
    EventType.INFO: ("INFO", "white"),
    EventType.DECISION: ("DECIDE", "bold magenta"),
    EventType.CODE_RUN: ("CODE", "blue"),
    EventType.WARNING: ("WARN", "bold yellow"),
    EventType.ERROR: ("ERROR", "bold red"),
}


class ConsoleUI:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def handle_event(self, event: AgentEvent) -> None:
        label, style = LABEL_AND_STYLE_BY_EVENT_TYPE[event.event_type]
        line = Text()
        line.append(event.created_at.strftime("%H:%M:%S") + "  ", style="dim")
        line.append(f"{label:<7}", style=style)
        line.append(f"{event.agent_name:<20}", style="bold")
        line.append(event.message)  # Text does not read [brackets] as markup
        self.console.print(line)

    def show_message(self, text: str, style: str = "") -> None:
        self.console.print(Text(text, style=style))

    def ask_text(self, question: str) -> str:
        """Ask a free-text question. An empty answer is allowed."""
        return Prompt.ask(question, default="", show_default=False, console=self.console)

    def ask_choice(self, question: str, choices: list[str]) -> str:
        return Prompt.ask(question, choices=choices, console=self.console)
