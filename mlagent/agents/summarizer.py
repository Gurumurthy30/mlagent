"""Summarizer agent: reads compressed project memory and writes README.md.

Synthesizes the complete ML pipeline execution trajectory into an executive
and technical README.md document containing all 8 required sections.
"""

from __future__ import annotations

from pydantic import Field

from mlagent.agents.base import AgentContext, AgentFailedError
from mlagent.graph.state import BaseModel, Ledger
from mlagent.helper.events import EventType
from mlagent.memory.assembler import MemoryAssembler
from mlagent.memory.cards import MemoryCard
from mlagent.memory.store import MemoryStore
from mlagent.prompts.summarizer import SUMMARIZER_PROMPT, SUMMARIZER_ROLE

AGENT_KEY = "summarizer"
AGENT_NAME = "Summarizer"


class SummarizerOutput(BaseModel):
    """Output schema for technical summarization and README generation."""

    thinking: str = Field(description="Review of memory cards, champion trajectory, and findings.")
    cli_summary: str = Field(description="Crisp 1-2 line summary of generated README and project outcome.")
    readme_markdown: str = Field(description="Full README.md content")


def run_summarizer(ledger: Ledger, agent_context: AgentContext) -> SummarizerOutput:
    """Execute technical summarization and output README.md."""
    event_bus = agent_context.event_bus
    workspace = agent_context.workspace
    store = MemoryStore(workspace.root)
    assembler = MemoryAssembler(store)

    event_bus.emit(
        EventType.AGENT_STARTED,
        AGENT_NAME,
        "authoring project README.md from dynamic memory cards",
    )

    schema_hint = SummarizerOutput.model_json_schema()
    prompt = assembler.build_prompt(
        agent_name="summarizer",
        ledger=ledger,
        task=SUMMARIZER_PROMPT,
        schema_hint=str(schema_hint),
    )

    try:
        output = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=SummarizerOutput,
            system_message=SUMMARIZER_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(f"Summarizer generation failed: {exc}") from exc

    # Write README.md to workspace root and project root
    workspace_readme = workspace.root / "README.md"
    workspace_readme.write_text(output.readme_markdown, encoding="utf-8")

    project_readme = workspace.project_folder / "README.md"
    try:
        project_readme.write_text(output.readme_markdown, encoding="utf-8")
    except OSError:
        pass

    # Record readme_summary card into MemoryStore
    summary_card = MemoryCard(
        tag="readme_summary",
        card_id="readme_summary",
        owner="summarizer",
        iteration=ledger.iteration,
        memory_summary=output.cli_summary or "Project README.md generated with all 8 sections.",
        payload={"readme_path": str(workspace_readme)},
        artifact_paths=[str(workspace_readme)],
        priority="core",
    )
    try:
        store.append_card(summary_card)
    except FileExistsError:
        pass

    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"README.md written: {output.cli_summary}",
    )

    return output
