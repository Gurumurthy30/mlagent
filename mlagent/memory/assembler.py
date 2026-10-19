"""Memory Assembler: constructs tailored, delimited prompts for each agent from memory cards.

Applies priority tiering (core > high > medium > low) and strictly prevents
raw Ledger dumps from leaking into model context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlagent.memory.cards import (
    MemoryCard,
    ProjectBriefCard,
)

if TYPE_CHECKING:
    from mlagent.graph.state import Ledger
    from mlagent.memory.store import MemoryStore


# Protected tags that must never be dropped under budget constraints
PROTECTED_TAGS = {"project_brief", "champion_pointer", "validation_plan"}

# Agent role charters
AGENT_CHARTERS: dict[str, str] = {
    "data_engineer": "You are the Data Engineer: scope problem requirements, modalities, targets, and data risks.",
    "eda_think": "You are the EDA Analyst (Planning): formulate quantitative hypotheses targeting distribution doubts.",
    "eda_blueprint": "You are the EDA Analyst (Blueprint): synthesize findings into column cleaning rules and feature backlog.",
    "ml_lead_round_0": "You are the ML Lead (Round 0): lock in leak-free validation splitting and Tier-1 baseline.",
    "ml_lead": "You are the ML Lead (Strategist): evaluate previous Champion vs Challenger performance and plan next hypothesis.",
    "feature_engineer": "You are the Feature Engineer: construct fold-isolated feature pipelines branching from Champion features.",
    "modeler": "You are the Modeler: train candidate algorithms matching designated complexity tiers across validation folds.",
    "tuner": "You are the Hyperparameter Tuner: execute targeted hyperparameter searches on Champion to close generalization gaps.",
    "evaluator": "You are the Independent Evaluator: audit validation integrity, generalization gaps, and update the Champion ratchet.",
    "finalizer": "You are the Finalizer: package winning models/ensembles, verify deliverables, and specify drift monitoring rules.",
    "summarizer": "You are the Technical Summarizer: review the full development story and author the comprehensive project README.md.",
}


class MemoryAssembler:
    """Selects, orders, and formats memory cards into agent prompts."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    @staticmethod
    def render_card(card: MemoryCard) -> str:
        # Format single card into clean Markdown representation
        artifact_str = ", ".join(card.artifact_paths) if card.artifact_paths else "none"
        return (
            f"[{card.tag}] {card.card_id} (iteration {card.iteration})\n"
            f"{card.memory_summary}\n"
            f"artifacts: {artifact_str}"
        )

    def _select_cards_for_agent(self, agent_name: str, ledger: Ledger) -> list[MemoryCard]:
        # Retrieve cards designated for agent
        cards: list[MemoryCard] = []

        # 1. Project Brief (core for all agents)
        brief_card = self.store.load_latest("project_brief")
        if brief_card:
            cards.append(brief_card)
        elif ledger.brief and ledger.brief.enhanced_problem_statement:
            # Fallback draft card if brief hasn't been committed to store yet
            cards.append(
                ProjectBriefCard(
                    tag="project_brief",
                    card_id="brief_initial",
                    owner="data_engineer",
                    memory_summary=f"Task: {ledger.brief.enhanced_problem_statement} | Metric: {ledger.brief.primary_metric}",
                    payload={"task": ledger.brief.enhanced_problem_statement, "metric": ledger.brief.primary_metric},
                )
            )

        # 2. Agent-specific card selection
        if agent_name == "data_engineer":
            # Profile pointer
            pass

        elif agent_name == "eda_think":
            champ = self.store.load_latest("champion_pointer")
            if champ:
                cards.append(champ)
            findings = self.store.load_cards("eda_finding", limit=6)
            cards.extend(findings)

        elif agent_name == "eda_blueprint":
            findings = self.store.load_cards("eda_finding", limit=30)
            cards.extend(findings)

        elif agent_name == "ml_lead_round_0":
            prep = self.store.load_latest("prep_spec")
            if prep:
                cards.append(prep)
            high_findings = self.store.load_cards("eda_finding", limit=15, filters={"priority": "high"})
            cards.extend(high_findings)

        elif agent_name == "ml_lead":
            champ = self.store.load_latest("champion_pointer")
            if champ:
                cards.append(champ)
            val = self.store.load_latest("validation_plan")
            if val:
                cards.append(val)
            lessons = self.store.load_cards("lesson_learned", limit=5)
            cards.extend(lessons)
            hyps = self.store.load_cards("hypothesis", limit=2)
            cards.extend(hyps)
            verdicts = self.store.load_cards("evaluation_verdict", limit=3)
            cards.extend(verdicts)

        elif agent_name == "feature_engineer":
            champ = self.store.load_latest("champion_pointer")
            if champ:
                cards.append(champ)
            fv = self.store.load_latest("feature_version")
            if fv:
                cards.append(fv)
            prep = self.store.load_latest("prep_spec")
            if prep:
                cards.append(prep)
            hyp = self.store.load_latest("hypothesis")
            if hyp:
                cards.append(hyp)

        elif agent_name == "modeler":
            val = self.store.load_latest("validation_plan")
            if val:
                cards.append(val)
            champ = self.store.load_latest("champion_pointer")
            if champ:
                cards.append(champ)
            fv = self.store.load_latest("feature_version")
            if fv:
                cards.append(fv)
            hyp = self.store.load_latest("hypothesis")
            if hyp:
                cards.append(hyp)
            runs = self.store.load_cards("experiment_run", limit=3)
            cards.extend(runs)

        elif agent_name == "tuner":
            val = self.store.load_latest("validation_plan")
            if val:
                cards.append(val)
            champ_run = self.store.load_latest("experiment_run")
            if champ_run:
                cards.append(champ_run)
            verdicts = self.store.load_cards("evaluation_verdict", limit=3)
            cards.extend(verdicts)

        elif agent_name == "evaluator":
            val = self.store.load_latest("validation_plan")
            if val:
                cards.append(val)
            latest_run = self.store.load_latest("experiment_run")
            if latest_run:
                cards.append(latest_run)
            champ = self.store.load_latest("champion_pointer")
            if champ:
                cards.append(champ)
            verdicts = self.store.load_cards("evaluation_verdict", limit=3)
            cards.extend(verdicts)

        elif agent_name == "finalizer":
            champ = self.store.load_latest("champion_pointer")
            if champ:
                cards.append(champ)
            approved_runs = self.store.load_cards("experiment_run", limit=10)
            cards.extend(approved_runs)
            lessons = self.store.load_cards("lesson_learned", limit=5)
            cards.extend(lessons)

        elif agent_name == "summarizer":
            prep = self.store.load_latest("prep_spec")
            if prep:
                cards.append(prep)
            val = self.store.load_latest("validation_plan")
            if val:
                cards.append(val)
            lessons = self.store.load_cards("lesson_learned", limit=8)
            cards.extend(lessons)
            final_card = self.store.load_latest("final_deliverable")
            if final_card:
                cards.append(final_card)
            champ = self.store.load_latest("champion_pointer")
            if champ:
                cards.append(champ)

        return cards

    def _apply_priority_tiering(self, cards: list[MemoryCard], max_cards: int = 40) -> list[MemoryCard]:
        # Order by priority tier: core -> high -> medium -> low
        priority_order = {"core": 0, "high": 1, "medium": 2, "low": 3}
        
        # Deduplicate while preserving order
        seen_ids: set[str] = set()
        unique_cards: list[MemoryCard] = []
        for c in cards:
            if c.card_id not in seen_ids:
                seen_ids.add(c.card_id)
                unique_cards.append(c)

        if len(unique_cards) <= max_cards:
            return unique_cards

        # Separate protected cards from trimmable cards
        protected: list[MemoryCard] = []
        trimmable: list[MemoryCard] = []

        for c in unique_cards:
            if c.tag in PROTECTED_TAGS or (c.tag == "lesson_learned" and c.priority == "core"):
                protected.append(c)
            else:
                trimmable.append(c)

        # Sort trimmable cards by priority descending (drop low first)
        trimmable.sort(key=lambda c: priority_order.get(c.priority, 2))
        allowed_trimmable = max_cards - len(protected)

        kept_trimmable = trimmable[: max(0, allowed_trimmable)]
        return protected + kept_trimmable

    def build_prompt(
        self,
        agent_name: str,
        ledger: Ledger,
        task: str,
        schema_hint: str = "",
    ) -> str:
        # Build delimited prompt with <system>, <memory>, <task>, <schema>
        charter = AGENT_CHARTERS.get(agent_name, "You are a professional machine learning engineering agent.")

        # 1. Collect and filter cards
        raw_cards = self._select_cards_for_agent(agent_name, ledger)
        tiered_cards = self._apply_priority_tiering(raw_cards)

        # 2. Render cards to Markdown
        rendered_cards = [self.render_card(c) for c in tiered_cards]
        memory_section = "\n\n".join(rendered_cards) if rendered_cards else "No previous memory cards recorded."

        # Add round summary table for ML Lead and Summarizer
        if agent_name in ("ml_lead", "summarizer"):
            summary_rows = ledger.round_summary_table()
            if summary_rows:
                table_text = "\n".join(summary_rows)
                memory_section += f"\n\n[round_summary_table]\n{table_text}"

        prompt_blocks = [
            f"<system>\n{charter}\n</system>",
            f"<memory>\n{memory_section}\n</memory>",
            f"<task>\n{task}\n</task>",
        ]

        if schema_hint:
            prompt_blocks.append(f"<schema>\n{schema_hint}\n</schema>")

        return "\n\n".join(prompt_blocks)
