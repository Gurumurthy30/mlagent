"""Unit tests for the Summarizer agent."""

from pathlib import Path
from unittest.mock import MagicMock

from mlagent.agents.base import AgentContext
from mlagent.agents.summarizer import SummarizerOutput, run_summarizer
from mlagent.configs.config import AppConfig
from mlagent.graph.state import Ledger
from mlagent.helper.events import EventBus
from mlagent.helper.workspace import Workspace
from mlagent.memory.cards import (
    ChampionPointerCard,
    FinalDeliverableCard,
    PrepSpecCard,
    ProjectBriefCard,
    ValidationPlanCard,
)
from mlagent.memory.store import MemoryStore

SAMPLE_README_CONTENT = """# Machine Learning Project Summary

## 1. Problem & Data
Predict customer churn using tabular features.

## 2. Key EDA Findings
Target distribution is imbalanced (80/20). Imputed missing continuous values.

## 3. Validation Strategy
Stratified 5-Fold cross-validation with 15% holdout.

## 4. Champion Journey
| Round | Route | Model | Feature Version | CV | Holdout | Status | Lesson |
|---|---|---|---|---|---|---|---|
| 0 | fe_and_model | lightgbm | 1 | 0.8800 | 0.8750 | approved | baseline |

## 5. Final Model
Winning Champion is LightGBM with tree depth 6 and learning rate 0.05.

## 6. Known Limitations & Failure Modes
Low performance on senior customer segment with tenure < 3 months.

## 7. Monitoring & Maintenance
| Feature | Metric | Warning | Retrain | Action |
|---|---|---|---|---|
| monthly_charges | psi | 0.10 | 0.25 | alert |

## 8. Reproduction
Pipeline: artifacts/final/model.pkl
Predictions: artifacts/final/predictions.csv
Splits: artifacts/splits.json
Run command: `mlagent start`
"""


def test_summarizer_produces_all_8_required_readme_sections(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    workspace.create_folders()

    # Seed store with memory cards
    store = MemoryStore(workspace.root)
    store.append_card(
        ProjectBriefCard(
            card_id="project_brief",
            owner="data_engineer",
            memory_summary="Customer churn classification.",
            payload={"task": "churn"},
        )
    )
    store.append_card(
        PrepSpecCard(
            card_id="prep_spec_round_0",
            owner="data_analyst",
            memory_summary="Executive prep summary.",
            payload={"prep": 1},
        )
    )
    store.append_card(
        ValidationPlanCard(
            card_id="validation_plan",
            owner="ml_lead",
            memory_summary="Stratified 5-fold CV.",
            payload={"splits": 5},
        )
    )
    store.append_card(
        FinalDeliverableCard(
            card_id="final_deliverable",
            owner="finalizer",
            memory_summary="Deliverable packaged.",
            payload={"final": 1},
        )
    )
    store.update_champion_pointer(
        ChampionPointerCard(
            card_id="champ_1",
            owner="evaluator",
            memory_summary="Champion v1.",
            payload={"c": 1},
        )
    )

    # Mock LLM Client
    mock_llm = MagicMock()
    mock_llm.ask_structured.return_value = SummarizerOutput(
        thinking="Analyzed all 8 cards and synthesized README.",
        cli_summary="README.md successfully generated with all 8 standard sections.",
        readme_markdown=SAMPLE_README_CONTENT,
    )

    context = AgentContext(
        config=AppConfig(),
        workspace=workspace,
        event_bus=EventBus(),
        llm_client=mock_llm,
        code_runner=MagicMock(),
        environment=MagicMock(),
    )

    ledger = Ledger()
    output = run_summarizer(ledger, context)

    # Verify README.md written to workspace
    readme_path = workspace.root / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")

    # Assert all 8 sections are present
    required_sections = [
        "1. Problem & Data",
        "2. Key EDA Findings",
        "3. Validation Strategy",
        "4. Champion Journey",
        "5. Final Model",
        "6. Known Limitations & Failure Modes",
        "7. Monitoring & Maintenance",
        "8. Reproduction",
    ]
    for section in required_sections:
        assert section in content, f"Missing section '{section}' in README.md"

    # Verify return value
    assert "README.md successfully generated" in output.cli_summary


def test_summarizer_reads_cards_from_store_not_raw_ledger(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    workspace.create_folders()

    store = MemoryStore(workspace.root)
    store.append_card(
        ProjectBriefCard(
            card_id="project_brief",
            owner="data_engineer",
            memory_summary="From memory store brief.",
            payload={"source": "store"},
        )
    )

    mock_llm = MagicMock()
    mock_llm.ask_structured.return_value = SummarizerOutput(
        thinking="ok",
        cli_summary="Done",
        readme_markdown=SAMPLE_README_CONTENT,
    )

    context = AgentContext(
        config=AppConfig(),
        workspace=workspace,
        event_bus=EventBus(),
        llm_client=mock_llm,
        code_runner=MagicMock(),
        environment=MagicMock(),
    )

    # Create empty Ledger with NO raw fields populated
    ledger = Ledger()

    run_summarizer(ledger, context)

    # Assert prompt passed to LLM client contains card from store
    call_args = mock_llm.ask_structured.call_args
    assert call_args is not None
    prompt_sent = call_args.kwargs.get("prompt", "")
    assert "From memory store brief." in prompt_sent
    assert "<memory>" in prompt_sent
    assert "<task>" in prompt_sent
