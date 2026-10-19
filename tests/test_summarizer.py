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

    # Verify README.md written to workspace and report folder
    readme_path = workspace.root / "README.md"
    assert readme_path.exists()
    report_readme_path = workspace.report_folder / "README.md"
    assert report_readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")
    assert report_readme_path.read_text(encoding="utf-8") == content

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


def test_align_and_save_predictions_uses_target_and_1_to_n_ids(tmp_path: Path) -> None:
    import pandas as pd
    from mlagent.agents.finalizer import _align_and_save_predictions
    from mlagent.graph.state import ProjectBrief

    # Scenario 1: No sample submission, input has 0-indexed IDs and 'prediction' column
    src_csv = tmp_path / "raw_preds.csv"
    pd.DataFrame({"id": [0, 1, 2, 3], "prediction": [10.5, 20.1, 15.3, 30.0]}).to_csv(src_csv, index=False)

    dest_1 = tmp_path / "out1" / "predictions.csv"
    dest_2 = tmp_path / "report" / "predictions.csv"
    brief = ProjectBrief(
        task_family="regression",
        target_column="SalePrice",
        id_columns=["Id"],
        primary_metric="rmse",
    )

    _align_and_save_predictions(
        src_path=src_csv,
        dst_paths=[dest_1, dest_2],
        brief=brief,
        project_folder=tmp_path,
    )

    for dst in [dest_1, dest_2]:
        assert dst.exists()
        df = pd.read_csv(dst)
        assert list(df.columns) == ["Id", "SalePrice"]
        # Must be 1 to N (1, 2, 3, 4), NEVER 0 to N-1 (0, 1, 2, 3)
        assert list(df["Id"]) == [1, 2, 3, 4]
        assert list(df["SalePrice"]) == [10.5, 20.1, 15.3, 30.0]

    # Scenario 2: With sample submission (e.g. Digit Recognizer: ImageId, Label)
    from mlagent.graph.state import DatasetProfile, FileProfile
    sample_sub = tmp_path / "sample_submission.csv"
    pd.DataFrame({"ImageId": [1, 2, 3], "Label": [0, 0, 0]}).to_csv(sample_sub, index=False)
    brief.profile = DatasetProfile(
        project_folder=str(tmp_path),
        files=[FileProfile(path="sample_submission.csv", inferred_role="sample_submission")],
    )

    src_sub = tmp_path / "model_out.csv"
    pd.DataFrame({"Id": [99, 99, 99], "prediction": [7, 2, 9]}).to_csv(src_sub, index=False)

    _align_and_save_predictions(
        src_path=src_sub,
        dst_paths=[dest_1, dest_2],
        brief=brief,
        project_folder=tmp_path,
    )

    for dst in [dest_1, dest_2]:
        df = pd.read_csv(dst)
        assert list(df.columns) == ["ImageId", "Label"]
        assert list(df["ImageId"]) == [1, 2, 3]
        assert list(df["Label"]) == [7, 2, 9]


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
