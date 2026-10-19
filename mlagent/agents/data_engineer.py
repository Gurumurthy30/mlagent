"""Data Engineer agent: deterministic profiling and project scope definition.

Inspects the data directory deterministically, computes column and file profiles,
and prompts the LLM to output a structured ProjectScope specifying the ML problem,
modality, task family, target column, ID columns, metrics, and initial EDA questions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Agent runtime context and errors
from mlagent.agents.base import AgentContext, AgentFailedError

# Configuration constants
from mlagent.configs.settings import (
    ARCHIVED_WORKSPACE_PREFIX,
    IGNORED_FOLDER_NAMES,
)

# Graph state schemas
from mlagent.graph.state import (
    ColumnProfile,
    DatasetProfile,
    FileProfile,
    Ledger,
    ProjectBrief,
    ProjectScope,
)

# Event notification types
from mlagent.helper.events import EventType

# Role and prompt templates
from mlagent.memory.assembler import MemoryAssembler
from mlagent.memory.cards import ProjectBriefCard
from mlagent.memory.store import MemoryStore
from mlagent.prompts.data_engineer import DATA_ENGINEER_PROMPT, DATA_ENGINEER_ROLE

AGENT_KEY = "data_engineer"
AGENT_NAME = "Data Engineer"


def _is_ignored_path(path: Path) -> bool:
    # Check if any path segment matches ignored or archived folders
    for part in path.parts:
        if part in IGNORED_FOLDER_NAMES or part.startswith(ARCHIVED_WORKSPACE_PREFIX):
            return True
    return False


def _profile_tabular_file(file_path: Path, size_mb: float) -> tuple[int, int, list[ColumnProfile]]:
    # Read sample data to compute column metrics without exceeding memory limits
    col_profiles: list[ColumnProfile] = []
    row_count = 0
    col_count = 0

    try:
        # For CSV/TSV, read first 1000 rows for distribution analysis
        sep = "\t" if file_path.suffix.lower() == ".tsv" else ","
        if file_path.suffix.lower() in [".csv", ".tsv"]:
            df_sample = pd.read_csv(file_path, sep=sep, nrows=1000)
            col_count = len(df_sample.columns)
            
            # Quick estimate of total row count for files under 100MB
            if size_mb < 100.0:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    row_count = max(0, sum(1 for _ in f) - 1)
            else:
                row_count = 1000  # Conservative lower bound
        elif file_path.suffix.lower() == ".parquet":
            df_sample = pd.read_parquet(file_path)
            row_count = len(df_sample)
            col_count = len(df_sample.columns)
        else:
            return 0, 0, []

        # Analyze individual column distributions
        for col_name in df_sample.columns[:60]:
            series = df_sample[col_name]
            dtype_str = str(series.dtype)
            missing_count = int(series.isna().sum())
            missing_pct = float(missing_count / len(series)) if len(series) > 0 else 0.0
            unique_count = int(series.nunique(dropna=True))
            sample_vals = [str(v) for v in series.dropna().head(3).tolist()]
            
            min_val = None
            max_val = None
            mean_val = None
            std_val = None
            
            # Extract basic numeric statistics if series is float or int
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                cleaned = series.dropna()
                if len(cleaned) > 0:
                    min_val = float(cleaned.min())
                    max_val = float(cleaned.max())
                    mean_val = float(cleaned.mean())
                    std_val = float(cleaned.std()) if len(cleaned) > 1 else 0.0

            col_profiles.append(
                ColumnProfile(
                    name=str(col_name),
                    dtype=dtype_str,
                    missing_count=missing_count,
                    missing_pct=missing_pct,
                    unique_count=unique_count,
                    sample_values=sample_vals,
                    min_val=min_val,
                    max_val=max_val,
                    mean_val=mean_val,
                    std_val=std_val,
                )
            )
    except Exception:
        # Fall back gracefully if file reading fails
        pass

    return row_count, col_count, col_profiles


def profile_data_folder(project_folder: Path) -> DatasetProfile:
    # Scan directory and build deterministic structural profile
    files_profile: list[FileProfile] = []
    train_columns: set[str] = set()
    test_columns: set[str] = set()
    readme_excerpt = ""

    if not project_folder.exists():
        return DatasetProfile(project_folder=str(project_folder))

    # Iterate through all files in directory tree
    for item in sorted(project_folder.rglob("*")):
        if not item.is_file() or _is_ignored_path(item):
            continue

        rel_path = str(item.relative_to(project_folder)).replace("\\", "/")
        size_mb = round(item.stat().st_size / (1024 * 1024), 2)
        suffix = item.suffix.lower()
        lower_name = item.name.lower()

        # Inferred role categorization
        inferred_role = "other"
        if "train" in lower_name:
            inferred_role = "train"
        elif "test" in lower_name:
            inferred_role = "test"
        elif "submission" in lower_name or "sample" in lower_name:
            inferred_role = "sample_submission"
        elif suffix in [".md", ".txt", ".rst"]:
            inferred_role = "description"

        # Read description excerpts
        if inferred_role == "description" and not readme_excerpt:
            try:
                readme_excerpt = item.read_text(encoding="utf-8", errors="ignore")[:1500]
            except Exception:
                pass

        # Tabular analysis
        kind = "other"
        row_count = 0
        col_count = 0
        columns: list[ColumnProfile] = []

        if suffix in [".csv", ".tsv", ".parquet"]:
            kind = "table"
            row_count, col_count, columns = _profile_tabular_file(item, size_mb)
            
            # Track train and test column sets
            col_names = {c.name for c in columns}
            if inferred_role == "train":
                train_columns.update(col_names)
            elif inferred_role == "test":
                test_columns.update(col_names)
        elif suffix in [".png", ".jpg", ".jpeg", ".tiff"]:
            kind = "image_dir"
        elif suffix in [".wav", ".mp3", ".flac"]:
            kind = "audio_dir"
        elif suffix in [".txt", ".json", ".xml"]:
            kind = "text_corpus"

        files_profile.append(
            FileProfile(
                path=rel_path,
                size_mb=size_mb,
                kind=kind,
                inferred_role=inferred_role,
                row_count=row_count,
                col_count=col_count,
                columns=columns,
            )
        )

    # Compute column set differences
    train_minus_test = sorted(list(train_columns - test_columns)) if test_columns else []
    shared_cols = sorted(list(train_columns & test_columns)) if test_columns else sorted(list(train_columns))

    return DatasetProfile(
        project_folder=str(project_folder),
        files=files_profile,
        train_minus_test_columns=train_minus_test,
        shared_columns=shared_cols,
        readme_excerpt=readme_excerpt,
    )


def run_data_engineer(ledger: Ledger, agent_context: AgentContext) -> ProjectScope:
    # Execute Stage 1 and Stage 2 Data Engineer agent
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "analyzing project data folder", ledger=ledger)

    project_folder = agent_context.workspace.project_folder
    
    # 1. Deterministic profile computation
    profile = profile_data_folder(project_folder)
    
    # 2. Build profile text for prompt
    profile_summary_lines = [
        f"Scanned files count: {len(profile.files)}",
        f"Train minus test columns (target candidate): {', '.join(profile.train_minus_test_columns) or 'None'}",
        f"Shared columns count: {len(profile.shared_columns)}",
    ]
    for fp in profile.files[:10]:
        profile_summary_lines.append(
            f"File: {fp.path} ({fp.size_mb} MB, role: {fp.inferred_role}, rows: {fp.row_count}, cols: {fp.col_count})"
        )
        if fp.columns:
            top_cols = [c.name for c in fp.columns[:15]]
            profile_summary_lines.append(f"  Columns: {', '.join(top_cols)}")

    if profile.readme_excerpt:
        profile_summary_lines.append(f"Documentation excerpt:\n{profile.readme_excerpt}")

    profile_text = "\n".join(profile_summary_lines)

    # 3. Request structured ProjectScope from LLM via MemoryAssembler
    store = MemoryStore(agent_context.workspace.root)
    assembler = MemoryAssembler(store)

    task_instruction = DATA_ENGINEER_PROMPT.format(
        user_task=ledger.user_task or "Frame the machine learning task based on project data.",
        user_metrics_text=ledger.user_metrics_text or "not specified",
        profile_text=profile_text,
    )
    prompt = assembler.build_prompt(
        agent_name="data_engineer",
        ledger=ledger,
        task=task_instruction,
        schema_hint=str(ProjectScope.model_json_schema()),
    )

    try:
        scope = agent_context.llm_client.ask_structured(
            agent_name=AGENT_NAME,
            prompt=prompt,
            answer_model=ProjectScope,
            system_message=DATA_ENGINEER_ROLE,
        )
    except Exception as exc:
        raise AgentFailedError(f"Data Engineer failed to scope project: {exc}") from exc

    # 4. Update ledger with project scope and locked brief
    ledger.scope = scope
    ledger.brief = ProjectBrief(
        profile=profile,
        enhanced_problem_statement=scope.enhanced_problem_statement,
        modality=scope.modality,
        task_family=scope.task_family,
        target_column=scope.target_column,
        id_columns=scope.id_columns,
        group_columns=scope.group_columns,
        time_columns=scope.time_columns,
        text_columns=scope.text_columns,
        primary_metric=scope.primary_metric,
        higher_is_better=scope.higher_is_better,
        secondary_metrics=scope.secondary_metrics,
        deliverable_format=scope.deliverable_format,
        data_sufficiency_and_risks=scope.data_sufficiency_and_risks,
    )

    # 5. Write ProjectBriefCard to MemoryStore
    summary_text = (
        f"Problem: {scope.enhanced_problem_statement}. Target: {scope.target_column or 'None'}. "
        f"Family: {scope.task_family}. Metric: {scope.primary_metric}."
    )
    brief_card = ProjectBriefCard(
        tag="project_brief",
        card_id="project_brief",
        owner="data_engineer",
        iteration=0,
        memory_summary=summary_text,
        payload={
            "problem_statement": scope.enhanced_problem_statement,
            "task_family": scope.task_family,
            "target_column": scope.target_column,
            "primary_metric": scope.primary_metric,
            "secondary_metrics": scope.secondary_metrics,
            "risks": scope.data_sufficiency_and_risks,
        },
        artifact_paths=[],
        priority="core",
    )
    try:
        store.append_card(brief_card)
    except FileExistsError:
        pass

    # Seed initial EDA questions from scope
    ledger.eda.open_questions = list(scope.initial_eda_questions)

    # Emit completion event with terminal cli_summary
    event_bus.emit(
        EventType.AGENT_FINISHED,
        AGENT_NAME,
        f"Scoped {scope.task_family} with metric {scope.primary_metric}: {scope.cli_summary}",
    )
    return scope
