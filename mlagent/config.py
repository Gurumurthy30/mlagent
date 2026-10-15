"""Configuration management for mlagent.

Loads, validates, and manages configuration from defaults, config.yaml, and CLI overrides.
One unified Pydantic AppConfig model is the single source of truth.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(Exception):
    """Raised when configuration cannot be read, contains typos, or values are out of range."""


class LlmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="gemma4:31b",
        description="Ollama model name (e.g. gemma4:31b, qwen2.5:32b)",
    )
    base_url: str = Field(
        default="https://ollama.com",
        description="Ollama server host address",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the LLM",
    )
    context_length: int = Field(
        default=16384,
        ge=2048,
        le=131072,
        description="Ollama context window in tokens (num_ctx)",
    )
    request_timeout_seconds: int = Field(
        default=900,
        ge=10,
        le=7200,
        description="HTTP request timeout for LLM calls in seconds",
    )
    retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum retries for JSON schema repair attempts",
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        url = value.strip().rstrip("/")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("base_url must start with http:// or https://")
        return url


ModelConfig = LlmConfig


class ComputeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_gpu: Literal["auto", "always", "never"] = Field(
        default="auto",
        description="GPU utilization mode: auto (ML Lead decides per task), always, or never",
    )

    @field_validator("use_gpu", mode="before")
    @classmethod
    def accept_yaml_booleans(cls, value: Any) -> Any:
        if value is True:
            return "always"
        if value is False:
            return "never"
        return value


class LimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_experiment_loops: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Maximum rounds of the experiment loop",
    )
    max_eda_loops: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum rounds for Data Analyst during EDA",
    )
    max_data_engineer_rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum rounds for Data Engineer inspection",
    )
    max_data_reentries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum times EDA can be reopened on Reviewer request",
    )
    code_timeout_minutes: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Execution time limit for a single script run in minutes",
    )
    max_code_attempts: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum retry attempts for Coder to fix script errors",
    )
    max_consecutive_crashes: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Stop rule threshold for consecutive script crashes",
    )
    plateau_rounds: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Stop rule patience: consecutive rounds with no counted gain",
    )
    min_relative_gain: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Minimum relative margin for counted gain when splits are few",
    )
    tune_trial_count: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Number of Optuna tuning trials per tuning round",
    )
    candidates_per_feature_round: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum candidate feature ideas proposed per round",
    )


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_tolerance_relative: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Relative mismatch allowed between logged and recomputed validation scores",
    )
    holdout_gap_std_multiplier: float = Field(
        default=3.0,
        ge=0.1,
        le=10.0,
        description="Multiplier of CV standard deviation allowed for holdout score drop",
    )
    holdout_gap_relative_limit: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Maximum relative drop allowed on holdout score before flagging",
    )
    large_score_jump_std_multiplier: float = Field(
        default=3.0,
        ge=0.1,
        le=10.0,
        description="Score jump threshold (std multiplier) that triggers rejection if leakage flag is set",
    )


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_engineer: int = Field(
        default=8500, ge=1000, le=64000, description="Digest character budget for Data Engineer"
    )
    data_analyst: int = Field(
        default=5000, ge=1000, le=64000, description="Digest character budget for Data Analyst"
    )
    ml_lead: int = Field(
        default=4500, ge=1000, le=64000, description="Digest character budget for ML Lead"
    )
    feature_engineer: int = Field(
        default=16000, ge=2000, le=128000, description="Digest character budget for Feature Engineer"
    )
    modeler: int = Field(
        default=7000, ge=1000, le=64000, description="Digest character budget for Modeler"
    )
    evaluator: int = Field(
        default=5500, ge=1000, le=64000, description="Digest character budget for Evaluator"
    )
    tuner: int = Field(
        default=5500, ge=1000, le=64000, description="Digest character budget for Tuner"
    )
    reviewer: int = Field(
        default=5500, ge=1000, le=64000, description="Digest character budget for Reviewer"
    )
    submitter: int = Field(
        default=4000, ge=1000, le=64000, description="Digest character budget for Submitter"
    )
    coder: int = Field(
        default=8000, ge=1000, le=64000, description="Digest character budget for Coder"
    )
    report_facts: int = Field(
        default=14000, ge=2000, le=128000, description="Context character budget for final report facts"
    )


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_folder_name: str = Field(
        default="mlagent_workspace",
        description="Folder name inside the project directory used for artifacts and memory",
    )
    archived_workspace_prefix: str = Field(
        default="mlagent_workspace_old_",
        description="Prefix for archived previous workspace runs",
    )
    big_file_mb: int = Field(
        default=200,
        ge=1,
        le=10000,
        description="File size in megabytes above which files are flagged as large",
    )
    max_files_in_listing: int = Field(
        default=40,
        ge=5,
        le=500,
        description="Maximum number of files shown in data engineer directory listing",
    )
    ignored_folders: list[str] = Field(
        default_factory=lambda: [".git", ".venv", "venv", "__pycache__", "node_modules"],
        description="Folder names to skip when scanning the project",
    )


class MachineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low_ram_warning_gb: float = Field(
        default=4.0,
        ge=0.5,
        le=64.0,
        description="Free system RAM in GB below which a low memory warning is shown",
    )


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_words: int = Field(
        default=900,
        ge=100,
        le=5000,
        description="Target word count limit for the final Markdown report",
    )


class UiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "rich", "plain"] = Field(
        default="auto",
        description="Terminal display mode: auto, rich, or plain",
    )
    theme: Literal["dark", "light", "mono"] = Field(
        default="dark",
        description="Color theme: dark, light, or mono",
    )
    verbosity: Literal["quiet", "normal", "detailed"] = Field(
        default="normal",
        description="Event logging verbosity: quiet, normal, or detailed",
    )
    timestamps: bool = Field(
        default=True,
        description="Show timestamps on log output lines",
    )
    show_environment: bool = Field(
        default=True,
        description="Display environment probe check table on startup",
    )
    confirm_before_start: bool = Field(
        default=True,
        description="Ask confirmation before starting execution",
    )
    ascii_only: bool = Field(
        default=False,
        description="Use ASCII characters only instead of Unicode symbols",
    )
    log_width: int = Field(
        default=100,
        ge=40,
        le=300,
        description="Terminal output line width limit",
    )


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LlmConfig = Field(default_factory=LlmConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    machine: MachineConfig = Field(default_factory=MachineConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    ui: UiConfig = Field(default_factory=UiConfig)

    @field_validator(
        "llm",
        "compute",
        "limits",
        "evaluation",
        "context",
        "workspace",
        "machine",
        "report",
        "ui",
        mode="before",
    )
    @classmethod
    def default_if_none(cls, value: Any) -> Any:
        return {} if value is None else value

    @property
    def model(self) -> LlmConfig:
        return self.llm


BASIC_KEYS = {
    ("llm", "name"),
    ("llm", "base_url"),
    ("llm", "context_length"),
    ("compute", "use_gpu"),
    ("limits", "max_experiment_loops"),
    ("limits", "max_eda_loops"),
    ("limits", "code_timeout_minutes"),
    ("limits", "max_code_attempts"),
    ("report", "max_words"),
    ("workspace", "workspace_folder_name"),
    ("ui", "mode"),
    ("ui", "theme"),
    ("ui", "verbosity"),
}


def _format_scalar(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return val
    return str(val)


def build_default_config_text() -> str:
    """Build the default config.yaml with BASIC keys uncommented and ADVANCED keys commented."""
    default_config = AppConfig()
    lines = [
        "# mlagent settings. Edit and run `mlagent start` again.",
        "# Basic settings are uncommented below. Advanced settings are commented.",
        "",
    ]

    for section_name, section_field in AppConfig.model_fields.items():
        section_obj = getattr(default_config, section_name)
        lines.append(f"{section_name}:")
        for field_name, field_info in section_obj.__class__.model_fields.items():
            val = getattr(section_obj, field_name)
            desc = field_info.description or ""
            is_basic = (section_name, field_name) in BASIC_KEYS
            if isinstance(val, list):
                if is_basic:
                    lines.append(f"  {field_name}:  # {desc}")
                    for item in val:
                        lines.append(f"    - {item}")
                else:
                    lines.append(f"  # {field_name}:  # {desc}")
                    for item in val:
                        lines.append(f"  #   - {item}")
            else:
                val_str = _format_scalar(val)
                if is_basic:
                    lines.append(f"  {field_name}: {val_str:<18} # {desc}")
                else:
                    lines.append(f"  # {field_name}: {val_str:<16} # {desc}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


DEFAULT_CONFIG_TEXT = build_default_config_text()


def apply_overrides(config_input: dict[str, Any] | AppConfig, overrides: list[str]) -> Any:
    """Apply --set section.key=value overrides into config dictionary or AppConfig instance."""
    is_app_config = isinstance(config_input, AppConfig)
    result = config_input.model_dump() if is_app_config else copy.deepcopy(config_input)

    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Invalid override '{override}'. Format must be section.key=value")
        key_path, raw_value = override.split("=", 1)
        parts = key_path.strip().split(".")
        if len(parts) != 2:
            raise ConfigError(f"Invalid override key '{key_path}'. Format must be section.key")
        section, key = parts[0].strip(), parts[1].strip()
        try:
            parsed_value = yaml.safe_load(raw_value)
        except yaml.YAMLError as err:
            raise ConfigError(f"Invalid value in override '{override}': {err}") from err
        if section not in result or not isinstance(result[section], dict):
            result[section] = {}
        result[section][key] = parsed_value

    if is_app_config:
        try:
            return AppConfig.model_validate(result)
        except ValidationError as err:
            raise ConfigError(f"Configuration error: {err}") from err
    return result


def load_config(
    config_path: Path | None = None,
    overrides: list[str] | None = None,
) -> AppConfig:
    """Read, override, and validate configuration."""
    raw_dict: dict[str, Any] = {}
    if config_path is not None and config_path.exists():
        try:
            content = config_path.read_text(encoding="utf-8")
            raw_dict = yaml.safe_load(content) or {}
            if not isinstance(raw_dict, dict):
                raise ConfigError(f"{config_path} must be a YAML mapping.")
        except (OSError, yaml.YAMLError) as error:
            raise ConfigError(f"Cannot read {config_path}: {error}") from error

    if overrides:
        raw_dict = apply_overrides(raw_dict, overrides)

    try:
        return AppConfig.model_validate(raw_dict)
    except ValidationError as err:
        errors = []
        for e in err.errors():
            loc = ".".join(str(p) for p in e["loc"])
            msg = e["msg"]
            errors.append(f"'{loc}': {msg}")
        raise ConfigError("Configuration error: " + "; ".join(errors)) from err


def load_or_create_config(
    config_path: Path,
    overrides: list[str] | None = None,
) -> AppConfig:
    """Load config.yaml. If it does not exist yet, write the default one first."""
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    return load_config(config_path, overrides=overrides)


def get_effective_config_details(
    config_path: Path | None = None,
    overrides: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return effective configuration values with their source (default, file, or cli)."""
    raw_file_dict: dict[str, Any] = {}
    if config_path is not None and config_path.exists():
        try:
            content = config_path.read_text(encoding="utf-8")
            raw_file_dict = yaml.safe_load(content) or {}
            if not isinstance(raw_file_dict, dict):
                raw_file_dict = {}
        except (OSError, yaml.YAMLError):
            raw_file_dict = {}

    cli_keys: set[tuple[str, str]] = set()
    if overrides:
        for override in overrides:
            if "=" in override:
                key_path = override.split("=", 1)[0].strip()
                parts = key_path.split(".")
                if len(parts) == 2:
                    cli_keys.add((parts[0].strip(), parts[1].strip()))

    effective_config = load_config(config_path, overrides=overrides)
    details: list[dict[str, Any]] = []

    for section_name, section_field in AppConfig.model_fields.items():
        section_obj = getattr(effective_config, section_name)
        for field_name, field_info in section_obj.__class__.model_fields.items():
            val = getattr(section_obj, field_name)
            is_basic = (section_name, field_name) in BASIC_KEYS
            if (section_name, field_name) in cli_keys:
                source = "cli"
            elif (
                section_name in raw_file_dict
                and isinstance(raw_file_dict[section_name], dict)
                and field_name in raw_file_dict[section_name]
            ):
                source = "file"
            else:
                source = "default"
            details.append(
                {
                    "section": section_name,
                    "key": field_name,
                    "value": val,
                    "source": source,
                    "tier": "basic" if is_basic else "advanced",
                    "description": field_info.description or "",
                }
            )
    return details


def generate_config_markdown_table() -> str:
    """Generate Markdown documentation table for all config settings."""
    default_config = AppConfig()
    lines = [
        "| Section | Key | Default | Tier | Description |",
        "|---|---|---|---|---|",
    ]
    for section_name, section_field in AppConfig.model_fields.items():
        section_obj = getattr(default_config, section_name)
        for field_name, field_info in section_obj.__class__.model_fields.items():
            val = getattr(section_obj, field_name)
            desc = field_info.description or ""
            is_basic = (section_name, field_name) in BASIC_KEYS
            tier = "basic" if is_basic else "advanced"
            lines.append(f"| `{section_name}` | `{field_name}` | `{val}` | {tier} | {desc} |")
    return "\n".join(lines)
