"""Loads the settings a user may edit, from mlagent_workspace/config.yaml."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_CONFIG_TEXT = """\
# mlagent settings. Edit and run `mlagent start` again.

model:
  name: gemma4:31b                  # Ollama model tag. It must be pulled already.
  base_url: http://localhost:11434
  temperature: 0.2
  context_length: 16384             # Ollama num_ctx. Higher needs more memory.

compute:
  use_gpu: auto                     # auto = the ML Lead decides per task | always | never

limits:
  max_eda_loops: 4                  # most rounds the Data Analyst may run
  max_experiment_loops: 15          # most rounds of the experiment loop
  code_timeout_minutes: 30          # time limit for one script run
"""


class ConfigError(Exception):
    """Raised when config.yaml cannot be read or contains invalid values."""


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "gemma4:31b"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    context_length: int = Field(default=16384, ge=2048)


class ComputeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_gpu: Literal["auto", "always", "never"] = "auto"

    @field_validator("use_gpu", mode="before")
    @classmethod
    def accept_yaml_booleans(cls, value):
        # YAML turns an unquoted yes/no into True/False, so map them to the matching choice.
        if value is True:
            return "always"
        if value is False:
            return "never"
        return value


class LimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_eda_loops: int = Field(default=4, ge=1)
    max_experiment_loops: int = Field(default=15, ge=1)
    code_timeout_minutes: int = Field(default=30, ge=1)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)


def load_config(config_path: Path) -> AppConfig:
    """Read and validate config.yaml. Raises ConfigError with a readable message."""
    try:
        config_text = config_path.read_text(encoding="utf-8")
        config_values = yaml.safe_load(config_text) or {}
        return AppConfig.model_validate(config_values)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f"Cannot use {config_path}: {error}") from error


def load_or_create_config(config_path: Path) -> AppConfig:
    """Load config.yaml. If it does not exist yet, write the default one first."""
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    return load_config(config_path)
