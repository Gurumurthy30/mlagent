"""Tests for config.py: invariants, validation bounds, forbidden extras, and precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlagent.config import (
    AppConfig,
    ConfigError,
    apply_overrides,
    build_default_config_text,
    generate_config_markdown_table,
    load_config,
)


@pytest.mark.unit
def test_default_config_validates() -> None:
    """AppConfig defaults instantiate and validate without error, regardless of exact default values."""
    cfg = AppConfig()
    assert isinstance(cfg, AppConfig)
    for section_name in AppConfig.model_fields:
        section = getattr(cfg, section_name)
        assert section is not None
        assert len(type(section).model_fields) > 0


@pytest.mark.unit
def test_all_config_fields_have_descriptions() -> None:
    """Every field in every section model has a non-empty description."""
    cfg = AppConfig()
    for section_name in AppConfig.model_fields:
        section = getattr(cfg, section_name)
        for field_name, finfo in type(section).model_fields.items():
            desc = finfo.description
            assert desc is not None and len(desc.strip()) > 0, (
                f"Field '{section_name}.{field_name}' must have a description."
            )


@pytest.mark.unit
def test_numeric_fields_have_bounds() -> None:
    """Every numeric configuration field enforces bounds (ge/gt/le/lt)."""
    cfg = AppConfig()
    for section_name in AppConfig.model_fields:
        section = getattr(cfg, section_name)
        for field_name, finfo in type(section).model_fields.items():
            ann_str = str(finfo.annotation)
            if "int" in ann_str or "float" in ann_str:
                has_bound = any(
                    hasattr(m, "ge") or hasattr(m, "gt") or hasattr(m, "le") or hasattr(m, "lt")
                    for m in finfo.metadata
                )
                assert has_bound, f"Numeric field '{section_name}.{field_name}' must have bounds."


@pytest.mark.unit
def test_unknown_keys_rejected_with_key_name(tmp_path: Path) -> None:
    """Typo keys or unknown inputs are rejected with the key named."""
    unknown_key = "arbitrary_unknown_typo_key_xyz"
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"limits:\n  {unknown_key}: 42\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_config(cfg_file)
    assert unknown_key in str(exc_info.value)


@pytest.mark.unit
def test_init_config_roundtrip(tmp_path: Path) -> None:
    """Default config text produced by build_default_config_text round-trips into AppConfig."""
    default_text = build_default_config_text()
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(default_text, encoding="utf-8")

    loaded = load_config(cfg_file)
    default_cfg = AppConfig()

    for section_name in AppConfig.model_fields:
        loaded_sec = getattr(loaded, section_name)
        default_sec = getattr(default_cfg, section_name)
        for field_name in type(default_sec).model_fields:
            assert getattr(loaded_sec, field_name) == getattr(default_sec, field_name)


@pytest.mark.unit
def test_precedence_cli_overrides_file_and_defaults(tmp_path: Path) -> None:
    """Precedence invariant: CLI override > config file > defaults."""
    default_cfg = AppConfig()

    # Pick dynamic test values distinct from defaults
    file_model = f"{default_cfg.llm.name}_from_file"
    cli_model = f"{default_cfg.llm.name}_from_cli"
    file_rounds = default_cfg.limits.max_experiment_loops + 5
    cli_eda = default_cfg.limits.max_eda_loops + 2

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"llm:\n  name: '{file_model}'\nlimits:\n  max_experiment_loops: {file_rounds}\n",
        encoding="utf-8",
    )

    # 1. File overrides default
    from_file = load_config(cfg_file)
    assert from_file.llm.name == file_model
    assert from_file.limits.max_experiment_loops == file_rounds
    assert from_file.limits.max_eda_loops == default_cfg.limits.max_eda_loops  # untouched default

    # 2. CLI overrides file and defaults
    cli_overridden = apply_overrides(
        from_file, [f"llm.name={cli_model}", f"limits.max_eda_loops={cli_eda}"]
    )
    assert cli_overridden.llm.name == cli_model
    assert cli_overridden.limits.max_experiment_loops == file_rounds  # kept from file
    assert cli_overridden.limits.max_eda_loops == cli_eda  # overridden from CLI


@pytest.mark.unit
def test_markdown_config_table_contains_all_model_keys() -> None:
    """Markdown configuration table dynamically includes every key declared in AppConfig."""
    table = generate_config_markdown_table()
    cfg = AppConfig()
    for section_name in AppConfig.model_fields:
        section = getattr(cfg, section_name)
        for field_name in type(section).model_fields:
            assert f"`{section_name}`" in table
            assert f"`{field_name}`" in table


@pytest.mark.unit
def test_environment_variable_model_and_base_url_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OLLAMA_MODEL and OLLAMA_BASE_URL are read when not set in config file."""
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:32b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-ollama:11434")

    # 1. No config file
    cfg_empty = load_config(tmp_path / "nonexistent.yaml")
    assert cfg_empty.llm.name == "qwen2.5:32b"
    assert cfg_empty.llm.base_url == "http://custom-ollama:11434"

    # 2. Config file exists without llm section
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("limits:\n  max_eda_loops: 5\n", encoding="utf-8")
    cfg_with_file = load_config(cfg_file)
    assert cfg_with_file.llm.name == "qwen2.5:32b"
    assert cfg_with_file.llm.base_url == "http://custom-ollama:11434"
    assert cfg_with_file.limits.max_eda_loops == 5


@pytest.mark.unit
def test_llm_context_setting_and_alias(tmp_path: Path) -> None:
    """llm.context defaults to 16384 and works with either context or context_length key."""
    cfg = AppConfig()
    assert cfg.llm.context == 16384
    assert cfg.llm.context_length == 16384

    # With 'context' key
    f1 = tmp_path / "c1.yaml"
    f1.write_text("llm:\n  context: 32768\n", encoding="utf-8")
    loaded1 = load_config(f1)
    assert loaded1.llm.context == 32768
    assert loaded1.llm.context_length == 32768

    # With backward-compatible 'context_length' key
    f2 = tmp_path / "c2.yaml"
    f2.write_text("llm:\n  context_length: 65536\n", encoding="utf-8")
    loaded2 = load_config(f2)
    assert loaded2.llm.context == 65536
    assert loaded2.llm.context_length == 65536


