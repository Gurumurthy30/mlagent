import pytest

from mlagent.config import (
    DEFAULT_CONFIG_TEXT,
    AppConfig,
    ConfigError,
    load_config,
    load_or_create_config,
)


def test_missing_config_file_is_created_with_defaults(tmp_path):
    config_path = tmp_path / "workspace" / "config.yaml"

    config = load_or_create_config(config_path)

    assert config_path.exists()
    assert config == AppConfig()


def test_default_config_text_matches_default_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")

    assert load_config(config_path) == AppConfig()


def test_existing_config_is_not_overwritten(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("limits:\n  max_experiment_loops: 7\n", encoding="utf-8")

    config = load_or_create_config(config_path)

    assert config.limits.max_experiment_loops == 7
    assert config.limits.max_eda_loops == 4  # a missing value falls back to its default


def test_unknown_key_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("limits:\n  max_experiment_loop: 7\n", encoding="utf-8")  # typo

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_invalid_value_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("limits:\n  code_timeout_minutes: 0\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_broken_yaml_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path)


@pytest.mark.parametrize(
    ("written_value", "expected_value"),
    [
        ("auto", "auto"),
        ("always", "always"),
        ("never", "never"),
        ("yes", "always"),
        ("no", "never"),
    ],
)
def test_use_gpu_choices_including_yaml_booleans(tmp_path, written_value, expected_value):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"compute:\n  use_gpu: {written_value}\n", encoding="utf-8")

    assert load_config(config_path).compute.use_gpu == expected_value


def test_base_url_rejects_empty_and_no_scheme(tmp_path):
    config_path = tmp_path / "config.yaml"

    config_path.write_text("model:\n  base_url: ''\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="base_url must start with http:// or https://"):
        load_config(config_path)

    config_path.write_text("model:\n  base_url: localhost:11434\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="base_url must start with http:// or https://"):
        load_config(config_path)


def test_base_url_strips_spaces_and_trailing_slashes(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  base_url: '  http://localhost:11434/  '\n", encoding="utf-8")

    config = load_config(config_path)
    assert config.model.base_url == "http://localhost:11434"

    config_path.write_text("model:\n  base_url: 'https://ollama.com/'\n", encoding="utf-8")
    assert load_config(config_path).model.base_url == "https://ollama.com"
