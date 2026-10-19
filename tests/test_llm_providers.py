"""Unit tests for multi-provider LLM configuration and client initialization."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from mlagent.configs.config import AppConfig, LlmConfig, load_config
from mlagent.helper.events import EventBus
from mlagent.helper.llm_client import LlmClient
from mlagent.helper.workspace import Workspace


def test_llm_config_defaults() -> None:
    config = AppConfig()
    assert config.llm.provider == "ollama"
    assert config.llm.name == "gemma4:31b"
    assert config.llm.base_url == "https://ollama.com"
    assert config.llm.temperature == 1.0
    assert config.llm.top_p == 0.95
    assert config.llm.top_k == 64
    assert config.llm.context == 16384


def test_nvidia_provider_defaults() -> None:
    config = AppConfig(llm=LlmConfig(provider="nvidia"))
    assert config.llm.provider == "nvidia"
    assert config.llm.base_url == "https://integrate.api.nvidia.com/v1"
    assert config.llm.name == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert config.llm.temperature == 1.0
    assert config.llm.top_p == 0.95
    assert config.llm.top_k == 64


def test_default_config_yaml_has_top_p_and_top_k() -> None:
    from mlagent.configs.config import build_default_config_text

    text = build_default_config_text()
    assert "temperature: 1.0" in text
    assert "top_p: 0.95" in text
    assert "top_k: 64" in text


def test_load_config_with_nvidia_overrides(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "llm:\n  provider: nvidia\n  name: custom/model\n  api_key: test-key-123\n",
        encoding="utf-8",
    )
    loaded = load_config(cfg_file)
    assert loaded.llm.provider == "nvidia"
    assert loaded.llm.name == "custom/model"
    assert loaded.llm.api_key == "test-key-123"
    assert loaded.llm.base_url == "https://integrate.api.nvidia.com/v1"
    assert loaded.llm.temperature == 1.0
    assert loaded.llm.top_p == 0.95


@patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
def test_llm_client_initializes_nvidia_without_top_k(
    mock_chat_nvidia: MagicMock, tmp_path: Path
) -> None:
    mock_instance = MagicMock()
    mock_chat_nvidia.return_value = mock_instance
    mock_instance.bind.return_value = mock_instance

    workspace = Workspace(tmp_path)
    config = LlmConfig(
        provider="nvidia",
        name="nvidia/nemotron-3.5-super-vl-120b-a12b",
        api_key="nv-test-key",
        temperature=1.0,
        top_p=0.95,
        context=16384,
    )
    client = LlmClient(model_config=config, workspace=workspace, event_bus=EventBus())
    assert client.model_config.provider == "nvidia"
    mock_chat_nvidia.assert_called_once()
    kwargs = mock_chat_nvidia.call_args.kwargs
    assert kwargs["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert kwargs["api_key"] == "nv-test-key"
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.95
    assert kwargs["max_completion_tokens"] == 16384
    # Ensure NIM does not receive top_k
    assert "top_k" not in kwargs
