import pytest
from fake_llm import FakeLlmClient

from mlagent.agents.base import AgentContext
from mlagent.code_runner import CodeRunner
from mlagent.config import AppConfig
from mlagent.environment_probe import EnvironmentInfo, get_shell_name
from mlagent.events import EventBus
from mlagent.memory import Workspace


@pytest.fixture
def workspace(tmp_path):
    new_workspace = Workspace(tmp_path)
    new_workspace.create_folders()
    return new_workspace


@pytest.fixture
def make_agent_context(workspace):
    """Builds an AgentContext with a scripted LLM and a real code runner."""

    def build_agent_context(
        scripted_answers=(), timeout_seconds: float = 30.0, config: AppConfig | None = None
    ) -> AgentContext:
        event_bus = EventBus()
        environment = EnvironmentInfo(
            operating_system="TestOS 1",
            shell_name=get_shell_name(),
            python_version="3.12.0",
            cpu_count=4,
            ram_total_gb=16.0,
            ram_free_gb=8.0,
        )
        return AgentContext(
            config=config or AppConfig(),
            workspace=workspace,
            event_bus=event_bus,
            llm_client=FakeLlmClient(scripted_answers, workspace, event_bus),
            code_runner=CodeRunner(workspace, timeout_seconds),
            environment=environment,
        )

    return build_agent_context
