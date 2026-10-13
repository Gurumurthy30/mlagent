"""Shared pieces for every agent.

AgentContext  the tools an agent uses (LLM, code runner, events, config)
render_prompt fills the {placeholders} of a prompt template when the agent is invoked
The run rules block tells every agent its limits, its machine and the safety rules.
"""

import json
import re
from dataclasses import dataclass

from mlagent.code_runner import CodeRunner
from mlagent.config import AppConfig
from mlagent.digest import build_digest, trim_text
from mlagent.environment_probe import EnvironmentInfo, environment_to_text
from mlagent.events import EventBus
from mlagent.llm_client import LlmClient
from mlagent.memory import Workspace
from mlagent.settings import MAX_CODE_ATTEMPTS, MAX_FACTS_CHARS, RANDOM_SEED
from mlagent.state import Ledger

AGENT_DISPLAY_NAMES = {
    "data_engineer": "Data Engineer",
    "data_analyst": "Data Analyst",
    "ml_lead": "ML Lead",
    "feature_engineer": "Feature Engineer",
    "modeler": "Modeler",
    "evaluator": "Evaluator",
    "tuner": "Tuner",
    "reviewer": "Reviewer",
    "submitter": "Submitter",
    "coder": "Coder",
}

GPU_PREFERENCE_TEXT = {
    "auto": "decide from the task and the machine above",
    "always": "use the GPU whenever the machine can",
    "never": "do not use the GPU",
}

RUN_RULES_TEMPLATE = """\
RUN LIMITS (fixed, respect them):
- EDA rounds used: {eda_rounds_used} of {max_eda_loops}
- Experiment rounds finished: {experiment_rounds_finished} of {max_experiment_loops}
- Code attempts for one coding task: {max_code_attempts}
- Time limit for one script run: {code_timeout_minutes} minutes. Longer runs are killed, \
so keep training short and bounded.
- Random seed to use everywhere: {random_seed}

THIS MACHINE:
{environment_text}
GPU choice from the config: {gpu_preference}

RULES:
- The project folder is read only. Never change, move or delete the user's files.
- Write every output file inside the workspace folder.
- Every score you report must come from a script's printed output. Never invent numbers.
- Start simple. Add complexity only when the validation score improves."""

CHARS_PER_TOKEN = 4
PROMPT_TEXT_ALLOWANCE_CHARS = 3000
MIN_DIGEST_CHARS = 500


class PromptError(Exception):
    """Raised when a prompt template needs a value that was not given."""


class AgentFailedError(Exception):
    """Raised when an agent cannot do its job. The run stops with this message."""


@dataclass
class AgentContext:
    config: AppConfig
    workspace: Workspace
    event_bus: EventBus
    llm_client: LlmClient
    code_runner: CodeRunner
    environment: EnvironmentInfo


def display_name(agent_key: str) -> str:
    return AGENT_DISPLAY_NAMES[agent_key]


PLACEHOLDER_PATTERN = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def render_prompt(template: str, **values: object) -> str:
    """Replace every {name} in the template with its value. A missing value raises PromptError.

    Text that is put into the prompt is never searched for placeholders again, so code or
    JSON inside a value is safe. JSON examples with quoted keys are not placeholders either.
    """

    def fill_placeholder(match: re.Match) -> str:
        placeholder_name = match.group(1)
        if placeholder_name not in values:
            raise PromptError(f"The prompt needs a value for {{{placeholder_name}}}.")
        return str(values[placeholder_name])

    return PLACEHOLDER_PATTERN.sub(fill_placeholder, template)


def build_run_rules(ledger: Ledger, agent_context: AgentContext) -> str:
    config = agent_context.config
    return render_prompt(
        RUN_RULES_TEMPLATE,
        eda_rounds_used=ledger.eda.eda_loops_used,
        max_eda_loops=config.limits.max_eda_loops,
        experiment_rounds_finished=ledger.iteration,
        max_experiment_loops=config.limits.max_experiment_loops,
        max_code_attempts=MAX_CODE_ATTEMPTS,
        code_timeout_minutes=config.limits.code_timeout_minutes,
        random_seed=RANDOM_SEED,
        environment_text=environment_to_text(agent_context.environment),
        gpu_preference=GPU_PREFERENCE_TEXT[config.compute.use_gpu],
    )


def build_common_prompt_values(
    agent_key: str, ledger: Ledger, agent_context: AgentContext
) -> dict[str, str]:
    """The values every agent prompt uses: run_rules, digest, project_path, workspace_path."""
    run_rules = build_run_rules(ledger, agent_context)
    half_of_context_chars = agent_context.config.model.context_length * CHARS_PER_TOKEN // 2
    room_for_digest = half_of_context_chars - len(run_rules) - PROMPT_TEXT_ALLOWANCE_CHARS
    digest = trim_text(build_digest(agent_key, ledger), max(room_for_digest, MIN_DIGEST_CHARS))
    return {
        "run_rules": run_rules,
        "digest": digest or "(nothing in memory yet)",
        "project_path": str(agent_context.workspace.project_folder),
        "workspace_path": str(agent_context.workspace.root),
    }


def facts_to_text(result_data: dict) -> str:
    """Turn the JSON a script printed into text for a prompt, cut to a safe length."""
    return trim_text(
        json.dumps(result_data, indent=1, ensure_ascii=False, default=str), MAX_FACTS_CHARS
    )


def format_numbered_list(lines: list[str]) -> str:
    return "\n".join(f"{number}. {line}" for number, line in enumerate(lines, start=1))
