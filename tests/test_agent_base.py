import pytest

from mlagent.agents.base import (
    AGENT_DISPLAY_NAMES,
    PromptError,
    build_common_prompt_values,
    build_run_rules,
    render_prompt,
)
from mlagent.config import AppConfig
from mlagent.digest import SECTION_BUILDERS_BY_AGENT
from mlagent.memory import create_new_ledger


def test_render_prompt_fills_every_placeholder():
    prompt = render_prompt(
        "Round {loop_number} of {max_loops}: {task}", loop_number=3, max_loops=15, task="go"
    )

    assert prompt == "Round 3 of 15: go"


def test_render_prompt_raises_for_a_missing_value():
    with pytest.raises(PromptError, match="max_loops"):
        render_prompt("Round {loop_number} of {max_loops}", loop_number=3)


def test_render_prompt_ignores_extra_values():
    assert render_prompt("Hello {name}", name="A", unused="B") == "Hello A"


def test_render_prompt_leaves_json_examples_alone():
    template = 'Answer like {"focus": "model", "scores": [0.1]} for {name}. Empty: {}'

    prompt = render_prompt(template, name="the reviewer")

    assert prompt == 'Answer like {"focus": "model", "scores": [0.1]} for the reviewer. Empty: {}'


def test_text_put_into_a_prompt_is_not_searched_for_placeholders_again():
    code_text = "values = {total}  # not a placeholder here"

    prompt = render_prompt("Code:\n{code}", code=code_text)

    assert prompt == f"Code:\n{code_text}"


def test_run_rules_contain_the_limits_from_config_and_ledger(make_agent_context):
    config = AppConfig.model_validate(
        {"limits": {"max_eda_loops": 5, "max_experiment_loops": 9, "code_timeout_minutes": 12}}
    )
    agent_context = make_agent_context(config=config)
    ledger = create_new_ledger("task", "metric")
    ledger.iteration = 3
    ledger.eda.eda_loops_used = 2

    run_rules = build_run_rules(ledger, agent_context)

    assert "EDA rounds used: 2 of 5" in run_rules
    assert "Experiment rounds finished: 3 of 9" in run_rules
    assert "Code attempts for one coding task: 4" in run_rules
    assert "Time limit for one script run: 12 minutes" in run_rules
    assert "Random seed to use everywhere: 42" in run_rules
    assert "OS: TestOS 1" in run_rules
    assert "{" not in run_rules


@pytest.mark.parametrize(
    ("use_gpu", "expected_text"),
    [
        ("auto", "decide from the task"),
        ("always", "use the GPU whenever"),
        ("never", "do not use the GPU"),
    ],
)
def test_run_rules_explain_the_gpu_choice(make_agent_context, use_gpu, expected_text):
    config = AppConfig.model_validate({"compute": {"use_gpu": use_gpu}})
    agent_context = make_agent_context(config=config)

    run_rules = build_run_rules(create_new_ledger("task", "metric"), agent_context)

    assert expected_text in run_rules


def test_common_values_contain_rules_digest_and_paths(make_agent_context):
    agent_context = make_agent_context()
    ledger = create_new_ledger("predict churn", "macro F1")

    values = build_common_prompt_values("data_engineer", ledger, agent_context)

    assert set(values) == {"run_rules", "digest", "project_path", "workspace_path"}
    assert "predict churn" in values["digest"]
    assert values["workspace_path"].endswith("mlagent_workspace")


def test_empty_digest_gets_a_readable_placeholder(make_agent_context):
    agent_context = make_agent_context()

    values = build_common_prompt_values("modeler", create_new_ledger("", ""), agent_context)

    assert "USER TASK" in values["digest"]
    values = build_common_prompt_values("evaluator", create_new_ledger("", ""), agent_context)
    assert values["digest"] == "(nothing in memory yet)"


def test_digest_is_shortened_when_the_model_context_is_small(make_agent_context):
    small_context_config = AppConfig.model_validate({"model": {"context_length": 2048}})
    agent_context = make_agent_context(config=small_context_config)
    ledger = create_new_ledger("predict churn " * 500, "metric")

    values = build_common_prompt_values("data_engineer", ledger, agent_context)

    assert len(values["digest"]) <= 500


def test_every_agent_that_gets_a_digest_has_a_display_name():
    assert set(SECTION_BUILDERS_BY_AGENT) == set(AGENT_DISPLAY_NAMES)
