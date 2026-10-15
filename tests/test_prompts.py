"""Static tests verifying prompt templates: placeholders, escaping, and formatting."""

from __future__ import annotations

import re
import string

import pytest
from hypothesis import given

from mlagent.agents.coder import CODER_PROMPT
from mlagent.agents.data_analyst import READ_PROMPT as ANALYST_READ_PROMPT
from mlagent.agents.data_analyst import SUMMARY_PROMPT, THINK_PROMPT
from mlagent.agents.data_engineer import READ_PROMPT as ENGINEER_READ_PROMPT
from mlagent.agents.evaluator import VERDICT_PROMPT
from mlagent.agents.feature_engineer import FOUNDATION_PROMPT, PROPOSE_PROMPT
from mlagent.agents.ml_lead import PLAN_PROMPT
from mlagent.agents.modeler import CHOOSE_PROMPT
from mlagent.agents.reviewer import REPORT_PROMPT, REVIEW_PROMPT
from mlagent.config import AppConfig
from mlagent.digest import build_digest, build_report_facts
from mlagent.state import Ledger
from tests.conftest import st_ledger

# Registry of prompt templates and the expected keys their .format() calls supply
PROMPT_REGISTRY: list[tuple[str, str, set[str]]] = [
    (
        "data_engineer.READ_PROMPT",
        ENGINEER_READ_PROMPT,
        {"user_task", "user_metrics", "brief_so_far", "round_number", "max_rounds", "facts_text", "max_files"},
    ),
    (
        "data_analyst.THINK_PROMPT",
        THINK_PROMPT,
        {"digest", "open_questions_text", "round_number", "max_rounds"},
    ),
    (
        "data_analyst.READ_PROMPT",
        ANALYST_READ_PROMPT,
        {"digest", "questions_text", "facts_text"},
    ),
    (
        "data_analyst.SUMMARY_PROMPT",
        SUMMARY_PROMPT,
        {"digest", "findings_text"},
    ),
    (
        "ml_lead.PLAN_PROMPT",
        PLAN_PROMPT,
        {"digest"},
    ),
    (
        "feature_engineer.FOUNDATION_PROMPT",
        FOUNDATION_PROMPT,
        {"digest"},
    ),
    (
        "feature_engineer.PROPOSE_PROMPT",
        PROPOSE_PROMPT,
        {"digest", "existing_features", "max_candidates"},
    ),
    (
        "modeler.CHOOSE_PROMPT",
        CHOOSE_PROMPT,
        {"digest", "round_goal"},
    ),
    (
        "evaluator.VERDICT_PROMPT",
        VERDICT_PROMPT,
        {"digest", "run_line", "checks_text"},
    ),
    (
        "reviewer.REVIEW_PROMPT",
        REVIEW_PROMPT,
        {"digest", "diagnosis_text"},
    ),
    (
        "reviewer.REPORT_PROMPT",
        REPORT_PROMPT,
        {"facts", "max_words"},
    ),
    (
        "coder.CODER_PROMPT",
        CODER_PROMPT,
        {
            "random_seed",
            "shell_name",
            "lookups_left",
            "result_tag",
            "result_keys_text",
            "working_folder",
            "script_path",
            "digest",
            "caller_name",
            "task_text",
            "attempt_number",
            "max_code_attempts",
            "history_text",
        },
    ),
]


def extract_template_placeholders(template: str) -> set[str]:
    """Extract all placeholder field names from a format string using string.Formatter."""
    formatter = string.Formatter()
    placeholders: set[str] = set()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is not None:
            # Handle potential attribute access or indexing if any (e.g. 'obj.attr' -> 'obj')
            root_name = field_name.split(".")[0].split("[")[0]
            placeholders.add(root_name)
    return placeholders


@pytest.mark.unit
@pytest.mark.parametrize("name,template,expected_keys", PROMPT_REGISTRY)
def test_prompt_template_placeholders_match_format_keys(
    name: str, template: str, expected_keys: set[str]
) -> None:
    """Every prompt template's placeholders exactly match the keys its format call supplies."""
    extracted = extract_template_placeholders(template)
    assert extracted == expected_keys, f"Mismatch in {name}: diff={extracted ^ expected_keys}"


@pytest.mark.unit
@pytest.mark.parametrize("name,template,expected_keys", PROMPT_REGISTRY)
def test_prompt_formats_without_error_and_escapes_literal_braces(
    name: str, template: str, expected_keys: set[str]
) -> None:
    """Prompt templates format without error with sample data, preserving escaped JSON braces."""
    dummy_values = {key: f"sample_val_{key}" for key in expected_keys}
    formatted = template.format(**dummy_values)
    assert isinstance(formatted, str)
    assert len(formatted) > 0

    # Ensure every dummy value was injected
    for key, val in dummy_values.items():
        assert val in formatted, f"Key {key} not found in formatted text of {name}"

    # Verify no raw unescaped format placeholders like '{some_name}' remain in the output
    # Note: JSON braces in examples should be formatted as literal '{' and '}'
    unexpanded_placeholders = re.findall(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})", formatted)
    # Filter out valid JSON key patterns like {"key": ...}
    # In JSON examples, { "name": str } -> after formatting looks like {"name": str} or similar
    # If any dummy key name remains in braces, that indicates failure
    for placeholder in unexpanded_placeholders:
        assert placeholder not in expected_keys, (
            f"Unexpanded placeholder '{placeholder}' found in formatted output of {name}"
        )


@pytest.mark.unit
@given(ledger=st_ledger())
def test_rendered_prompts_within_context_budget(ledger: Ledger) -> None:
    """Rendered digests and facts obey character limits for random ledgers."""
    config = AppConfig()

    # Modeler prompt digest
    modeler_digest = build_digest("modeler", ledger, char_limit=config.context.modeler)
    assert len(modeler_digest) <= config.context.modeler

    # Feature engineer prompt digest
    fe_digest = build_digest("feature_engineer", ledger, char_limit=config.context.feature_engineer)
    assert len(fe_digest) <= config.context.feature_engineer

    # Reviewer prompt digest
    reviewer_digest = build_digest("reviewer", ledger, char_limit=config.context.reviewer)
    assert len(reviewer_digest) <= config.context.reviewer

    # Report facts
    report_facts = build_report_facts(ledger, char_limit=config.context.report_facts)
    assert len(report_facts) <= config.context.report_facts
