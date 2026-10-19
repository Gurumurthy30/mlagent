"""Tests verifying Gemma 32B production prompt specifications."""

import json

from mlagent.prompts.coder import CODER_PROMPT, CODER_ROLE
from mlagent.prompts.data_analyst import (
    EDA_ANALYST_ROLE,
    EDA_BLUEPRINT_EXAMPLE,
    EDA_BLUEPRINT_PROMPT,
    EDA_PLAN_PROMPT,
    EDA_READ_EXAMPLE,
    EDA_READ_PROMPT,
    EDA_THINK_EXAMPLE,
)
from mlagent.prompts.data_engineer import (
    DATA_ENGINEER_EXAMPLE,
    DATA_ENGINEER_PROMPT,
    DATA_ENGINEER_ROLE,
)
from mlagent.prompts.evaluator import (
    EVALUATOR_EXAMPLE,
    EVALUATOR_PROMPT,
    EVALUATOR_ROLE,
)
from mlagent.prompts.feature_engineer import (
    FEATURE_ENGINEER_EXAMPLE,
    FEATURE_ENGINEER_PROMPT,
    FEATURE_ENGINEER_ROLE,
)
from mlagent.prompts.finalizer import (
    FINALIZER_EXAMPLE,
    FINALIZER_PROMPT,
    FINALIZER_ROLE,
)
from mlagent.prompts.ml_lead import (
    ML_LEAD_ROLE,
    ML_LEAD_ROUND_0_EXAMPLE,
    ML_LEAD_STRATEGIST_EXAMPLE,
    STRATEGIST_PROMPT,
    VALIDATION_PLAN_PROMPT,
)
from mlagent.prompts.modeler import (
    MODELER_EXAMPLE,
    MODELER_PROMPT,
    MODELER_ROLE,
)
from mlagent.prompts.summarizer import (
    SUMMARIZER_EXAMPLE,
    SUMMARIZER_PROMPT,
    SUMMARIZER_ROLE,
)
from mlagent.prompts.tuner import (
    TUNER_EXAMPLE,
    TUNER_PROMPT,
    TUNER_ROLE,
)


def test_no_system_role_in_prompts():
    """Verify Gemma rule: never emit system role."""
    all_roles = [
        DATA_ENGINEER_ROLE,
        EDA_ANALYST_ROLE,
        ML_LEAD_ROLE,
        FEATURE_ENGINEER_ROLE,
        MODELER_ROLE,
        TUNER_ROLE,
        EVALUATOR_ROLE,
        FINALIZER_ROLE,
        SUMMARIZER_ROLE,
        CODER_ROLE,
    ]
    for role in all_roles:
        assert "<system>" not in role
        assert '"role": "system"' not in role


def test_gemma_turn_tags_and_prefill():
    """Verify all prompts open with user turn and close with model prefill."""
    structured_prompts = [
        DATA_ENGINEER_PROMPT.format(user_task="t", user_metrics_text="m", profile_text="p"),
        EDA_PLAN_PROMPT.format(eda_round=1, max_eda_rounds=3, open_questions_text="none"),
        EDA_READ_PROMPT.format(eda_round=1, questions_text="q", script_output="out"),
        EDA_BLUEPRINT_PROMPT,
        VALIDATION_PLAN_PROMPT,
        STRATEGIST_PROMPT.format(iteration=1),
        FEATURE_ENGINEER_PROMPT.format(new_feature_version=2, parent_feature_version=1),
        MODELER_PROMPT.format(experiment_id="EXP_001", feature_version=1),
        TUNER_PROMPT.format(source_experiment_id="EXP_001", model_family="lightgbm"),
        EVALUATOR_PROMPT.format(experiment_id="EXP_001"),
        FINALIZER_PROMPT.format(champion_experiment_id="EXP_001"),
        SUMMARIZER_PROMPT,
    ]

    for p in structured_prompts:
        assert "<start_of_turn>user" in p
        assert "<end_of_turn>" in p
        assert "<start_of_turn>model" in p
        assert p.rstrip().endswith("{")
        assert "Return only the JSON object." in p


def test_react_reasoning_blocks():
    """Verify reasoning-heavy agents have ReAct blocks and deterministic ones do not."""
    react_prompts = [
        EDA_PLAN_PROMPT.format(eda_round=1, max_eda_rounds=3, open_questions_text="none"),
        EDA_BLUEPRINT_PROMPT,
        VALIDATION_PLAN_PROMPT,
        STRATEGIST_PROMPT.format(iteration=1),
        TUNER_PROMPT.format(source_experiment_id="EXP_001", model_family="lightgbm"),
    ]
    for p in react_prompts:
        assert "<reasoning>" in p
        assert "[observation]" in p
        assert "[thought]" in p
        assert "[action]" in p

    deterministic_prompts = [
        DATA_ENGINEER_PROMPT.format(user_task="t", user_metrics_text="m", profile_text="p"),
        EDA_READ_PROMPT.format(eda_round=1, questions_text="q", script_output="out"),
        FEATURE_ENGINEER_PROMPT.format(new_feature_version=2, parent_feature_version=1),
        MODELER_PROMPT.format(experiment_id="EXP_001", feature_version=1),
        EVALUATOR_PROMPT.format(experiment_id="EXP_001"),
        FINALIZER_PROMPT.format(champion_experiment_id="EXP_001"),
        SUMMARIZER_PROMPT,
    ]
    for p in deterministic_prompts:
        assert "<reasoning>" not in p


def test_few_shot_examples_are_valid_json():
    """Verify all embedded few-shot examples parse cleanly as JSON."""
    examples = [
        DATA_ENGINEER_EXAMPLE,
        EDA_THINK_EXAMPLE,
        EDA_READ_EXAMPLE,
        EDA_BLUEPRINT_EXAMPLE,
        ML_LEAD_ROUND_0_EXAMPLE,
        ML_LEAD_STRATEGIST_EXAMPLE,
        FEATURE_ENGINEER_EXAMPLE,
        MODELER_EXAMPLE,
        TUNER_EXAMPLE,
        EVALUATOR_EXAMPLE,
        FINALIZER_EXAMPLE,
        SUMMARIZER_EXAMPLE,
    ]
    for ex in examples:
        data = json.loads(ex)
        assert isinstance(data, dict)
        assert "thinking" in data or "cli_summary" in data


def test_coder_prompt_structure():
    """Verify Coder prompt formats cleanly with code-prefill."""
    s = CODER_PROMPT.format(
        random_seed=42,
        shell_name="powershell",
        lookups_left=2,
        result_tag="RESULT",
        result_keys_text="- ok: bool",
        working_folder="/tmp",
        script_path="/tmp/script.py",
        digest="digest",
        caller_name="Modeler",
        task_text="train",
        attempt_number=1,
        max_code_attempts=3,
        history_text="none",
    )
    assert "<start_of_turn>user" in s
    assert "<start_of_turn>model" in s
    assert "```python" in s
