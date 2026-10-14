import pytest
from scripted_answers import coder_answer_printing, two_failed_coder_attempts

from mlagent.agents.base import AgentFailedError
from mlagent.agents.feature_engineer import (
    Candidate,
    CandidateList,
    FoundationPlan,
    FoundationStep,
    ScreenedCandidate,
    ScreenResult,
    clean_candidates,
    describe_skipped_steps,
    find_candidate_problem,
    judge_candidates,
    normalize_family,
    propose_foundation,
    run_feature_engineer,
    write_candidates_file,
)
from mlagent.events import EventType
from mlagent.memory import create_new_ledger
from mlagent.state import FeatureRecord

FEATURES_SOURCE = '''"""Basic preparation."""
import json


def prepare_inputs(table):
    return table


if __name__ == "__main__":
    result = {"module_path": __file__, "functions": ["prepare_inputs"], "description": "ok"}
    print("<FEATURES>" + json.dumps(result) + "</FEATURES>")
'''


def features_module_answer() -> str:
    return f"```python\n{FEATURES_SOURCE}\n```"


def foundation_plan_answer(steps=None) -> FoundationPlan:
    if steps is None:
        steps = [
            FoundationStep(
                scope="customer_id",
                action="drop_input",
                reason="id column",
                finding_id="",
            )
        ]
    return FoundationPlan(analysis="analysis of raw inputs", steps=steps)


def round_zero_ledger():
    ledger = create_new_ledger("Predict churn.", "macro F1")
    ledger.brief.summary = "Customer table."
    ledger.plan.primary_metric = "macro_f1"
    ledger.plan.playbook = "Use class weights."
    return ledger


def features_round_ledger(workspace):
    ledger = round_zero_ledger()
    ledger.current_feature_version = 1
    ledger.features = [
        FeatureRecord(
            feature_id="feature_001", name="basic_preparation", kept=True, feature_version=1
        )
    ]
    ledger.focus = "features"
    ledger.iteration = 1
    ledger.eda.feature_ideas = []
    workspace.features_folder.mkdir(parents=True, exist_ok=True)
    workspace.features_file(1).write_text(FEATURES_SOURCE, encoding="utf-8")
    return ledger


def candidate(name, reason="a reason", finding_id="finding_001"):
    code = f"def {name}(table):\n    return table"
    return Candidate(name=name, code=code, reason=reason, finding_id=finding_id)


def screen_answer(candidate_scores: dict, baseline=(0.8, 0.8, 0.8, 0.8, 0.8)) -> str:
    candidates = [
        {"name": name, "scores": scores, "error": "" if scores else "KeyError"}
        for name, scores in candidate_scores.items()
    ]
    return coder_answer_printing(
        "SCREEN", {"baseline_scores": list(baseline), "candidates": candidates}
    )


GOOD_SCORES = [0.82, 0.83, 0.82, 0.84, 0.83]
NOISY_SCORES = [0.95, 0.70, 0.85, 0.72, 0.84]


def test_round_zero_writes_the_first_features_file(make_agent_context, workspace):
    agent_context = make_agent_context([foundation_plan_answer(), features_module_answer()])

    result = run_feature_engineer(round_zero_ledger(), agent_context)

    assert result.changed is True
    assert result.current_feature_version == 1
    assert [(f.feature_id, f.name, f.kept) for f in result.features] == [
        ("feature_001", "basic_preparation", True)
    ]
    assert result.features[0].reason == "ok (1 planned steps)"
    assert workspace.features_file(1).exists()


def test_round_zero_task_says_the_script_is_the_features_file(make_agent_context, workspace):
    agent_context = make_agent_context([foundation_plan_answer(), features_module_answer()])

    run_feature_engineer(round_zero_ledger(), agent_context)

    foundation_prompt, coder_prompt = agent_context.llm_client.received_prompts
    assert "You are the Feature Engineer" in foundation_prompt
    assert f"Write the file {workspace.features_file(1)}" in coder_prompt
    assert "Your script IS the features file" in coder_prompt
    assert "Use class weights." in coder_prompt
    assert "Never use the target or labels" in coder_prompt
    assert "Images, text and audio: loading, fixed size or length, normalization" in coder_prompt
    assert "1. [drop_input] customer_id: id column" in coder_prompt


def test_round_zero_failure_stops_the_run(make_agent_context):
    agent_context = make_agent_context(
        [foundation_plan_answer(), *two_failed_coder_attempts("no data")]
    )

    with pytest.raises(AgentFailedError, match="basic input preparation"):
        run_feature_engineer(round_zero_ledger(), agent_context)


def test_a_features_round_keeps_candidates_with_a_counted_gain(make_agent_context, workspace):
    ledger = features_round_ledger(workspace)
    candidates = [candidate("tenure_bucket"), candidate("noisy_ratio"), candidate("broken_idea")]
    scores = {"tenure_bucket": GOOD_SCORES, "noisy_ratio": NOISY_SCORES, "broken_idea": None}
    agent_context = make_agent_context(
        [CandidateList(candidates=candidates), screen_answer(scores), features_module_answer()]
    )

    result = run_feature_engineer(ledger, agent_context)

    assert result.changed is True
    assert result.current_feature_version == 2
    assert workspace.features_file(2).exists()
    new_records = result.features[1:]
    assert [(r.feature_id, r.name, r.kept) for r in new_records] == [
        ("feature_002", "tenure_bucket", True),
        ("feature_003", "noisy_ratio", False),
        ("feature_004", "broken_idea", False),
    ]
    assert new_records[0].feature_version == 2
    assert new_records[0].score_gain == pytest.approx(0.028)
    assert new_records[1].score_gain == pytest.approx(0.012)
    assert new_records[1].feature_version == 0
    assert new_records[2].score_gain is None
    assert new_records[0].finding_id == "finding_001"


def test_the_prompts_carry_the_source_the_candidates_and_only_the_kept_ones(
    make_agent_context, workspace
):
    ledger = features_round_ledger(workspace)
    ledger.features.append(
        FeatureRecord(
            feature_id="feature_002",
            name="prior_feature",
            reason="[ratio] earlier test",
            kept=True,
            score_gain=0.03,
            feature_version=1,
        )
    )
    candidates = [candidate("tenure_bucket"), candidate("noisy_ratio")]
    scores = {"tenure_bucket": GOOD_SCORES, "noisy_ratio": NOISY_SCORES}
    agent_context = make_agent_context(
        [CandidateList(candidates=candidates), screen_answer(scores), features_module_answer()]
    )

    run_feature_engineer(ledger, agent_context)

    propose_prompt, screening_prompt, version_prompt = agent_context.llm_client.received_prompts
    assert "CURRENT FEATURES FILE (version 1)" in propose_prompt
    assert "def prepare_inputs(table)" in propose_prompt
    assert "Never repeat an idea" in propose_prompt
    assert "tenure_bucket" in screening_prompt and "noisy_ratio" in screening_prompt
    assert "screen_candidates_round_1" in screening_prompt
    assert str(workspace.splits_path) in screening_prompt
    assert "baseline_scores" in screening_prompt
    assert "def tenure_bucket" in version_prompt
    assert "noisy_ratio" not in version_prompt
    assert f"Write the file {workspace.features_file(2)}" in version_prompt


def test_no_counted_gain_means_no_new_version(make_agent_context, workspace):
    ledger = features_round_ledger(workspace)
    agent_context = make_agent_context(
        [
            CandidateList(candidates=[candidate("noisy_ratio")]),
            screen_answer({"noisy_ratio": NOISY_SCORES}),
        ]
    )

    result = run_feature_engineer(ledger, agent_context)

    assert result.changed is False
    assert result.current_feature_version == 1
    assert [(r.name, r.kept) for r in result.features[1:]] == [("noisy_ratio", False)]
    assert len(agent_context.llm_client.received_prompts) == 2
    assert not workspace.features_file(2).exists()


def test_a_failed_screening_leaves_everything_unchanged(make_agent_context, workspace):
    ledger = features_round_ledger(workspace)
    agent_context = make_agent_context(
        [CandidateList(candidates=[candidate("tenure_bucket")]), *two_failed_coder_attempts("boom")]
    )

    result = run_feature_engineer(ledger, agent_context)

    assert result.changed is False
    assert result.features == ledger.features
    assert result.current_feature_version == 1


def test_a_screening_result_that_breaks_the_contract_counts_as_failed(
    make_agent_context, workspace
):
    ledger = features_round_ledger(workspace)
    broken_answer = coder_answer_printing("SCREEN", {"baseline_scores": "none", "candidates": []})
    agent_context = make_agent_context(
        [CandidateList(candidates=[candidate("tenure_bucket")]), broken_answer]
    )

    result = run_feature_engineer(ledger, agent_context)

    assert result.changed is False
    assert result.features == ledger.features


def test_a_failure_to_write_the_new_file_keeps_the_old_version(make_agent_context, workspace):
    ledger = features_round_ledger(workspace)
    agent_context = make_agent_context(
        [
            CandidateList(candidates=[candidate("tenure_bucket")]),
            screen_answer({"tenure_bucket": GOOD_SCORES}),
            *two_failed_coder_attempts("syntax problem"),
        ]
    )
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    result = run_feature_engineer(ledger, agent_context)

    assert result.changed is False
    assert result.current_feature_version == 1
    assert [(r.name, r.kept, r.feature_version) for r in result.features[1:]] == [
        ("tenure_bucket", False, 0)
    ]
    assert any("could not write the new features file" in e.message for e in received_events)


def test_candidates_already_tried_are_not_tested_again(make_agent_context, workspace):
    ledger = features_round_ledger(workspace)
    ledger.features.append(FeatureRecord(feature_id="feature_002", name="tenure_bucket"))
    agent_context = make_agent_context([CandidateList(candidates=[candidate("Tenure Bucket")])])

    result = run_feature_engineer(ledger, agent_context)

    assert result.changed is False
    assert len(agent_context.llm_client.received_prompts) == 1


def test_clean_candidates_normalizes_names_and_drops_unusable_ones():
    messy_candidates = [
        Candidate(name=" Tenure Bucket! ", code="def tenure_bucket(t): ..."),
        Candidate(name="tenure_bucket", code="def duplicate(t): ..."),
        Candidate(name="no_code", code="   "),
        Candidate(name="", code="def x(t): ..."),
        Candidate(name="too_long", code="x" * 1600),
        Candidate(name="already_tried", code="def a(t): ..."),
        Candidate(name="fine", code="def fine(t): ..."),
    ]

    cleaned = clean_candidates(messy_candidates, tried_names={"already_tried"})

    assert [c.name for c in cleaned] == ["tenure_bucket", "fine"]


def test_at_most_twenty_candidates_are_screened():
    many_candidates = [candidate(f"idea_{number}") for number in range(30)]

    assert len(clean_candidates(many_candidates, tried_names=set())) == 20


def test_judging_handles_missing_wrong_length_and_nan_scores():
    screen_result = ScreenResult(
        baseline_scores=[10.0, 10.0, 10.0, 10.0],
        candidates=[
            ScreenedCandidate(name="better", scores=[9.0, 9.1, 8.9, 9.0]),
            ScreenedCandidate(name="short", scores=[9.0, 9.0]),
            ScreenedCandidate(name="failed", scores=None, error="KeyError"),
            ScreenedCandidate(name="nan", scores=[9.0, float("nan"), 9.0, 9.0]),
        ],
    )
    candidates = [candidate(name) for name in ("better", "short", "failed", "nan", "unreported")]

    judged = judge_candidates(candidates, screen_result, higher_is_better=False)

    assert [(c.name, kept) for c, _, kept in judged] == [
        ("better", True),
        ("short", False),
        ("failed", False),
        ("nan", False),
        ("unreported", False),
    ]
    assert judged[0][1] == pytest.approx(1.0)
    assert all(gain is None for _, gain, _ in judged[1:])


def test_the_input_ledger_is_not_changed(make_agent_context, workspace):
    ledger = features_round_ledger(workspace)
    agent_context = make_agent_context(
        [
            CandidateList(candidates=[candidate("tenure_bucket")]),
            screen_answer({"tenure_bucket": GOOD_SCORES}),
            features_module_answer(),
        ]
    )

    run_feature_engineer(ledger, agent_context)

    assert len(ledger.features) == 1
    assert ledger.current_feature_version == 1


def test_events_show_the_screening_and_the_result(make_agent_context, workspace):
    ledger = features_round_ledger(workspace)
    agent_context = make_agent_context(
        [
            CandidateList(candidates=[candidate("tenure_bucket"), candidate("noisy_ratio")]),
            screen_answer({"tenure_bucket": GOOD_SCORES, "noisy_ratio": NOISY_SCORES}),
            features_module_answer(),
        ]
    )
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    run_feature_engineer(ledger, agent_context)

    messages = [e.message for e in received_events if e.agent_name == "Feature Engineer"]
    assert any(m.startswith("screening 2 candidates") for m in messages)
    assert "1 of 2 candidates showed a counted gain" in messages
    assert received_events[-1].event_type == EventType.AGENT_FINISHED
    assert "features file version 2" in messages[-1]


def test_describe_skipped_steps():
    assert describe_skipped_steps(None) == ""
    assert describe_skipped_steps([]) == ""
    assert (
        describe_skipped_steps(
            [
                {"step": 1, "status": "skipped", "note": "column missing"},
                {"step": 2, "status": "done", "note": "completed"},
                {"step": "tenure", "status": "skipped", "note": "constant"},
            ]
        )
        == "1: column missing; tenure: constant"
    )


def test_find_candidate_problem():
    broken_syntax = Candidate(name="bad", code="def bad(: return", reason="x")
    assert "syntax error" in find_candidate_problem(broken_syntax, require_fit_and_apply=False)

    no_func = Candidate(name="no_fn", code="x = 1", reason="x")
    assert find_candidate_problem(no_func, require_fit_and_apply=False) == "no function defined"

    valid_one_fn = Candidate(name="f", code="def f(table): return table", reason="x")
    assert find_candidate_problem(valid_one_fn, require_fit_and_apply=False) == ""
    assert (
        find_candidate_problem(valid_one_fn, require_fit_and_apply=True)
        == "missing fit_<name> and apply_<name>"
    )

    valid_fit_apply = Candidate(
        name="good",
        code="def fit_good(t): return None\ndef apply_good(t, s): return t",
        reason="x",
    )
    assert find_candidate_problem(valid_fit_apply, require_fit_and_apply=True) == ""


def test_write_candidates_file(tmp_path):
    target = tmp_path / "candidates.py"
    good = Candidate(name="c1", code="def c1(t): return t", reason="r1")
    bad = Candidate(name="c2", code="syntax error here", reason="r2")
    accepted, rejected = write_candidates_file([good, bad], target, require_fit_and_apply=False)

    assert [c.name for c in accepted] == ["c1"]
    assert len(rejected) == 1
    assert rejected[0][0].name == "c2"
    assert "syntax error" in rejected[0][1]
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "import pandas as pd" in content
    assert "def c1(t): return t" in content


def test_normalize_family():
    assert normalize_family(" RATIO ") == "ratio"
    assert normalize_family("non_existent_family") == "other"

