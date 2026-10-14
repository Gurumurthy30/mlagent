from scripted_answers import coder_answer_printing, two_failed_coder_attempts

from mlagent.agents.data_analyst import (
    DEFAULT_CHECKLIST,
    ChecklistDraft,
    CoverageMark,
    EdaSummary,
    NewFinding,
    Question,
    QuestionPlan,
    RoundReading,
    apply_reading,
    run_data_analyst,
)
from mlagent.config import AppConfig
from mlagent.events import EventType
from mlagent.memory import create_new_ledger
from mlagent.state import FeatureIdea, FileInfo

CHECKLIST_ITEMS = ["target distribution", "missing values", "leakage candidates"]
SCRIPT_OUTPUT = {
    "answers": [{"question": "q", "result": "churn rate is 12 percent"}],
    "problems": [],
}


def ledger_with_brief():
    ledger = create_new_ledger("Predict which customers churn.", "macro F1")
    ledger.brief.summary = "Customer table with a churn label."
    ledger.brief.modality = "tabular"
    ledger.brief.task_family = "classification"
    ledger.brief.primary_metric = "macro_f1"
    ledger.brief.files = [FileInfo(path="train.csv", kind="table", role="train")]
    return ledger


def checklist_answer(items=CHECKLIST_ITEMS):
    return ChecklistDraft(items=list(items))


def think_answer(*covers_lists):
    questions = [Question(text=f"question {n}", covers=c) for n, c in enumerate(covers_lists, 1)]
    return QuestionPlan(questions=questions)


def reading_answer(findings=(), covered=()):
    return RoundReading(
        findings=[NewFinding(statement=text, priority="high") for text in findings],
        coverage=[CoverageMark(item=item, status="covered") for item in covered],
    )


def summary_answer(**changes):
    values = {
        "issues": ["target is imbalanced"],
        "feature_ideas": [FeatureIdea(description="tenure bucket", finding_id="finding_001")],
        "leakage_risks": ["refund_date leaks churn"],
        "validation_hints": ["use stratified folds"],
        "model_hints": ["gradient boosting"],
    }
    values.update(changes)
    return EdaSummary(**values)


def coder_answer():
    return coder_answer_printing("FINDINGS", SCRIPT_OUTPUT)


def test_a_full_first_pass_covers_the_checklist_and_writes_the_summary(make_agent_context):
    agent_context = make_agent_context(
        [
            checklist_answer(),
            think_answer(CHECKLIST_ITEMS),
            coder_answer(),
            reading_answer(["Churn rate is 12 percent."], covered=CHECKLIST_ITEMS),
            summary_answer(),
        ]
    )

    eda = run_data_analyst(ledger_with_brief(), agent_context)

    assert set(eda.checklist.values()) == {"covered"}
    assert [finding.finding_id for finding in eda.findings] == ["finding_001"]
    assert eda.findings[0].priority == "high"
    assert eda.eda_loops_used == 1
    assert eda.issues == ["target is imbalanced"]
    assert eda.feature_ideas[0].finding_id == "finding_001"
    assert eda.leakage_risks == ["refund_date leaks churn"]
    assert eda.validation_hints == ["use stratified folds"]
    assert eda.model_hints == ["gradient boosting"]
    assert len(agent_context.llm_client.received_prompts) == 5


def test_the_prompts_carry_the_round_counter_the_open_items_and_the_output(make_agent_context):
    agent_context = make_agent_context(
        [
            checklist_answer(),
            think_answer(["missing values"]),
            coder_answer(),
            reading_answer(["Churn rate is 12 percent."], covered=CHECKLIST_ITEMS),
            summary_answer(),
        ]
    )

    run_data_analyst(ledger_with_brief(), agent_context)

    prompts = agent_context.llm_client.received_prompts
    assert "Customer table with a churn label." in prompts[0]
    assert "THIS IS EDA ROUND 1 OF 4." in prompts[1]
    assert "- missing values" in prompts[1]
    assert "question 1" in prompts[2] and "Never fully load a file larger than 200 MB" in prompts[2]
    assert "<FINDINGS>JSON</FINDINGS>" in prompts[2]
    assert "churn rate is 12 percent" in prompts[3]
    assert "question 1 (covers: missing values)" in prompts[3]
    assert "Rules: use only the text above." in prompts[4]
    assert "finding_001" in prompts[4]


def test_the_loop_continues_while_items_are_open_and_stops_when_all_are_covered(
    make_agent_context,
):
    agent_context = make_agent_context(
        [
            checklist_answer(),
            think_answer(["target distribution"]),
            coder_answer(),
            reading_answer(["Fact one."], covered=["target distribution"]),
            think_answer(["missing values", "leakage candidates"]),
            coder_answer(),
            reading_answer(["Fact two."], covered=["missing values", "leakage candidates"]),
            summary_answer(),
        ]
    )

    eda = run_data_analyst(ledger_with_brief(), agent_context)

    assert eda.eda_loops_used == 2
    assert [finding.statement for finding in eda.findings] == ["Fact one.", "Fact two."]
    assert "THIS IS EDA ROUND 2 OF 4." in agent_context.llm_client.received_prompts[4]


def test_the_round_limit_from_the_config_ends_the_loop_with_items_still_open(make_agent_context):
    config = AppConfig.model_validate({"limits": {"max_eda_loops": 2}})
    rounds = []
    for number in (1, 2):
        rounds += [
            think_answer(["missing values"]),
            coder_answer(),
            reading_answer([f"Fact {number}."]),
        ]
    agent_context = make_agent_context(
        [checklist_answer(), *rounds, summary_answer()], config=config
    )

    eda = run_data_analyst(ledger_with_brief(), agent_context)

    assert eda.eda_loops_used == 2
    assert set(eda.checklist.values()) == {"open"}
    assert "THIS IS EDA ROUND 2 OF 2." in agent_context.llm_client.received_prompts[4]


def test_a_round_without_new_findings_ends_the_loop(make_agent_context):
    agent_context = make_agent_context(
        [
            checklist_answer(),
            think_answer(["target distribution"]),
            coder_answer(),
            reading_answer(["Fact one."], covered=["target distribution"]),
            think_answer(["missing values"]),
            coder_answer(),
            reading_answer(["fact one."]),
            summary_answer(),
        ]
    )

    eda = run_data_analyst(ledger_with_brief(), agent_context)

    assert eda.eda_loops_used == 2
    assert len(eda.findings) == 1
    assert eda.checklist["missing values"] == "open"


def test_a_failed_script_does_not_stop_the_run_and_is_noted_as_an_issue(make_agent_context):
    agent_context = make_agent_context(
        [
            checklist_answer(),
            think_answer(["missing values"]),
            *two_failed_coder_attempts("no data"),
        ]
    )

    eda = run_data_analyst(ledger_with_brief(), agent_context)

    assert eda.findings == []
    assert eda.eda_loops_used == 1
    assert eda.issues == ["An EDA script failed: RuntimeError: no data"]
    assert len(agent_context.llm_client.received_prompts) == 4


def test_an_empty_checklist_from_the_llm_falls_back_to_the_default(make_agent_context):
    agent_context = make_agent_context([checklist_answer(["  ", ""]), QuestionPlan(questions=[])])

    eda = run_data_analyst(ledger_with_brief(), agent_context)

    assert list(eda.checklist) == DEFAULT_CHECKLIST


def test_checklist_items_are_cleaned_deduplicated_and_limited(make_agent_context):
    messy_items = [" Target distribution ", "target   distribution"] + [
        f"item {n}" for n in range(30)
    ]
    agent_context = make_agent_context([checklist_answer(messy_items), QuestionPlan(questions=[])])

    eda = run_data_analyst(ledger_with_brief(), agent_context)

    assert len(eda.checklist) == 12
    assert list(eda.checklist)[0] == "Target distribution"
    assert list(eda.checklist)[1] == "item 0"


def test_no_questions_from_the_llm_ends_the_loop_without_running_code(make_agent_context):
    agent_context = make_agent_context([checklist_answer(), QuestionPlan(questions=[])])

    eda = run_data_analyst(ledger_with_brief(), agent_context)

    assert eda.findings == []
    assert eda.eda_loops_used == 0
    assert len(agent_context.llm_client.received_prompts) == 2


def test_at_most_four_questions_are_sent_to_the_coder(make_agent_context):
    six_questions = think_answer(*[[] for _ in range(6)])
    agent_context = make_agent_context(
        [
            checklist_answer(),
            six_questions,
            coder_answer(),
            reading_answer(),
        ]
    )

    run_data_analyst(ledger_with_brief(), agent_context)

    coder_prompt = agent_context.llm_client.received_prompts[2]
    assert "4. question 4" in coder_prompt
    assert "question 5" not in coder_prompt


def test_reentry_runs_one_round_with_the_reviewers_question(make_agent_context):
    ledger = ledger_with_brief()
    ledger.eda.checklist = {"target distribution": "covered", "missing values": "open"}
    ledger.eda.eda_loops_used = 2
    apply_reading(ledger, reading_answer(["Old fact one.", "Old fact two."]))
    agent_context = make_agent_context(
        [
            coder_answer(),
            reading_answer(["New fact from the question."]),
            summary_answer(feature_ideas=[]),
        ]
    )
    question = "Do the same customers appear in train and test?"

    eda = run_data_analyst(ledger, agent_context, question=question)

    assert question in agent_context.llm_client.received_prompts[0]
    assert [finding.finding_id for finding in eda.findings] == [
        "finding_001",
        "finding_002",
        "finding_003",
    ]
    assert eda.eda_loops_used == 2
    assert len(agent_context.llm_client.received_prompts) == 3


def test_the_input_ledger_is_never_changed(make_agent_context):
    ledger = ledger_with_brief()
    agent_context = make_agent_context(
        [
            checklist_answer(),
            think_answer(CHECKLIST_ITEMS),
            coder_answer(),
            reading_answer(["Fact."], covered=CHECKLIST_ITEMS),
            summary_answer(),
        ]
    )

    run_data_analyst(ledger, agent_context)

    assert ledger.eda.findings == []
    assert ledger.eda.checklist == {}


def test_feature_ideas_with_an_unknown_finding_id_lose_the_id(make_agent_context):
    ideas = [
        FeatureIdea(description="valid link", finding_id="finding_001"),
        FeatureIdea(description="invented link", finding_id="finding_099"),
    ]
    agent_context = make_agent_context(
        [
            checklist_answer(),
            think_answer(["missing values"]),
            coder_answer(),
            reading_answer(["Fact."], covered=CHECKLIST_ITEMS),
            summary_answer(feature_ideas=ideas),
        ]
    )

    eda = run_data_analyst(ledger_with_brief(), agent_context)

    assert [(idea.description, idea.finding_id) for idea in eda.feature_ideas] == [
        ("valid link", "finding_001"),
        ("invented link", ""),
    ]


def test_coverage_marks_match_item_text_loosely_and_ignore_unknown_items():
    ledger = ledger_with_brief()
    ledger.eda.checklist = {"Missing values": "open", "Target distribution": "open"}
    reading = RoundReading(
        findings=[],
        coverage=[
            CoverageMark(item="  missing   VALUES ", status="covered"),
            CoverageMark(item="something invented", status="covered"),
            CoverageMark(item="Target distribution", status="not_applicable"),
        ],
    )

    apply_reading(ledger, reading)

    assert ledger.eda.checklist == {
        "Missing values": "covered",
        "Target distribution": "not_applicable",
    }
    assert "something invented" not in ledger.eda.checklist


def test_duplicate_findings_are_ignored_even_within_one_reading():
    ledger = ledger_with_brief()
    reading = reading_answer(["Same fact.", "same   fact.", "Another fact."])

    new_count = apply_reading(ledger, reading)

    assert new_count == 2
    assert [finding.finding_id for finding in ledger.eda.findings] == ["finding_001", "finding_002"]


def test_events_show_start_round_progress_and_finish(make_agent_context):
    agent_context = make_agent_context(
        [
            checklist_answer(),
            think_answer(CHECKLIST_ITEMS),
            coder_answer(),
            reading_answer(["Fact."], covered=CHECKLIST_ITEMS),
            summary_answer(),
        ]
    )
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    run_data_analyst(ledger_with_brief(), agent_context)

    analyst_events = [e for e in received_events if e.agent_name == "Data Analyst"]
    assert analyst_events[0].event_type == EventType.AGENT_STARTED
    assert any(
        "round 1 of 4: 1 new findings, 0 checklist items still open" in e.message
        for e in analyst_events
    )
    assert analyst_events[-1].event_type == EventType.AGENT_FINISHED
