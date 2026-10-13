from scripted_answers import coder_answer_printing, two_failed_coder_attempts

from mlagent.agents.reviewer import ReviewDraft, run_reviewer
from mlagent.events import EventType
from mlagent.memory import create_new_ledger
from mlagent.state import Evaluation, ExperimentRun

DIAGNOSIS = {
    "weak_spots": ["class 'churn' has recall 0.41"],
    "comparison": "",
    "notes": ["probabilities are overconfident"],
}


def diagnosis_answer() -> str:
    return coder_answer_printing("DIAG", DIAGNOSIS)


def make_ledger(workspace, focus="start"):
    ledger = create_new_ledger("Predict churn.", "macro F1")
    ledger.brief.summary = "Customer table."
    ledger.plan.primary_metric = "macro_f1"
    ledger.plan.splits_path = str(workspace.splits_path)
    ledger.focus = focus
    return ledger


def add_run(ledger, workspace, scores, iteration=0, approved=True, with_predictions=True):
    experiment_id = ledger.next_experiment_id()
    predictions_path = workspace.artifacts_folder / experiment_id / "validation_predictions.csv"
    ledger.runs.append(
        ExperimentRun(
            experiment_id=experiment_id,
            focus=ledger.focus,
            model_family="lightgbm",
            iteration=iteration,
            validation_scores=scores,
            score_mean=sum(scores) / len(scores),
            validation_predictions_path=str(predictions_path) if with_predictions else "",
        )
    )
    if approved:
        ledger.evaluations.append(Evaluation(experiment_id=experiment_id, verdict="approve"))
    return experiment_id


def draft(**changes) -> ReviewDraft:
    values = {
        "assessment": "baseline is fine, recall is weak",
        "focus": "model",
        "reason": "try a stronger learner",
        "instructions": ["try gradient boosting"],
        "question": "",
    }
    values.update(changes)
    return ReviewDraft(**values)


def test_a_review_has_the_next_focus_and_the_facts_of_the_round(make_agent_context, workspace):
    ledger = make_ledger(workspace)
    add_run(ledger, workspace, [0.80, 0.81, 0.79])
    agent_context = make_agent_context([diagnosis_answer(), draft()])

    review = run_reviewer(ledger, agent_context)

    assert review.iteration == 1
    assert review.best_experiment_id == "experiment_001"
    assert review.focus == "model"
    assert review.reviewed_focus == "start"
    assert review.counted_gain is True
    assert review.assessment == "baseline is fine, recall is weak"
    assert review.instructions == ["try gradient boosting"]
    assert review.guard_note == ""


def test_the_diagnosis_task_and_the_decision_prompt(make_agent_context, workspace):
    ledger = make_ledger(workspace)
    add_run(ledger, workspace, [0.80, 0.81, 0.79])
    agent_context = make_agent_context([diagnosis_answer(), draft()])

    run_reviewer(ledger, agent_context)

    diagnosis_prompt, review_prompt = agent_context.llm_client.received_prompts
    assert "Diagnose where the best model fails" in diagnosis_prompt
    assert "validation_predictions.csv" in diagnosis_prompt
    assert "Previous best run: none" in diagnosis_prompt
    assert f"Splits file: {workspace.splits_path}" in diagnosis_prompt
    assert "<DIAG>JSON</DIAG>" in diagnosis_prompt
    assert "class 'churn' has recall 0.41" in review_prompt
    assert "BUDGET: 14 experiment rounds are left." in review_prompt
    assert "Experiment rounds finished: 1 of 15" in review_prompt
    assert "finish only after 3 rounds" in review_prompt
    assert "Tune needs an approved run and 2 rounds" in review_prompt
    assert "Data at most 2 times" in review_prompt
    assert "up to 5 short points" in review_prompt


def test_the_previous_best_run_is_named_in_the_diagnosis_task(make_agent_context, workspace):
    ledger = make_ledger(workspace, focus="model")
    add_run(ledger, workspace, [0.80, 0.80, 0.80], iteration=0)
    add_run(ledger, workspace, [0.84, 0.85, 0.84], iteration=1)
    ledger.iteration = 1
    agent_context = make_agent_context([diagnosis_answer(), draft()])

    review = run_reviewer(ledger, agent_context)

    diagnosis_prompt = agent_context.llm_client.received_prompts[0]
    assert "Previous best run: experiment_001, predictions at" in diagnosis_prompt
    assert review.best_experiment_id == "experiment_002"
    assert review.iteration == 2
    assert review.counted_gain is True


def test_a_round_without_a_better_run_has_no_counted_gain(make_agent_context, workspace):
    ledger = make_ledger(workspace, focus="model")
    add_run(ledger, workspace, [0.80, 0.80, 0.80], iteration=0)
    add_run(ledger, workspace, [0.79, 0.80, 0.78], iteration=1)
    ledger.iteration = 1
    agent_context = make_agent_context([diagnosis_answer(), draft()])

    assert run_reviewer(ledger, agent_context).counted_gain is False


def test_without_an_approved_run_there_is_no_diagnosis_and_no_code_is_run(
    make_agent_context, workspace
):
    ledger = make_ledger(workspace)
    ledger.runs.append(ExperimentRun(experiment_id="experiment_001", error="out of memory"))
    agent_context = make_agent_context([draft(focus="features")])

    review = run_reviewer(ledger, agent_context)

    assert len(agent_context.llm_client.received_prompts) == 1
    assert (
        "no approved run with validation predictions"
        in agent_context.llm_client.received_prompts[0]
    )
    assert "out of memory" in agent_context.llm_client.received_prompts[0]
    assert review.best_experiment_id == ""
    assert review.counted_gain is False


def test_a_best_run_without_saved_predictions_is_not_diagnosed(make_agent_context, workspace):
    ledger = make_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8], with_predictions=False)
    agent_context = make_agent_context([draft()])

    run_reviewer(ledger, agent_context)

    assert len(agent_context.llm_client.received_prompts) == 1


def test_a_failed_diagnosis_does_not_stop_the_review(make_agent_context, workspace):
    ledger = make_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8])
    agent_context = make_agent_context([*two_failed_coder_attempts("no labels"), draft()])

    review = run_reviewer(ledger, agent_context)

    assert review.focus == "model"
    assert "diagnosis script failed" in agent_context.llm_client.received_prompts[2]


def test_an_illegal_choice_is_replaced_and_the_note_is_shown(make_agent_context, workspace):
    ledger = make_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8])
    agent_context = make_agent_context([diagnosis_answer(), draft(focus="finish")])
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    review = run_reviewer(ledger, agent_context)

    assert review.focus == "features"
    assert "finish was replaced by features: too early" in review.guard_note
    warnings = [e for e in received_events if e.event_type == EventType.WARNING]
    assert any("finish was replaced by features" in e.message for e in warnings)


def test_the_question_is_kept_only_for_a_data_choice(make_agent_context, workspace):
    ledger = make_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8])
    data_draft = draft(focus="data", question="Do train and test share customers?")
    model_draft = draft(focus="model", question="ignored question")

    with_data = run_reviewer(ledger, make_agent_context([diagnosis_answer(), data_draft]))
    with_model = run_reviewer(ledger, make_agent_context([diagnosis_answer(), model_draft]))

    assert with_data.focus == "data"
    assert with_data.question == "Do train and test share customers?"
    assert with_model.question == ""


def test_instructions_are_cut_to_five(make_agent_context, workspace):
    ledger = make_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8])
    many_instructions = [f"point {number}" for number in range(8)]
    agent_context = make_agent_context([diagnosis_answer(), draft(instructions=many_instructions)])

    review = run_reviewer(ledger, agent_context)

    assert review.instructions == many_instructions[:5]


def test_the_input_ledger_is_not_changed(make_agent_context, workspace):
    ledger = make_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8])
    agent_context = make_agent_context([diagnosis_answer(), draft()])

    run_reviewer(ledger, agent_context)

    assert ledger.iteration == 0
    assert ledger.reviews == []


def test_events_report_the_decision(make_agent_context, workspace):
    ledger = make_ledger(workspace)
    add_run(ledger, workspace, [0.8, 0.8, 0.8])
    agent_context = make_agent_context([diagnosis_answer(), draft()])
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    run_reviewer(ledger, agent_context)

    reviewer_events = [e for e in received_events if e.agent_name == "Reviewer"]
    assert reviewer_events[0].message == "reviewing round 0"
    assert any(e.message == "next focus: model. try a stronger learner" for e in reviewer_events)
