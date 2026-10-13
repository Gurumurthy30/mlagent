"""The Reviewer: the senior who reads the results and says what to do next.

At the end of every round the Coder runs one diagnosis script on the best run's saved
validation predictions. The LLM reads it and picks the next focus. The code guards in
stop_rules.py then replace any choice the rules do not allow.
"""

from pydantic import BaseModel

from mlagent.agents.base import (
    AgentContext,
    build_common_prompt_values,
    display_name,
    facts_to_text,
    render_prompt,
)
from mlagent.agents.coder import run_coder
from mlagent.digest import describe_brief, describe_run
from mlagent.events import EventType
from mlagent.settings import (
    MAX_DATA_REENTRIES,
    MAX_REVIEW_INSTRUCTIONS,
    MIN_EXPERIMENT_LOOPS,
    MIN_ROUNDS_BETWEEN_TUNING,
    STUCK_FOCUS_LIMIT,
)
from mlagent.state import ExperimentRun, Ledger, Review, ReviewFocus
from mlagent.stop_rules import apply_reviewer_guards, pick_best_run, round_had_counted_gain

AGENT_KEY = "reviewer"
AGENT_NAME = display_name(AGENT_KEY)
RESULT_TAG = "DIAG"

DIAG_RESULT_KEYS_TEXT = """\
- "weak_spots": list of short texts: where the model fails, each with numbers. Look at errors \
per class or per segment, large residuals, ranges of the target or inputs with bad scores, \
and groups, sources or time periods with bad scores. Also state the spread of the score \
between validation splits, the gap between validation and holdout, and whether the model \
looks under-fitted (poor everywhere) or over-fitted (good on training, worse on validation)
- "comparison": text about where this run gained or lost against the previous best run, with \
numbers, or "" when there is no earlier run
- "notes": list of other short observations, for example calibration problems, a prediction \
shift between validation and test, or which inputs the model relies on most"""

REVIEW_PROMPT = """\
You are the Reviewer of an ML team, the senior who reads the results and says what to do next. \
You decide where the team's limited rounds are spent. Be specific: the next agent can only act \
on what you tell it.

{run_rules}

WHAT THE TEAM KNOWS:
{digest}

DIAGNOSIS OF THE BEST RUN (from its saved validation predictions):
{diagnosis_text}

BUDGET: {rounds_left} experiment rounds are left.

Think it through before you choose:
1. What changed in the last round, and was the gain counted? A gain that is not marked as \
counted is noise. Do not build on it.
2. What is the bottleneck? Read the diagnosis:
   - weak on specific classes, segments or ranges: new input preparation aimed at them \
(features), or a different kind of model
   - features keep adding gains: more features. Several features rounds without a counted \
gain: switch to a model round
   - the model is under-fitted: a stronger or more flexible model
   - the model is over-fitted or the splits disagree strongly: a simpler or more regularized \
model, or data work
   - validation and holdout disagree, or something looks leaked: data (ask about it)
   - a strong model family and stable features, but its settings were never searched: tune
   - results point to something the EDA never checked: data
3. What was already tried? Read the runs, the features tried and the previous reviews, \
including notes where code replaced a choice. Do not send the team back to something that \
failed.
4. How many rounds are left? With few rounds left, choose the option with the best expected \
gain per round: cheap, safe steps. Finish when more work is unlikely to beat the noise.

Choose the focus of the next round:
- features: new input preparation steps (for tables, new columns)
- model: a different model family or setup
- tune: a bounded search over the settings of the best models
- data: more analysis of the data, when the results point to something the EDA never checked. \
Give the question.
- finish: only when more work is unlikely to help.
The code enforces these rules and replaces an illegal choice: finish only after \
{min_loops} rounds that include a features round and a model round, and with an approved run. \
Tune needs an approved run and {tune_gap} rounds since the last tune. Data at most \
{max_reentries} times. The same focus {stuck_limit} times in a row without a counted gain is \
replaced. Change one thing at a time and read the counted gain marks in the reviews. Pick the \
best legal choice yourself, so the code does not have to.

Give:
- assessment: what improved, what did not, where the bottleneck is. Use numbers from the \
runs and the diagnosis.
- focus: one of the five words above.
- reason: one sentence that names the evidence.
- instructions: up to {max_instructions} short points for the agent that runs next, written \
for that agent (the Feature Engineer for features, the Modeler for model, the Tuner for tune). \
Each point is concrete: a kind of feature or a weak spot to attack, a model family or setting \
to try, a range to search, something to avoid because it failed. No general advice such as \
"improve the model". Leave it empty for finish.
- question: only for data. One specific question about the data that one script can answer. \
Otherwise an empty string.
Return JSON: assessment, focus, reason, instructions, question."""


class ReviewDraft(BaseModel):
    assessment: str
    focus: ReviewFocus
    reason: str
    instructions: list[str] = []
    question: str = ""


def find_previous_best_run(ledger: Ledger, best_run: ExperimentRun) -> ExperimentRun | None:
    earlier_runs = [run for run in ledger.approved_runs() if run.iteration < best_run.iteration]
    return pick_best_run(earlier_runs, ledger.plan.higher_is_better)


def build_diagnosis_task(
    ledger: Ledger,
    best_run: ExperimentRun,
    previous_best_run: ExperimentRun | None,
    agent_context: AgentContext,
) -> str:
    workspace = agent_context.workspace
    previous_text = "none"
    if previous_best_run is not None:
        previous_text = f"{previous_best_run.experiment_id}, predictions at "
        previous_text += previous_best_run.validation_predictions_path or "none"
    direction = "higher" if ledger.plan.higher_is_better else "lower"
    return (
        "Diagnose where the best model fails, using its saved validation predictions and the "
        "true labels. Print a short result.\n"
        f"Best run: {describe_run(ledger, best_run)}\n"
        f"Saved validation predictions: {best_run.validation_predictions_path}\n"
        f"Previous best run: {previous_text}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Primary metric: {ledger.plan.primary_metric} ({direction} is better)\n"
        f"Project folder (read only): {workspace.project_folder}\n\n"
        "Look at whatever fits this data: error by class, by value range or by group; the "
        "worst predicted examples and what they have in common (length, source, missing "
        "fields, speaker); confidence and calibration when the predictions are probabilities. "
        "When a previous best run exists, say where the best run gained or lost.\n"
        f"Write only inside {workspace.artifacts_folder / AGENT_KEY}.\n\n"
        f"{describe_brief(ledger)}"
    )


def run_diagnosis(
    ledger: Ledger, best_run: ExperimentRun, reviewed_round: int, agent_context: AgentContext
) -> str:
    """Return the diagnosis as text for the prompt. A failed script gives a plain note instead."""
    coder_result = run_coder(
        task_text=build_diagnosis_task(
            ledger, best_run, find_previous_best_run(ledger, best_run), agent_context
        ),
        result_tag=RESULT_TAG,
        result_keys_text=DIAG_RESULT_KEYS_TEXT,
        working_folder=agent_context.workspace.artifacts_folder / AGENT_KEY,
        script_name=f"diagnose_round_{reviewed_round}.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )
    if not coder_result.succeeded:
        return "The diagnosis script failed, so there is no diagnosis this round."
    return facts_to_text(coder_result.result_data)


def run_reviewer(ledger: Ledger, agent_context: AgentContext) -> Review:
    """Review the round that just finished. The returned Review has the next focus."""
    event_bus = agent_context.event_bus
    reviewed_round = ledger.iteration
    working_ledger = ledger.model_copy(deep=True)
    working_ledger.iteration = reviewed_round + 1
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, f"reviewing round {reviewed_round}")

    best_run = working_ledger.best_approved_run()
    if best_run is not None and best_run.validation_predictions_path:
        diagnosis_text = run_diagnosis(working_ledger, best_run, reviewed_round, agent_context)
    else:
        diagnosis_text = "There is no approved run with validation predictions to diagnose yet."

    common_values = build_common_prompt_values(AGENT_KEY, working_ledger, agent_context)
    max_loops = agent_context.config.limits.max_experiment_loops
    prompt = render_prompt(
        REVIEW_PROMPT,
        **common_values,
        diagnosis_text=diagnosis_text,
        rounds_left=max(max_loops - working_ledger.iteration, 0),
        min_loops=MIN_EXPERIMENT_LOOPS,
        tune_gap=MIN_ROUNDS_BETWEEN_TUNING,
        max_reentries=MAX_DATA_REENTRIES,
        stuck_limit=STUCK_FOCUS_LIMIT,
        max_instructions=MAX_REVIEW_INSTRUCTIONS,
    )
    draft = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, ReviewDraft)

    review = Review(
        iteration=working_ledger.iteration,
        best_experiment_id=best_run.experiment_id if best_run else "",
        assessment=draft.assessment,
        focus=draft.focus,
        reason=draft.reason,
        instructions=draft.instructions[:MAX_REVIEW_INSTRUCTIONS],
        question=draft.question if draft.focus == "data" else "",
        reviewed_focus=ledger.focus,
        counted_gain=round_had_counted_gain(ledger, reviewed_round),
    )
    review = apply_reviewer_guards(ledger, review)

    message = f"next focus: {review.focus}. {review.reason}"
    event_bus.emit(EventType.DECISION, AGENT_NAME, message)
    if review.guard_note:
        event_bus.emit(EventType.WARNING, AGENT_NAME, review.guard_note)
    event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, f"round {reviewed_round} reviewed")
    return review
