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
from mlagent.digest import describe_brief, describe_dataset_columns, describe_run
from mlagent.events import EventType
from mlagent.script_contract import SPLITS_FORMAT_TEXT, VALIDATION_PREDICTIONS_FORMAT_TEXT
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
- "weak_spots": list of at most 5 short texts, the worst first. Each names one segment (a \
class, a range of the target or of an input, a group, a source, a time period), how many \
validation examples it holds, the primary metric or error rate in it, and the same number \
over all examples. Report only segments with at least 20 examples where the result is \
clearly worse than average. Take the true labels from the training data by example_id
- "comparison": text about where this run gained or lost against the previous best run, with \
numbers: the score change in the main segments, and how many examples the previous run had \
right that this run has wrong, and the reverse (join the two validation prediction files on \
example_id). Write "" when there is no earlier run
- "notes": list of other short texts, each starting with a label:
  "SPREAD:" the score on each validation split and the difference between the best and the \
worst
  "HOLDOUT GAP:" the holdout score against the mean validation score (the rows with fold -1)
  "FIT:" under-fitted, over-fitted or unknown. Say under-fitted or over-fitted only when the \
training score is known from the run's own results. Otherwise write unknown, do not guess
  "SHIFT:" how the test predictions differ from the validation predictions (class shares or \
mean prediction)
  "OTHER:" calibration problems, or the inputs the model relies on most
Write every number with at most 3 digits after the point, and keep each text under 200 \
characters."""

REVIEW_PROMPT = """\
You are the Reviewer of an ML team, the senior who reads the results and decides what the \
team does next. The team has few rounds, so each one must count. The next agent sees only \
your instructions or your question, not your assessment. Put the numbers and names it needs \
inside them.

HOW TO DECIDE
Write the assessment as five short lines, in this order, and let them lead to your choice:
1. LAST ROUND: what changed, the score before and after, and whether the gain is marked as \
counted. A gain that is not counted, or is smaller than the spread between validation \
splits, is noise. Do not build on it.
2. BOTTLENECK: the weakest spot in the diagnosis, with its numbers.
3. TRIED: what was tried and did not help (runs, features, earlier reviews, and notes where \
code replaced a choice). Do not send the team back to it.
4. ROUNDS: how many are left and what that allows (see below).
5. LIMITS: which focuses are illegal now (see the limits below).

Match the bottleneck to a focus, taking the first rule that fits:
- Validation and holdout disagree, a score looks too good, or the splits disagree strongly \
with no clear cause: data. Ask what could explain it.
- The results point at something the EDA never looked at: data.
- Weak on named classes, segments, ranges or periods: features aimed at them. When features \
rounds in a row have given no counted gain, a different model instead.
- Poor on every segment, and the diagnosis says under-fitted: a stronger or more flexible \
model.
- Good on some splits and bad on others, or the diagnosis says over-fitted: a simpler or more \
regularized model.
- The model family is strong, the features are stable, and its settings were never \
searched: tune.
- None of these, and more work is unlikely to beat the noise: finish.
When two rules fit, take the step that can be done in one round and checked with one number.

WHAT THE ROUNDS ALLOW
- More than 3 rounds left: one bold step is fine, such as a new family of features or \
another model family.
- 2 or 3 left: the safest gain, such as a targeted features step or a tune of the best model.
- The last round: only a cheap step that cannot damage the best run, or finish when it is \
legal. Combining the best runs is the Submitter's job, so do not spend a round on it.

THE FIVE FOCUSES
- features: new input preparation steps (for tables, new columns)
- model: a different model family or setup
- tune: a bounded search over the settings of the best model
- data: more analysis of the data, when the results point at something the EDA never checked
- finish: only when more work is unlikely to help

LIMITS ENFORCED BY CODE
The code replaces an illegal choice, so pick the best legal one yourself. Finish is legal \
only after {min_loops} rounds that include a features round and a model round, and with an \
approved run. Tune needs an approved run and {tune_gap} rounds since the last tune. Data is \
legal at most {max_reentries} times. The same focus {stuck_limit} times in a row without a \
counted gain is replaced. Change one thing at a time, so that the counted gain marks tell \
which change helped.

WHAT TO WRITE
- assessment: the five lines above.
- reason: one sentence with the evidence that decides the focus.
- focus: one of the five words.
- instructions: up to {max_instructions} short points for the agent that runs next (the \
Feature Engineer for features, the Modeler for model, the Tuner for tune). Every point names \
the weak spot or the evidence with its numbers, and the concrete action. Leave it empty for \
finish and for data.
  Good for features: "Errors are 3 times higher where income is missing (180 examples): add \
a missing flag for income and the ratio of income to loan amount."
  Good for model: "The linear model is poor on all splits (0.61): try a gradient boosted \
tree with depth 6 and 600 trees."
  Good for tune: "Search learning rate 0.01 to 0.1 on a log scale, leaves 15 to 63, \
minimum samples per leaf 10 to 100. Keep the features."
  Bad: "improve the features", "try other models", "tune more".
  Also say what to avoid, with the reason, when something failed before.
- question: only for data. One specific question that one script can answer, naming the \
columns to look at, for example: "Do the 200 validation examples with the largest errors \
share a source or a month? Compare the columns source and month for them against the rest." \
Otherwise an empty string.

{run_rules}

WHAT THE TEAM KNOWS:
{digest}

DIAGNOSIS OF THE BEST RUN (from its saved validation predictions):
{diagnosis_text}

BUDGET: {rounds_left} experiment rounds are left.

Write your review now. Return JSON: assessment, reason, focus, instructions, question."""


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
    dataset_cols_text = describe_dataset_columns(workspace.project_folder, ledger.brief)
    return (
        "Diagnose where the best model fails, using its saved validation predictions and the "
        "true labels. Print a short result.\n"
        f"Best run: {describe_run(ledger, best_run)}\n"
        f"Saved validation predictions: {best_run.validation_predictions_path}\n"
        f"Previous best run: {previous_text}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Primary metric: {ledger.plan.primary_metric} ({direction} is better)\n"
        f"Project folder (read only): {workspace.project_folder}\n\n"
        "ALIGNMENT AND DATA RULES:\n"
        "- Use `from mlagent_tools import load_splits, load_val_preds_with_labels` to reliably "
        "load splits and align validation predictions with true labels by example_id.\n"
        "- You MUST check that rows matched: if 0 rows matched (or merge produced empty DataFrame), "
        "raise an exception immediately! Never report zero errors or a healthy model when rows "
        "did not match.\n"
        "- Only use real columns that exist in the dataset (see column list below). Never invent "
        "or guess column names.\n\n"
        "Look at whatever fits this data: error by class, by value range or by group; the "
        "worst predicted examples and what they have in common (length, source, missing "
        "fields, speaker); confidence and calibration when the predictions are probabilities. "
        "When a previous best run exists, say where the best run gained or lost.\n"
        f"Write only inside {workspace.artifacts_folder / AGENT_KEY}.\n\n"
        f"{describe_brief(ledger)}\n\n"
        f"{dataset_cols_text}\n\n"
        f"{VALIDATION_PREDICTIONS_FORMAT_TEXT}\n\n"
        f"{SPLITS_FORMAT_TEXT}"
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
