"""The ML Lead: decides how we measure, what the baseline is, and what the plan is.

The LLM writes the plan. The Coder then writes the splits file for the chosen validation scheme,
and code checks that file. Every later agent follows this plan.
"""

from pydantic import BaseModel

from mlagent.agents.base import (
    AgentContext,
    AgentFailedError,
    build_common_prompt_values,
    display_name,
    render_prompt,
)
from mlagent.agents.coder import last_error_line, run_coder
from mlagent.digest import describe_brief, trim_text
from mlagent.events import EventType
from mlagent.script_contract import SPLITS_FORMAT_TEXT, check_splits_file
from mlagent.settings import (
    BIG_FILE_MB,
    DEFAULT_SPLIT_COUNT,
    HOLDOUT_FRACTION,
    MAX_PLAYBOOK_CHARS,
    MIN_EXAMPLES_FOR_HOLDOUT,
    RANDOM_SEED,
)
from mlagent.state import ExperimentPlan, Ledger

AGENT_KEY = "ml_lead"
AGENT_NAME = display_name(AGENT_KEY)
RESULT_TAG = "SPLITS"
MAX_RISKS = 5

SPLITS_RESULT_KEYS_TEXT = """\
- "splits_path": the path you saved the file to
- "n_splits": integer
- "n_examples": number of entries in example_ids (0 when you wrote seeds)
- "n_holdout": number of examples with fold -1"""

PLAN_PROMPT = """\
You are the ML Lead of an ML team. You are the senior who decides how the team measures \
progress, what the baseline is, and what the plan is. Every agent after you follows this plan, \
and a bad validation setup makes every later score misleading, so take your time on it.

{run_rules}

WHAT THE TEAM KNOWS (task, files, EDA findings, hints):
{digest}

Think before you write the plan:
1. What exactly is predicted, and how will the final result be scored? Read the metric from \
the brief. Does it need class labels, probabilities, ranks or numbers? Is it sensitive to \
imbalance, outliers or scale?
2. How will the real test data differ from the training data: later in time, other groups, \
other sources, new categories? The validation must imitate that difference. If validation is \
easier than the real test, every score will lie.
3. What do the EDA validation_hints and leakage_risks say about structure: time order, groups, \
duplicates, imbalance, speakers, patients, near-duplicates?
4. How much data is there and how slow is training? Use the machine description and the time \
limit in the run rules. A plan the machine cannot finish is a bad plan.

Write the plan:
- validation_scheme: exactly one of kfold, stratified_kfold, group_kfold, time_split, \
single_split, repeated_seeds.
  - stratified_kfold for classification with imbalance
  - group_kfold when rows of one group (user, patient, speaker, source) must not be split \
across folds
  - time_split when the test lies in the future
  - single_split when training is slow, for example deep learning on images, audio or large \
text
  - repeated_seeds only when there are no labels
  - kfold otherwise
  When two structures apply, choose the one that causes the worse leakage if ignored. \
validation_reason: one sentence that names the evidence from the EDA or the task.
- primary_metric and higher_is_better: take them from the brief unless the user's words say \
otherwise. Never invent a metric the task does not use. extra_metrics: other metrics worth \
reporting (for example per-class or calibration), at most 3.
- baseline_description: the simplest sensible first model plus the basic input preparation \
for this data. It must run fast and be hard to get wrong. It is the number every later idea \
has to beat.
- target_score: a number only if the user stated a goal. Otherwise null. Never guess one.
- use_gpu: true only if the task needs a GPU (deep learning on images, audio, text) and the \
machine above can use one. The configuration and the machine have the last word.
- risks: up to {max_risks} items, most dangerous first, each specific to this project and \
each with the action that prevents it. Not generic advice.
- playbook: 6 to 10 short lines, read by the Feature Engineer, Modeler, Tuner, Evaluator and \
Reviewer. Make it concrete for this task. Cover:
  - which model families usually work on this kind of data, and which to start with
  - input preparation that usually helps, and what to avoid
  - quirks of the metric (for example probabilities instead of labels, clipping, thresholds)
  - how to handle imbalance, groups or time in training
  - how to stay inside the time limit (subsample, fewer epochs, smaller models)
  - how the team should spend its experiments: cheap, broad tests first, expensive ones later
  Use only facts from the task, the brief and the EDA. If you are unsure about something, \
say so in the line instead of stating it as a fact.
Return JSON with exactly these fields."""


class PlanDraft(BaseModel):
    validation_scheme: str
    validation_reason: str
    primary_metric: str
    higher_is_better: bool
    extra_metrics: list[str] = []
    baseline_description: str
    target_score: float | None = None
    use_gpu: bool = False
    risks: list[str] = []
    playbook: str


def decide_use_gpu(plan_wants_gpu: bool, use_gpu_setting: str, can_train_on_gpu: bool) -> bool:
    """The config and the machine have the last word over what the LLM asked for."""
    if use_gpu_setting == "never" or not can_train_on_gpu:
        return False
    if use_gpu_setting == "always":
        return True
    return plan_wants_gpu


def build_plan(draft: PlanDraft, ledger: Ledger, agent_context: AgentContext) -> ExperimentPlan:
    brief = ledger.brief
    has_own_metric = bool(draft.primary_metric.strip())
    return ExperimentPlan(
        validation_scheme=draft.validation_scheme.strip(),
        validation_reason=draft.validation_reason.strip(),
        primary_metric=draft.primary_metric.strip() if has_own_metric else brief.primary_metric,
        higher_is_better=draft.higher_is_better if has_own_metric else brief.higher_is_better,
        extra_metrics=draft.extra_metrics,
        baseline_description=draft.baseline_description.strip(),
        target_score=draft.target_score,
        use_gpu=decide_use_gpu(
            draft.use_gpu,
            agent_context.config.compute.use_gpu,
            agent_context.environment.can_train_on_gpu,
        ),
        risks=draft.risks[:MAX_RISKS],
        playbook=trim_text(draft.playbook.strip(), MAX_PLAYBOOK_CHARS),
    )


def build_splits_task(
    plan: ExperimentPlan,
    ledger: Ledger,
    agent_context: AgentContext,
    previous_error: str = "",
) -> str:
    hint_text = "; ".join(ledger.eda.validation_hints) or "none"
    rules = [
        f"- Project folder (read only): {agent_context.workspace.project_folder}",
        f"- Save the file to exactly this path: {agent_context.workspace.splits_path}",
        f"- Never fully load a file larger than {BIG_FILE_MB} MB. Read only what you need "
        "to list the examples. On big data use hashing or exact matching for duplicates and "
        "say in the result what you skipped.",
        "- List every training example exactly once in example_ids, in the order the training "
        "data lists them (row order for tables, sorted file names for folders). Use the same "
        "ids the training data uses.",
        "- Print the number of examples you found before splitting. It must match the number of "
        "rows (or files) of the training data. If it is 1 or far too small, you read the data "
        "wrongly: fix that first.",
        "- Only when the EDA structure above names duplicates or groups: keep rows with identical "
        "content (ignoring the id column) together, in the holdout or in one fold. Otherwise every "
        "example is independent: do not build groups and do not hash rows.",
        f"- Use {DEFAULT_SPLIT_COUNT} validation splits for the kfold schemes, fewer when the "
        "data is tiny, but keep at least 30 examples in every validation split. For "
        "single_split use one validation split of about 20 percent.",
        f"- When there are at least {MIN_EXAMPLES_FOR_HOLDOUT} examples, hold out about "
        f"{int(HOLDOUT_FRACTION * 100)} percent of them (fold -1) first. With fewer examples "
        "hold out nothing.",
        "- The holdout follows the scheme: random for kfold, stratified on the target for "
        "stratified_kfold, whole groups for group_kfold, the latest examples in time order "
        "for time_split. Make the validation splits from the remaining examples with the same "
        "rule.",
        "- After saving, check the file: every example appears once, the holdout and folds do "
        "not overlap, no duplicate group named in the EDA crosses a boundary, and for time_split "
        "every validation example is later than its training examples. Fix it if a check fails.",
    ]
    if previous_error:
        rules.append(
            f"- A first attempt failed with: {previous_error}. Write the simplest correct "
            "version: keep the scheme but drop every optional rule (duplicate groups and "
            "extra checks), and make sure the file follows the format below."
        )
    rules_text = "\n".join(rules)
    return (
        "Create the validation splits for this project and save them as one JSON file.\n"
        f"Scheme: {plan.validation_scheme}. Reason: {plan.validation_reason}\n"
        f"Seed: {RANDOM_SEED}. Structure found in the EDA: {hint_text}\n\n"
        f"{describe_brief(ledger)}\n\n"
        f"Rules:\n{rules_text}\n\n"
        f"{SPLITS_FORMAT_TEXT}"
    )


MAX_SPLITS_TRIES = 2


def create_splits_file(
    plan: ExperimentPlan, planned_ledger: Ledger, agent_context: AgentContext
) -> None:
    previous_error = ""
    for try_number in range(1, MAX_SPLITS_TRIES + 1):
        task_text = build_splits_task(plan, planned_ledger, agent_context, previous_error)
        coder_result = run_coder(
            task_text=task_text,
            result_tag=RESULT_TAG,
            result_keys_text=SPLITS_RESULT_KEYS_TEXT,
            working_folder=agent_context.workspace.artifacts_folder / AGENT_KEY,
            script_name="make_splits.py",
            caller_key=AGENT_KEY,
            ledger=planned_ledger,
            agent_context=agent_context,
        )
        if coder_result.succeeded:
            problems = check_splits_file(agent_context.workspace.splits_path)
            if problems:
                problems = ["splits file is not usable: " + " ".join(problems)]
        else:
            problems = [last_error_line(coder_result.error_text)]
        if not problems:
            return
        previous_error = " ".join(problems)
        agent_context.event_bus.emit(
            EventType.WARNING,
            AGENT_NAME,
            f"splits try {try_number} failed: {previous_error}",
        )
    log_path = agent_context.workspace.commands_log_path
    raise AgentFailedError(
        f"The ML Lead could not create the validation splits: {previous_error}. "
        f"Full log: {log_path}"
    )


def run_ml_lead(ledger: Ledger, agent_context: AgentContext) -> ExperimentPlan:
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "writing the plan")

    common_values = build_common_prompt_values(AGENT_KEY, ledger, agent_context)
    prompt = render_prompt(PLAN_PROMPT, **common_values, max_risks=MAX_RISKS)
    draft = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, PlanDraft)
    plan = build_plan(draft, ledger, agent_context)
    direction = "higher" if plan.higher_is_better else "lower"
    event_bus.emit(
        EventType.DECISION,
        AGENT_NAME,
        f"validation {plan.validation_scheme}, metric {plan.primary_metric} ({direction} is "
        f"better), GPU {'yes' if plan.use_gpu else 'no'}",
    )

    planned_ledger = ledger.model_copy(deep=True)
    planned_ledger.plan = plan
    create_splits_file(plan, planned_ledger, agent_context)

    plan.splits_path = str(agent_context.workspace.splits_path)
    event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, "plan and splits are ready")
    return plan
