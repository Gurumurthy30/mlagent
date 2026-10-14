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
    MAX_ERROR_TEXT_CHARS,
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
You are the ML Lead of an ML team. You decide how the team measures progress, what the \
baseline is and what the plan is. Every agent after you follows this plan, and a validation \
setup that is easier than the real test makes every later score misleading. You do not write \
code and you do not choose the final model.

{run_rules}

WHAT THE TEAM KNOWS (task, files, EDA findings, hints):
{digest}

Start with analysis, 3 to 5 sentences. This is your thinking, so be honest in it. Answer in \
this order:
1. What is predicted and how is it scored? Does the metric need labels, probabilities, ranks \
or numbers? Is it sensitive to imbalance, outliers or scale?
2. How will the real test data differ from the training data: later in time, other groups, \
other sources, new categories? Name the EDA evidence. If nothing in the EDA says they differ, \
say that.
3. Which structure must the validation respect: time order, groups, duplicates, imbalance, \
speakers, patients? Read validation_hints and leakage_risks.
4. How slow is training? One model run trains once per validation split plus once for the \
final fit, so one training must be several times shorter than the time limit in the run rules. \
Use the machine description too.

Then write the plan fields:
- validation_scheme: exactly one of kfold, stratified_kfold, group_kfold, time_split, \
single_split, repeated_seeds.
  - stratified_kfold for classification with imbalance
  - group_kfold when rows of one group (user, patient, speaker, site, source) must not be \
split across folds
  - time_split when the test lies in the future. It gives one validation block, the latest \
part of the training data.
  - single_split when one training is too slow to repeat for every fold, for example deep \
learning on images, audio or large text
  - repeated_seeds only when there are no labels
  - kfold otherwise
  When two structures apply, choose the one that causes the worse leakage if ignored. \
validation_reason: one sentence that names the EDA finding or the task fact behind it.
- primary_metric and higher_is_better: take them from the brief unless the user's words say \
otherwise. Never invent a metric the task does not use. extra_metrics: other metrics worth \
reporting, at most 3.
- baseline_description: the simplest sensible first model plus the basic input preparation \
for this data. It must run fast and be hard to get wrong. Every later idea has to beat it.
- target_score: a number only if the user stated a goal, otherwise null. Never guess one.
- use_gpu: true only if the task needs a GPU (deep learning on images, audio, text) and the \
machine above can use one. The config and the machine have the last word.
- risks: up to {max_risks} items, most dangerous first. Each is specific to this project and \
names the action that prevents it. No generic advice.
- playbook: 6 to 8 short lines, under 1100 characters in total, the most important line first. \
Every later agent reads it, but it is cut at 1200 characters. Cover:
  - which kinds of model usually work on this data and which one to start with (kinds of \
method, not library names)
  - input preparation that usually helps, and what to avoid
  - quirks of the metric (labels or probabilities, thresholds, clipping)
  - how to handle imbalance, groups or time in training
  - how to stay inside the time limit (subsample, fewer epochs, smaller models)
  - how to spend the rounds: cheap broad tests first, expensive ones later
  Take facts about this data only from the task, the brief and the EDA. For what usually \
works you may use general ML knowledge. Write "probably" in a line you are not sure about.

Example of the style only, for another kind of task (do not copy its content):
analysis: Twelve bird species are scored by macro F1, so the result needs labels, and rare \
species weigh as much as common ones. The recordings come from 60 sites and finding_004 says \
the test clips come from sites that are not in train, so folds must keep a site together. \
Training a neural network is slow, but 5 folds of a small model fit the time limit.
validation_scheme: group_kfold
validation_reason: Test clips come from new sites (finding_004), so every fold keeps whole sites.
baseline_description: 5-second mel spectrogram crops, a small pretrained audio CNN, 3 epochs.
playbook:
- Start with a small pretrained audio model on mel crops. Try bigger models only after cheap \
tests.
- Prepare input: one sample rate, fixed length, per-clip normalization. Avoid augmentation \
that can change the label.
- Macro F1 needs labels: choose thresholds on validation, never on the holdout.
- Class counts are very uneven: class weights or balanced sampling.
- One training must stay under 6 minutes: subsample clips, 3 epochs, cache spectrograms.
- Spend early rounds on cheap tests (crop length, augmentation), the big model last.
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
        "data lists them. Ids: for a table the row position (0 based) in the training file, "
        "for a folder the file path relative to the project folder. Say which in "
        "id_description. Not needed for repeated_seeds.",
        "- Print the number of examples you found before splitting. It must match the number of "
        "rows (or files) of the training data. If it is 1 or far too small, you read the data "
        "wrongly: fix that first.",
        "- Only when the EDA structure above names duplicates or groups: keep rows with identical "
        "content (ignoring the id column) together, in the holdout or in one fold. Otherwise every "
        "example is independent: do not build groups and do not hash rows.",
        f"- kfold, stratified_kfold and group_kfold: use {DEFAULT_SPLIT_COUNT} validation splits, "
        "fewer when the data is tiny, but keep at least 30 examples in every validation split. "
        "For stratified_kfold every class must appear in every validation split.",
        "- single_split: one validation split of about 20 percent. time_split: also one "
        "validation split, the latest 20 percent of the examples that are not held out. A "
        "split is trained on all other non-holdout examples, so for time_split they must all "
        "be earlier.",
        f"- When there are at least {MIN_EXAMPLES_FOR_HOLDOUT} examples, hold out about "
        f"{int(HOLDOUT_FRACTION * 100)} percent of them (fold -1) first. With fewer examples "
        "hold out nothing.",
        "- The holdout follows the scheme: random for kfold, stratified on the target for "
        "stratified_kfold, whole groups for group_kfold, the latest examples in time order "
        "for time_split. Make the validation splits from the remaining examples with the same "
        "rule. For group_kfold with an imbalanced target, also spread the classes over the "
        "folds as evenly as the groups allow.",
        "- After saving, check the file: every example appears once, the holdout and folds do "
        "not overlap, no duplicate group named in the EDA crosses a boundary, and for time_split "
        "every validation and holdout example is later than all training examples. Fix it if a "
        "check fails.",
    ]
    if previous_error:
        rules.append(
            f"- A first attempt failed with: {trim_text(previous_error, MAX_ERROR_TEXT_CHARS)}. "
            "Write the simplest correct version: keep the scheme but drop every optional rule "
            "(duplicate groups and extra checks), and make sure the file follows the format "
            "below."
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
