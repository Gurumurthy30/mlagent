"""The Modeler: trains models.

Round 0 trains the baseline. A features round retrains the best setup so far on the new
features. A model round tries a new model family or setup. Code rejects a setup that was
already tried, and checks that the script's result line follows the contract.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from mlagent.agents.base import (
    AgentContext,
    build_common_prompt_values,
    display_name,
    render_prompt,
)
from mlagent.agents.coder import run_coder
from mlagent.digest import describe_brief
from mlagent.events import EventType
from mlagent.scoring import all_finite, make_setup_signature, mean_and_std
from mlagent.script_contract import (
    MODEL_RESULT_KEYS_TEXT,
    MODEL_SCRIPT_RULES_TEXT,
    SPLITS_FORMAT_TEXT,
)
from mlagent.settings import MAX_FEATURE_SOURCE_CHARS, RANDOM_SEED
from mlagent.state import ExperimentRun, Ledger

AGENT_KEY = "modeler"
AGENT_NAME = display_name(AGENT_KEY)
RESULT_TAG = "RESULT"
MAX_CHOICE_TRIES = 3

CHOOSE_PROMPT = """\
You are the Modeler of an ML team. You decide what to train next. Like an experienced ML \
engineer, you pick the experiment with the best chance of a real gain for its cost, and you \
use what earlier runs already taught the team.

{run_rules}

WHAT THE TEAM KNOWS (task, plan, EDA, features, runs so far, the Reviewer's instructions):
{digest}

THIS ROUND: {round_goal}

Think before you choose:
1. What does THIS ROUND ask for? Follow it. When it asks for the best setup on new features, \
return that same family and parameters. The new features version makes it a new run. When it \
asks for a new family or setup, do not just nudge parameters of an old run. Parameter search \
belongs to the Tuner.
2. Read the runs list. Which families worked, which crashed or were rejected, and why? Do not \
repeat a failed idea. Do not choose a family that was already tried without a clear reason.
3. Read the Reviewer's instructions and the EDA model_hints. They come first.
4. What does this data call for: size, imbalance, noise, groups or time, type of input? Use \
the playbook in the plan.
5. Will it finish? Estimate the training time from the data size and the machine description, \
over all validation splits plus the final fit. Choose a cheaper setup if it is close to the \
time limit.

Choose one model family and its parameters. Rules:
- It must fit the data type, the machine above and the GPU decision in the plan. Use only \
libraries listed in the environment above.
- Round 0 is the baseline: a simple, fast model that is hard to get wrong, with sensible \
default parameters. Later rounds may be bolder, but one meaningful change at a time, so the \
result tells the team what helped.
- When a model needs different input preparation than the features file gives (for example \
scaling for a linear model, one-hot for trees), say so in the reason. Do not hide it in the \
parameters.
- For deep learning name the architecture and the pretrained weights. Only name weights you \
are sure exist under that exact name.
- Parameters are a flat object with short names, numbers, strings and true or false: no \
code, no lists of settings to search. Be concrete, because they are logged and compared.
- Use the same family spelling as the runs list for the same family (for example lightgbm, \
not LightGBM Classifier), so the team's duplicate check works.
- A change that cannot change the result (seed, verbosity, number of threads) is not a new \
setup.
- The final output must suit the metric in the plan (probabilities, labels or numbers).
- Never repeat a setup from the runs list (same family, same parameters, same features \
version).
{blocked_note}
Return JSON: model_family, model_params (an object), reason (one sentence that names the run, \
finding or instruction that supports the choice, and what you expect)."""


class ModelChoice(BaseModel):
    model_family: str
    model_params: dict = {}
    reason: str = ""


class ModelRunResult(BaseModel):
    validation_scores: list[float]
    holdout_score: float | None = None
    extra_scores: dict[str, float] = {}
    validation_predictions_path: str = ""
    test_predictions_path: str
    model_path: str = ""
    notes: str = ""


def describe_round_goal(ledger: Ledger) -> str:
    if ledger.focus == "start":
        return f"Train the baseline from the plan: {ledger.plan.baseline_description}"
    return (
        "Choose a NEW model family or setup that is likely to beat the best run. Use the EDA "
        "model hints, the playbook and the Reviewer's instructions."
    )


def find_tried_signatures(ledger: Ledger) -> set[str]:
    """Crashed runs do not count: a crash is a bug in a script, not a verdict on the setup."""
    return {
        make_setup_signature(run.model_family, run.model_params, run.feature_version)
        for run in ledger.runs
        if not run.crashed
    }


def choose_setup(ledger: Ledger, agent_context: AgentContext) -> ModelChoice | None:
    """Pick the setup to train. Returns None when only already tried setups came back."""
    tried_signatures = find_tried_signatures(ledger)
    best_run = ledger.best_approved_run()
    if ledger.focus == "features" and best_run is not None:
        choice = ModelChoice(
            model_family=best_run.model_family,
            model_params=best_run.model_params,
            reason="the best setup so far, on the new features",
        )
        signature = make_setup_signature(
            choice.model_family, choice.model_params, ledger.current_feature_version
        )
        return None if signature in tried_signatures else choice

    common_values = build_common_prompt_values(AGENT_KEY, ledger, agent_context)
    blocked_note = ""
    for _ in range(MAX_CHOICE_TRIES):
        prompt = render_prompt(
            CHOOSE_PROMPT,
            **common_values,
            round_goal=describe_round_goal(ledger),
            blocked_note=blocked_note,
        )
        choice = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, ModelChoice)
        signature = make_setup_signature(
            choice.model_family, choice.model_params, ledger.current_feature_version
        )
        if signature not in tried_signatures:
            return choice
        blocked_note = (
            f"Your last choice repeated an earlier run: {choice.model_family} "
            f"{json.dumps(choice.model_params)}. Choose a different setup."
        )
    return None


def build_training_task(
    ledger: Ledger, choice: ModelChoice, experiment_folder: Path, agent_context: AgentContext
) -> str:
    workspace = agent_context.workspace
    plan = ledger.plan
    features_path = workspace.features_file(ledger.current_feature_version)
    features_source = features_path.read_text(encoding="utf-8")[:MAX_FEATURE_SOURCE_CHARS]
    direction = "higher" if plan.higher_is_better else "lower"
    instructions = ledger.reviews[-1].instructions if ledger.reviews else []
    instruction_text = "\n".join(f"- {instruction}" for instruction in instructions) or "none"
    return (
        "Train and score this model with ONE script.\n"
        f"Model: {choice.model_family}\n"
        f"Parameters: {json.dumps(choice.model_params)}\n"
        f"Reason: {choice.reason}\n\n"
        f"Primary metric: {plan.primary_metric} ({direction} is better)\n"
        f"Extra metrics: {', '.join(plan.extra_metrics) or 'none'}\n"
        f"Seed: {RANDOM_SEED}\n"
        f"Use the GPU: {'yes' if plan.use_gpu else 'no'}\n"
        f"Project folder (read only): {workspace.project_folder}\n"
        f"Experiment folder (save everything here): {experiment_folder}\n"
        f"Splits file: {workspace.splits_path}\n"
        f"Features file: {features_path} (add {workspace.features_folder} to sys.path, then "
        f"import {features_path.stem})\n"
        f"Deliverable format of the predictions: {ledger.brief.deliverable_format}\n"
        f"Instructions from the Reviewer:\n{instruction_text}\n\n"
        f"SCRIPT RULES:\n{MODEL_SCRIPT_RULES_TEXT}\n\n"
        f"{SPLITS_FORMAT_TEXT}\n\n"
        f"{describe_brief(ledger)}\n\n"
        f"Features file source:\n{features_source}"
    )


def resolve_output_path(path_text: str, experiment_folder: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else experiment_folder / path


def fill_run_from_result(run: ExperimentRun, result_data: dict, experiment_folder: Path) -> str:
    """Copy the script's result into the run. Returns the problem found, or '' when it is fine."""
    try:
        result = ModelRunResult.model_validate(result_data)
    except ValidationError as error:
        return f"The result line did not follow the contract: {error}"
    if not all_finite(result.validation_scores):
        return "validation_scores must be a non-empty list of finite numbers."

    test_predictions_path = resolve_output_path(result.test_predictions_path, experiment_folder)
    if not test_predictions_path.is_file():
        return f"The test predictions file does not exist: {test_predictions_path}"
    validation_predictions_text = ""
    if result.validation_predictions_path:
        validation_path = resolve_output_path(result.validation_predictions_path, experiment_folder)
        if not validation_path.is_file():
            return f"The validation predictions file does not exist: {validation_path}"
        validation_predictions_text = str(validation_path)

    model_path_text = ""
    if result.model_path:
        model_path = resolve_output_path(result.model_path, experiment_folder)
        if not model_path.exists():
            return f"The model file does not exist: {model_path}"
        model_path_text = str(model_path)

    run.validation_scores = result.validation_scores
    run.score_mean, run.score_std = mean_and_std(result.validation_scores)
    run.holdout_score = result.holdout_score
    run.extra_scores = result.extra_scores
    run.validation_predictions_path = validation_predictions_text
    run.test_predictions_path = str(test_predictions_path)
    run.model_path = model_path_text
    return ""


def run_modeler(ledger: Ledger, agent_context: AgentContext) -> ExperimentRun:
    """Train one model. A crash is returned as a run with an error, never raised."""
    event_bus = agent_context.event_bus
    experiment_id = ledger.next_experiment_id()
    experiment_folder = agent_context.workspace.experiment_folder(experiment_id)
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, f"{experiment_id}, focus {ledger.focus}")

    run = ExperimentRun(
        experiment_id=experiment_id,
        focus=ledger.focus,
        feature_version=ledger.current_feature_version,
        iteration=ledger.iteration,
        artifact_dir=str(experiment_folder),
    )
    choice = choose_setup(ledger, agent_context)
    if choice is None:
        run.error = "Every setup the Modeler chose had already been tried."
        event_bus.emit(EventType.WARNING, AGENT_NAME, run.error)
        return run

    run.model_family = choice.model_family
    run.model_params = choice.model_params
    run.description = f"{choice.model_family} {json.dumps(choice.model_params)}"
    event_bus.emit(EventType.DECISION, AGENT_NAME, f"training {run.description}")

    coder_result = run_coder(
        task_text=build_training_task(ledger, choice, experiment_folder, agent_context),
        result_tag=RESULT_TAG,
        result_keys_text=MODEL_RESULT_KEYS_TEXT,
        working_folder=experiment_folder,
        script_name="train_model.py",
        caller_key=AGENT_KEY,
        ledger=ledger,
        agent_context=agent_context,
    )
    run.script_path = str(coder_result.script_path)
    if not coder_result.succeeded:
        run.error = coder_result.error_text or "The Coder could not produce a working script."
    else:
        run.error = fill_run_from_result(run, coder_result.result_data, experiment_folder)

    if run.crashed:
        event_bus.emit(EventType.WARNING, AGENT_NAME, f"{experiment_id} crashed")
    else:
        event_bus.emit(
            EventType.AGENT_FINISHED, AGENT_NAME, f"{experiment_id} score {run.score_mean:.4f}"
        )
    return run
