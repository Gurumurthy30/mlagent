"""The ML Lead: decides validation scheme, baseline description, risks, and playbook.

The LLM writes the plan. The Coder then writes splits.json and code checks it.
Every later agent follows this plan.
"""

from typing import Literal

from pydantic import BaseModel

from mlagent.agents.base import AgentContext, AgentFailedError
from mlagent.agents.coder import last_error_line, run_coder
from mlagent.digest import build_digest, shorten_text
from mlagent.events import EventType
from mlagent.script_contract import SPLITS_FORMAT_TEXT, check_splits_file
from mlagent.settings import (
    BIG_FILE_MB,
    DEFAULT_SPLIT_COUNT,
    HOLDOUT_FRACTION,
    MIN_EXAMPLES_FOR_HOLDOUT,
    RANDOM_SEED,
)
from mlagent.state import ExperimentPlan, Ledger

AGENT_KEY = "ml_lead"
AGENT_NAME = "ML Lead"
RESULT_TAG = "SPLITS"

ValidationScheme = Literal[
    "kfold", "stratified_kfold", "group_kfold", "time_split", "single_split", "repeated_seeds"
]

PLAN_PROMPT = """\
You are the ML Lead of an autonomous ML team. You decide the validation scheme, the baseline, and the technical playbook.
You do not write code or choose the final model.

GOAL:
A reliable validation scheme that prevents leakage and reflects the real test distribution, a fast baseline, and a playbook guiding the team.

SITUATION:
{digest}

HOW TO THINK:
- Validation scheme must mirror how the test data was collected (time, groups, or random). Ignoring grouping or time causes severe leakage.
- The baseline must be simple, robust, fast, and hard to break. Every subsequent model must beat it.
- Order of experiments: broad cheap tests first (fast algorithms, standard features), expensive setups later.
- Deep learning is needed only for unstructured data (images, audio, text) or large complex sequences. Tabular data usually favors tree ensembles.

YOU MAY:
- Recommend fewer splits or single split if dataset size or compute makes full cross-validation prohibitively slow.
- Disagree with ambitious modeling up-front and insist on a fast baseline first.

OUTPUT:
Provide a JSON object with:
- thinking: 3-5 sentences analyzing the task, test differences, validation risks, and compute budget.
- validation_scheme: one of "kfold", "stratified_kfold", "group_kfold", "time_split", "single_split", "repeated_seeds".
- validation_reason: why this scheme fits the data structure and prevents leakage.
- baseline_description: simplest sensible first model and input preparation.
- target_score: target score if requested by user, else null.
- use_gpu: true only if deep learning is needed and GPU is available.
- risks: list of up to 5 project-specific risks and mitigations.
- playbook: guidance lines covering model families, input prep, metric quirks, and round spending."""


class PlanDraft(BaseModel):
    thinking: str = ""
    validation_scheme: ValidationScheme = "kfold"
    validation_reason: str = ""
    baseline_description: str = ""
    target_score: float | None = None
    use_gpu: bool = False
    risks: list[str] = []
    playbook: str = ""


SPLITS_RESULT_KEYS = """\
- "splits_path": the path you saved the file to
- "n_splits": integer
- "n_examples": number of entries in example_ids (0 when you wrote seeds)
- "n_holdout": number of examples with fold -1"""


def run_ml_lead(ledger: Ledger, agent_context: AgentContext) -> ExperimentPlan:
    event_bus = agent_context.event_bus
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, "writing the plan")

    # Step 1: Draft the plan using LLM
    digest = build_digest(AGENT_KEY, ledger)
    prompt = PLAN_PROMPT.format(digest=digest)
    draft = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, PlanDraft)

    # Resolve GPU setting with system capability
    can_gpu = agent_context.environment.can_train_on_gpu
    use_gpu_config = agent_context.config.compute.use_gpu
    use_gpu = False if use_gpu_config == "never" or not can_gpu else (True if use_gpu_config == "always" else draft.use_gpu)

    plan = ExperimentPlan(
        validation_scheme=draft.validation_scheme,
        validation_reason=draft.validation_reason.strip(),
        baseline_description=draft.baseline_description.strip(),
        target_score=draft.target_score,
        use_gpu=use_gpu,
        risks=draft.risks[:5],
        playbook=shorten_text(draft.playbook.strip(), 1200),
    )

    # Step 2: Ask Coder to create splits.json
    workspace = agent_context.workspace
    splits_file = workspace.splits_path
    hint_text = "; ".join(ledger.eda.validation_hints) or "none"
    task_text = (
        f"Create the validation splits for this project and save as JSON to {splits_file}.\n"
        f"Scheme: {plan.validation_scheme}. Reason: {plan.validation_reason}\n"
        f"Structure from EDA: {hint_text}\n"
        f"Random seed: {RANDOM_SEED}. Default splits: {DEFAULT_SPLIT_COUNT}.\n"
        f"Project folder: {workspace.project_folder}\n"
        f"Rules: List each training example once (row index for table, relative path for files). "
        f"Holdout: hold out ~{int(HOLDOUT_FRACTION * 100)}% (fold -1) if >= {MIN_EXAMPLES_FOR_HOLDOUT} examples.\n"
        f"Do not exceed {BIG_FILE_MB} MB in memory at once.\n\n"
        f"{SPLITS_FORMAT_TEXT}"
    )

    last_err = ""
    for try_num in range(1, 3):
        coder_task = task_text if not last_err else f"{task_text}\n\nPrevious attempt failed with:\n{last_err}"
        coder_res = run_coder(
            task_text=coder_task,
            result_tag=RESULT_TAG,
            result_keys_text=SPLITS_RESULT_KEYS,
            working_folder=workspace.artifacts_folder / AGENT_KEY,
            script_name=f"make_splits_try_{try_num}.py",
            caller_key=AGENT_KEY,
            ledger=ledger,
            agent_context=agent_context,
        )
        if coder_res.succeeded:
            problems = check_splits_file(splits_file)
            if not problems:
                last_err = ""
                break
            last_err = "Splits file failed validation: " + " ".join(problems)
        else:
            last_err = last_error_line(coder_res.error_text)
        event_bus.emit(EventType.WARNING, AGENT_NAME, f"splits try {try_num} failed: {last_err}")

    if last_err:
        raise AgentFailedError(f"Could not create validation splits: {last_err}")

    plan.splits_path = str(splits_file)
    event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, f"plan ready: {plan.validation_scheme}")
    return plan
