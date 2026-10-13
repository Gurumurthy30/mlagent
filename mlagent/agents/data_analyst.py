"""The Data Analyst: exploratory data analysis, the way a careful human analyst works.

Ask a question, check the data, read the result, ask the next question. Code tracks which
checklist items are covered, so the LLM cannot declare itself done while items are still open.
The loop ends when every item is covered, when a round finds nothing new, or at the round limit.
"""

from datetime import datetime

from pydantic import BaseModel

from mlagent.agents.base import (
    AgentContext,
    build_common_prompt_values,
    display_name,
    facts_to_text,
    format_numbered_list,
    render_prompt,
)
from mlagent.agents.coder import last_error_line, run_coder
from mlagent.digest import describe_brief, describe_eda
from mlagent.events import EventType
from mlagent.settings import (
    BIG_FILE_MB,
    MAX_CHECKLIST_ITEMS,
    MAX_QUESTIONS_PER_ROUND,
)
from mlagent.state import (
    ChecklistStatus,
    EdaReport,
    FeatureIdea,
    Finding,
    Ledger,
    Priority,
)

AGENT_KEY = "data_analyst"
AGENT_NAME = display_name(AGENT_KEY)
RESULT_TAG = "FINDINGS"
EDA_FOLDER_NAME = "eda"

DEFAULT_CHECKLIST = [
    "label or target distribution",
    "missing, empty or corrupt data",
    "train versus test differences",
    "duplicates and groups",
    "leakage candidates",
]

RESULT_KEYS_TEXT = """\
- "answers": list with one object per question: {"question": the question text, "result": a \
short factual text with the key numbers or a compact table, at most 800 characters. Say when the \
result comes from a sample and how big the sample was}
- "problems": list of strings for anything that could not be computed, with the reason"""

CHECKLIST_PROMPT = """\
You are the Data Analyst of an ML team. Before you explore the data you write your coverage \
checklist: the things that must be checked before anyone builds a model. The team cannot move \
on until every item is covered, so every item must be worth checking and checkable by a script.

{run_rules}

WHAT THE TEAM KNOWS:
{digest}

First work out from the task and the project files what kind of problem this is: the data type \
(tables, images, text, audio, video, time series, graphs, or a mix), what is predicted (a class, \
a number, a sequence, a ranking, clusters, or nothing), and how the result is scored. Then write \
8 to 12 checklist items for exactly this problem.

Cover these areas, in the words of this data. Skip an area only if it truly cannot matter here:
- the target or labels: distribution, imbalance, noise, rare or unseen values
- data quality: missing, empty, corrupt or inconsistent records or files
- train versus test: differences in range, categories, size, source or time
- duplicates and near-duplicates, inside train and between train and test
- structure that decides a fair split: groups, time order, users, speakers, patients, sources
- leakage: anything that could reveal the answer or only exists after the outcome
- signal: whether the inputs show any quick, visible link to the target
- what is special about this data type (image sizes, text length, sampling rate, gaps in time)
- anything the project documents claim that the data must be checked against

Rules:
- Each item is one short phrase that one script can answer. No explanations.
- No duplicates and no vague items such as "explore the data".
- Data checking only. Do not list modeling or training steps.
Return JSON: items (list of strings)."""

THINK_PROMPT = """\
You are the Data Analyst of an ML team. You explore the data like a detective, not like a form \
filler: ask a question, check the data, read the result, then let the result decide the next \
question.

{run_rules}

WHAT THE TEAM KNOWS (task, files, findings so far):
{digest}

CHECKLIST ITEMS STILL OPEN:
{open_items}

THIS IS EDA ROUND {round_number} OF {max_rounds}.

Think it through before you write the questions:
1. What is already established? Never ask it again.
2. What is surprising, odd or contradictory in the findings so far? Every surprise deserves a \
follow-up that asks why: which rows, files or values, and how many.
3. What could silently ruin the final score: leakage, a train and test mismatch, a split that is \
too easy, a quirk of the metric?
4. Which open items matter most for this task?

Then write 2 to {max_questions} questions:
- Early rounds are broad: cover the most important open items first. Later rounds dig deeper: \
follow up on high-priority findings and surprises. In the last round, cover what is still open.
- Each question is specific. Name the file, column or group, and say what to measure: counts, \
shares, ranges, a train versus test comparison, or how well one input predicts the target.
- Each question can be answered by one short script and has a short, mostly numeric answer.
- Prefer questions whose answer would change a decision about cleaning, features, the split or \
the model. Skip trivia.
- covers: copy the open checklist items exactly. A follow-up question may cover none, then use \
an empty list.
Return JSON: questions (list of objects with text and covers)."""

READ_PROMPT = """\
You are the Data Analyst of an ML team. A script answered your questions. Turn its output into \
findings the rest of the team can act on.

{run_rules}

WHAT THE TEAM KNOWS:
{digest}

YOUR QUESTIONS, with the checklist items each one was meant to cover:
{questions_text}

THE SCRIPT OUTPUT:
{facts_text}

Rules for findings:
- Use only the output above. Never invent or round up numbers. If a result came from a sample, \
say so in the evidence.
- One fact per finding. Do not repeat what the team already knows. Skip trivial facts.
- statement: what is true, in one sentence.
- evidence: the exact numbers from the output.
- implication: what the team should do about it, concrete: a cleaning step, a feature or input \
preparation idea, a way to split, a metric concern, or a model choice.
- priority: high when it changes the plan or could make the score misleading, medium when it \
helps a lot, low otherwise.
- Pay most attention to surprises, contradictions with the task description or the project \
documents, and train versus test differences.
- If the output shows nothing new, return an empty findings list. That is a valid answer.
- If the script reported problems, do not guess the missing answers.

Rules for coverage:
- covered: the output really answers the item. A question that failed or was only partly \
answered does not cover it.
- not_applicable: the item cannot apply to this data, for example a time check when there is no \
time information. Use it only when the output or the brief shows that.
- Copy each item text exactly. Leave out items the output did not touch.
Return JSON: findings, coverage."""

SUMMARY_PROMPT = """\
You are the Data Analyst of an ML team. The exploration is finished. Write the summary the rest \
of the team will work from. Every later decision about features, validation and models starts \
here, so be specific and honest.

{run_rules}

THE USER'S TASK: {user_task}

{brief_text}

{findings_text}

Rules: use only the findings above. Never add facts that are not in them. An empty list is \
better than a guess.
- issues: the most important data problems, most important first, each with the key number
- feature_ideas: concrete ideas, each tied to the finding it comes from in finding_id (for \
example finding_003). Leave finding_id empty only if no finding supports the idea. For images, \
text, audio and other non-table data these are ideas for input preparation, such as resizing, \
augmentation, cleaning, embeddings or spectrograms.
- leakage_risks: columns, files or steps that may reveal the answer, with the reason
- validation_hints: structure that decides how to split the data: time order, groups, class \
imbalance, speakers, near-duplicates, train versus test shift
- model_hints: what kind of model or method suits this data and why, based on size, structure, \
noise and signal seen in the findings
Return JSON: issues, feature_ideas, leakage_risks, validation_hints, model_hints."""


class ChecklistDraft(BaseModel):
    items: list[str]


class Question(BaseModel):
    text: str
    covers: list[str] = []


class QuestionPlan(BaseModel):
    questions: list[Question]


class NewFinding(BaseModel):
    statement: str
    evidence: str = ""
    implication: str = ""
    priority: Priority = "medium"


class CoverageMark(BaseModel):
    item: str
    status: ChecklistStatus


class RoundReading(BaseModel):
    findings: list[NewFinding]
    coverage: list[CoverageMark]


class EdaSummary(BaseModel):
    issues: list[str]
    feature_ideas: list[FeatureIdea]
    leakage_risks: list[str]
    validation_hints: list[str]
    model_hints: list[str]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def open_checklist_items(eda: EdaReport) -> list[str]:
    return [item for item, status in eda.checklist.items() if status == "open"]


def describe_questions(questions: list[Question]) -> str:
    lines = []
    for question in questions:
        covers_text = "; ".join(question.covers) or "none"
        lines.append(f"{question.text} (covers: {covers_text})")
    return format_numbered_list(lines)


def create_checklist(
    working_ledger: Ledger, agent_context: AgentContext
) -> dict[str, ChecklistStatus]:
    common_values = build_common_prompt_values(AGENT_KEY, working_ledger, agent_context)
    prompt = render_prompt(CHECKLIST_PROMPT, **common_values)
    draft = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, ChecklistDraft)

    items: list[str] = []
    for item in draft.items:
        is_new = normalize_text(item) not in [normalize_text(known) for known in items]
        if item.strip() and is_new:
            items.append(item.strip())
    items = items[:MAX_CHECKLIST_ITEMS] or DEFAULT_CHECKLIST
    return {item: "open" for item in items}


def think(working_ledger: Ledger, agent_context: AgentContext, round_number: int) -> list[Question]:
    common_values = build_common_prompt_values(AGENT_KEY, working_ledger, agent_context)
    prompt = render_prompt(
        THINK_PROMPT,
        **common_values,
        open_items="\n".join(f"- {item}" for item in open_checklist_items(working_ledger.eda)),
        round_number=round_number,
        max_rounds=agent_context.config.limits.max_eda_loops,
        max_questions=MAX_QUESTIONS_PER_ROUND,
    )
    question_plan = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, QuestionPlan)
    return question_plan.questions[:MAX_QUESTIONS_PER_ROUND]


def build_check_task(
    questions: list[Question], working_ledger: Ledger, agent_context: AgentContext
) -> str:
    return (
        "Answer these questions about the data with ONE script.\n"
        f"Questions:\n{format_numbered_list([question.text for question in questions])}\n\n"
        f"What is known about the project:\n{describe_brief(working_ledger)}\n\n"
        "Rules:\n"
        f"- Project folder (read only): {agent_context.workspace.project_folder}\n"
        f"- Never fully load a file larger than {BIG_FILE_MB} MB. Use a sample, chunks or "
        "counts, and say so in the result.\n"
        "- Compare train and test using the files marked train and test in the brief.\n"
        "- Print only the final result line. Keep each result short."
    )


def read_output(
    working_ledger: Ledger,
    agent_context: AgentContext,
    questions: list[Question],
    script_output: dict,
) -> RoundReading:
    common_values = build_common_prompt_values(AGENT_KEY, working_ledger, agent_context)
    prompt = render_prompt(
        READ_PROMPT,
        **common_values,
        questions_text=describe_questions(questions),
        facts_text=facts_to_text(script_output),
    )
    return agent_context.llm_client.ask_structured(AGENT_NAME, prompt, RoundReading)


def apply_reading(working_ledger: Ledger, reading: RoundReading) -> int:
    """Add the new findings and mark checklist coverage. Returns how many findings were new."""
    eda = working_ledger.eda
    known_statements = {normalize_text(finding.statement) for finding in eda.findings}
    new_finding_count = 0
    for new_finding in reading.findings:
        statement_key = normalize_text(new_finding.statement)
        if not statement_key or statement_key in known_statements:
            continue
        known_statements.add(statement_key)
        eda.findings.append(
            Finding(
                finding_id=working_ledger.next_finding_id(),
                statement=new_finding.statement.strip(),
                evidence=new_finding.evidence,
                implication=new_finding.implication,
                priority=new_finding.priority,
            )
        )
        new_finding_count += 1

    items_by_key = {normalize_text(item): item for item in eda.checklist}
    for coverage_mark in reading.coverage:
        checklist_item = items_by_key.get(normalize_text(coverage_mark.item))
        if checklist_item is not None:
            eda.checklist[checklist_item] = coverage_mark.status
    return new_finding_count


def summarize(working_ledger: Ledger, agent_context: AgentContext) -> None:
    eda = working_ledger.eda
    common_values = build_common_prompt_values(AGENT_KEY, working_ledger, agent_context)
    prompt = render_prompt(
        SUMMARY_PROMPT,
        run_rules=common_values["run_rules"],
        user_task=working_ledger.user_task,
        brief_text=describe_brief(working_ledger),
        findings_text=describe_eda(working_ledger),
    )
    summary = agent_context.llm_client.ask_structured(AGENT_NAME, prompt, EdaSummary)

    known_finding_ids = {finding.finding_id for finding in eda.findings}
    eda.issues = summary.issues
    eda.feature_ideas = [
        FeatureIdea(
            description=idea.description,
            finding_id=idea.finding_id if idea.finding_id in known_finding_ids else "",
        )
        for idea in summary.feature_ideas
    ]
    eda.leakage_risks = summary.leakage_risks
    eda.validation_hints = summary.validation_hints
    eda.model_hints = summary.model_hints


def run_data_analyst(
    ledger: Ledger, agent_context: AgentContext, question: str | None = None
) -> EdaReport:
    """Run the EDA loop. With a question (the Reviewer asked for more data work) run one round."""
    event_bus = agent_context.event_bus
    working_ledger = ledger.model_copy(deep=True)
    eda = working_ledger.eda
    is_reentry = question is not None
    max_rounds = 1 if is_reentry else agent_context.config.limits.max_eda_loops
    start_message = f"answering: {question}" if is_reentry else "starting the exploration"
    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, start_message)

    if not eda.checklist:
        eda.checklist = create_checklist(working_ledger, agent_context)
    working_folder = agent_context.workspace.artifacts_folder / EDA_FOLDER_NAME

    failure_texts: list[str] = []
    for round_number in range(1, max_rounds + 1):
        if is_reentry:
            questions = [Question(text=question)]
            script_name = f"eda_reentry_{datetime.now().strftime('%H%M%S')}.py"
        else:
            questions = think(working_ledger, agent_context, round_number)
            script_name = f"eda_round_{round_number}.py"
        if not questions:
            break

        coder_result = run_coder(
            task_text=build_check_task(questions, working_ledger, agent_context),
            result_tag=RESULT_TAG,
            result_keys_text=RESULT_KEYS_TEXT,
            working_folder=working_folder,
            script_name=script_name,
            caller_key=AGENT_KEY,
            ledger=working_ledger,
            agent_context=agent_context,
        )
        new_finding_count = 0
        if coder_result.succeeded:
            reading = read_output(
                working_ledger, agent_context, questions, coder_result.result_data
            )
            new_finding_count = apply_reading(working_ledger, reading)
        else:
            failure_texts.append(coder_result.error_text)

        if not is_reentry:
            eda.eda_loops_used = round_number
        event_bus.emit(
            EventType.INFO,
            AGENT_NAME,
            f"round {round_number} of {max_rounds}: {new_finding_count} new findings, "
            f"{len(open_checklist_items(eda))} checklist items still open",
        )
        if new_finding_count == 0 or not open_checklist_items(eda):
            break

    if eda.findings:
        summarize(working_ledger, agent_context)
    if failure_texts:
        eda.issues.append(f"An EDA script failed: {last_error_line(failure_texts[-1])}")

    event_bus.emit(
        EventType.AGENT_FINISHED, AGENT_NAME, f"{len(eda.findings)} findings, summary written"
    )
    return eda
