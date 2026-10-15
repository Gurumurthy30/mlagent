"""The Data Analyst: exploratory data analysis working round by round.

Each visit performs exactly ONE round:
1. think (picks up to 5 questions, decides if done)
2. Coder run (executes inspection)
3. read (extracts findings and updates open questions)
When done or max rounds reached, summarizes findings.
"""

from __future__ import annotations

from pydantic import BaseModel

from mlagent.agents.base import AgentContext, facts_to_text
from mlagent.agents.coder import run_coder
from mlagent.digest import build_digest
from mlagent.events import EventType
from mlagent.settings import BIG_FILE_MB
from mlagent.state import EdaReport, FeatureIdea, Finding, Ledger

AGENT_KEY = "data_analyst"
AGENT_NAME = "Data Analyst"
RESULT_TAG = "FINDINGS"

EDA_RESULT_KEYS = """\
- "answers": list of objects {"question": text, "result": short factual text or numbers (at most 800 chars)}
- "problems": list of strings for anything that failed"""

THINK_PROMPT = """\
You are the Data Analyst of an autonomous ML team. You explore the dataset iteratively.
You do not build predictive models.

GOAL:
Ask targeted questions to uncover data distributions, leakage risks, train/test shifts, and fair validation structures.

SITUATION:
{digest}

OPEN QUESTIONS SO FAR:
{open_questions_text}

THIS IS EDA ROUND {round_number} OF {max_rounds}.

HOW TO THINK:
- If open questions list is empty (round 1), initialize 5-8 essential questions (target distribution, missing values, train/test differences, leakage candidates, groups/time structure).
- Otherwise, pick at most 5 concrete questions from open questions or follow-ups to high-priority findings.
- Questions must be concrete and answerable by a short Python script returning numbers or small summaries.
- Decide if the team already knows enough about distributions, leakage, and validation structure: set done=true.

YOU MAY:
- Declare done=true early if data is clean and baseline needs are obvious.
- Add new open questions if prior findings reveal unexpected anomalies.

OUTPUT:
Return JSON:
- thinking: 2-3 sentences assessing what is known and what needs checking.
- open_questions: list of all questions still worth checking eventually.
- questions_to_ask: list of up to 5 questions to answer in THIS round.
- done: boolean, true if the team already knows enough to proceed to planning/modeling."""

READ_PROMPT = """\
You are the Data Analyst of an ML team. An inspection script answered your questions.
Extract actionable findings for the team.

SITUATION:
{digest}

QUESTIONS ASKED:
{questions_text}

SCRIPT OUTPUT:
{facts_text}

HOW TO THINK:
- State facts with exact numbers from the script output. Never invent or estimate numbers.
- For each finding, give an implication (what the team should do in cleaning, features, splitting, or modeling).
- Mark priority as "high" if it dictates validation scheme or reveals leakage/critical issues, "medium" if it guides features/models, "low" otherwise.
- Update open questions: remove questions answered by this output.

YOU MAY:
- Return an empty findings list if questions revealed nothing surprising or actionable.

OUTPUT:
Return JSON:
- findings: list of objects with statement, evidence, implication, and priority ("high"|"medium"|"low").
- open_questions: updated list of remaining open questions."""

SUMMARY_PROMPT = """\
You are the Data Analyst summarizing the completed exploration for the team.

SITUATION:
{digest}

ALL FINDINGS:
{findings_text}

HOW TO THINK:
- Write a 2 to 3 sentence summary of the dataset: main traits, risks, and recommended strategy.
- Group issues, feature ideas, leakage risks, validation hints, and model hints from the findings.
- Tie feature ideas to finding IDs where applicable.

OUTPUT:
Return JSON:
- summary: 2-3 sentences summarizing the key takeaways from EDA.
- issues: list of data issues.
- feature_ideas: list of objects {"description": str, "finding_id": str}.
- leakage_risks: list of leakage risks.
- validation_hints: list of validation hints (grouping, time, stratification).
- model_hints: list of model hints."""


class ThinkDraft(BaseModel):
    thinking: str = ""
    open_questions: list[str] = []
    questions_to_ask: list[str] = []
    done: bool = False


class ReadDraft(BaseModel):
    findings: list[Finding] = []
    open_questions: list[str] = []


class SummaryDraft(BaseModel):
    summary: str = ""
    issues: list[str] = []
    feature_ideas: list[FeatureIdea] = []
    leakage_risks: list[str] = []
    validation_hints: list[str] = []
    model_hints: list[str] = []


def run_data_analyst(ledger: Ledger, agent_context: AgentContext) -> EdaReport:
    """Execute exactly ONE round of EDA and return updated EdaReport."""
    event_bus = agent_context.event_bus
    working_eda = ledger.eda.model_copy(deep=True)
    working_eda.eda_loops_used += 1
    round_number = working_eda.eda_loops_used
    max_rounds = agent_context.config.limits.max_eda_loops

    event_bus.emit(EventType.AGENT_STARTED, AGENT_NAME, f"EDA round {round_number} of {max_rounds}")

    # Step 1: Think (choose up to 5 questions)
    digest = build_digest(AGENT_KEY, ledger)
    open_q_text = "\n".join(f"- {q}" for q in working_eda.open_questions) or "(none yet)"
    think_prompt = THINK_PROMPT.format(
        digest=digest,
        open_questions_text=open_q_text,
        round_number=round_number,
        max_rounds=max_rounds,
    )
    think_res = agent_context.llm_client.ask_structured(AGENT_NAME, think_prompt, ThinkDraft)

    working_eda.open_questions = think_res.open_questions
    questions_to_ask = think_res.questions_to_ask[:5]
    if think_res.done or not questions_to_ask:
        working_eda.done = True

    # Step 2: Coder run
    new_findings: list[Finding] = []
    high_count = 0
    if questions_to_ask and not working_eda.done:
        workspace = agent_context.workspace
        q_list_text = "\n".join(f"{i}. {q}" for i, q in enumerate(questions_to_ask, 1))
        coder_task = (
            f"Answer these {len(questions_to_ask)} questions about the dataset in ONE script:\n"
            f"{q_list_text}\n\n"
            f"Project folder (read-only): {workspace.project_folder}\n"
            f"Do not load files larger than {BIG_FILE_MB} MB in full; sample or chunk.\n"
            "Return JSON FINDINGS with answers and problems."
        )

        coder_res = run_coder(
            task_text=coder_task,
            result_tag=RESULT_TAG,
            result_keys_text=EDA_RESULT_KEYS,
            working_folder=workspace.artifacts_folder / "eda",
            script_name=f"eda_round_{round_number}.py",
            caller_key=AGENT_KEY,
            ledger=ledger,
            agent_context=agent_context,
        )

        # Step 3: Read
        facts_text = facts_to_text(coder_res.result_data if coder_res.succeeded and isinstance(coder_res.result_data, dict) else {"error": coder_res.error_text})
        read_prompt = READ_PROMPT.format(
            digest=digest,
            questions_text=q_list_text,
            facts_text=facts_text,
        )
        read_res = agent_context.llm_client.ask_structured(AGENT_NAME, read_prompt, ReadDraft)
        working_eda.open_questions = read_res.open_questions

        # Add findings with code-assigned IDs
        existing_statements = {f.statement.strip().lower() for f in working_eda.findings}
        for f in read_res.findings:
            if f.statement.strip().lower() not in existing_statements:
                finding_id = f"finding_{len(working_eda.findings) + 1:03d}"
                f.finding_id = finding_id
                working_eda.findings.append(f)
                new_findings.append(f)
                existing_statements.add(f.statement.strip().lower())
                if f.priority == "high":
                    high_count += 1

    # Check if EDA finishes this round
    if working_eda.eda_loops_used >= max_rounds:
        working_eda.done = True

    # If done or finished, run summarize
    if working_eda.done:
        findings_text = "\n".join(
            f"- {f.finding_id} [{f.priority}] {f.statement} (evidence: {f.evidence}) -> {f.implication}"
            for f in working_eda.findings
        ) or "No specific findings."
        sum_prompt = SUMMARY_PROMPT.format(digest=digest, findings_text=findings_text)
        summary_res = agent_context.llm_client.ask_structured(AGENT_NAME, sum_prompt, SummaryDraft)
        working_eda.summary = summary_res.summary
        working_eda.issues = summary_res.issues
        working_eda.feature_ideas = summary_res.feature_ideas
        working_eda.leakage_risks = summary_res.leakage_risks
        working_eda.validation_hints = summary_res.validation_hints
        working_eda.model_hints = summary_res.model_hints

    # CLI Summary line
    summary_line = (
        f"EDA round {round_number}/{max_rounds}: asked {len(questions_to_ask)} questions, "
        f"{len(new_findings)} findings ({high_count} high). Still open: {len(working_eda.open_questions)}."
    )
    event_bus.emit(EventType.INFO, AGENT_NAME, summary_line)
    if working_eda.done and working_eda.summary:
        event_bus.emit(EventType.AGENT_FINISHED, AGENT_NAME, f"EDA summary: {working_eda.summary}")

    return working_eda
