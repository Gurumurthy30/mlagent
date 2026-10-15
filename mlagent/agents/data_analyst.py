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
You are the Data Analyst of an autonomous ML team. Your purpose is hypothesis-driven exploratory \
data analysis to uncover target drivers, validation structures, data leakage, and feature leads. \
You do not build final predictive models.

DECISION RULES (in priority order):
1. HYPOTHESIS-DRIVEN INVESTIGATION: Every question must test a concrete hypothesis about:
   - Target dynamics: class balance, extreme values, multimodal distribution, log-normality.
   - Informative missingness: whether missing values correlate with target or distinguish segments.
   - Covariate shift & adversarial separation: whether any feature separates train from test.
   - Relational & group structure: composite strings (e.g. "deck/num/side"), group IDs, time order.
   - Leak hunt: single features that predict the target suspiciously well (AUC > 0.90 or r > 0.85).
2. CHECKLIST IS THE FLOOR, NOT THE CEILING: In Round 1, establish baseline coverage of target, \
missing values, and train/test shape. In subsequent rounds, pursue discovery leads uncovered by findings.
3. CONCRETE SCRIPTABILITY: Each question must be directly answerable by a short Python script returning \
exact counts, percentages, correlations, or compact frequency tables.
4. STOP DECISION: Set done=true when all high-priority hypotheses regarding target drivers, leakage risks, \
and validation structure are resolved, or when rounds are exhausted. Do not stop simply because a list is covered.

WORKED EXAMPLE:
- Poor Question: "Analyze all columns." (Too vague, unscriptable).
- Grandmaster Question: "Calculate target distribution across categories of HomePlanet and CryoSleep, \
and check if missing CryoSleep correlates with zero luxury spending." (Hypothesis-driven, yields exact numbers).

SITUATION:
{digest}

OPEN QUESTIONS SO FAR:
{open_questions_text}

THIS IS EDA ROUND {round_number} OF {max_rounds}.

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 3-5 sentences assessing what the team knows, what hypotheses remain open, and the rationale for next questions.
- open_questions: list of all questions still worth checking eventually.
- questions_to_ask: list of up to 5 prioritized questions to answer in THIS round (empty if done).
- done: boolean, true only if all critical validation and target hypotheses are resolved."""

READ_PROMPT = """\
You are the Data Analyst of an ML team. An inspection script answered your questions. \
Your purpose is to turn raw script execution outputs into rigorous, actionable findings that \
directly guide splitting, feature engineering, and modeling.

DECISION RULES (in priority order):
1. NO IMPLICATION, NO FINDING: Every finding MUST include a concrete implication (an action for \
splitting, a feature idea, a model choice, or a risk). Facts without implications must not be recorded.
2. EXACT EVIDENCE REQUIRED: Every finding statement must cite exact numbers from the script output \
(counts, percentages, correlations, quantiles). Never guess, extrapolate, or invent numbers.
3. STRICT PRIORITY ASSIGNMENT:
   - "high": Dictates the validation split scheme (groups, time, stratification) or reveals data leakage / corrupted data.
   - "medium": Directly suggests an engineered feature family (group aggregation, interaction, missing flag) or model architecture.
   - "low": Minor data hygiene observations (trivial formatting, minor constants).
4. DEDUPLICATION & RESOLUTION: Do not repeat findings the team already knows. Remove questions from \
open_questions that are answered by the script output.

WORKED EXAMPLE:
- Poor Finding: statement="Cabin has slashes", evidence="seen in data", implication="clean it", priority="low"
- Grandmaster Finding: statement="Cabin string decomposes into deck/num/side where Deck B/C survival is 73% vs Deck T 20%", \
evidence="8693 rows: deck B n=779 (73.4% True), deck C n=747 (68.0% True), deck T n=5 (20.0% True)", \
implication="Extract deck and side as separate categorical features; target-encode or frequency-encode deck", priority="high"

SITUATION:
{digest}

QUESTIONS ASKED:
{questions_text}

SCRIPT OUTPUT:
{facts_text}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 2-3 sentences evaluating the script output against the questions asked.
- findings: list of objects with statement (text), evidence (exact numbers), implication (concrete modeling/feature action), \
and priority ("high"|"medium"|"low").
- open_questions: updated list of remaining open questions."""

SUMMARY_PROMPT = """\
You are the Data Analyst summarizing the completed exploration for the team and writing the Feature Engineering Brief. \
This summary is the primary bridge between data exploration and downstream modeling.

DECISION RULES (in priority order):
1. THE FEATURE ENGINEERING BRIEF:
   - Compile up to 10 prioritized feature leads into feature_ideas.
   - Every feature idea MUST reference its supporting finding_id (e.g. "finding_002").
   - Explicitly specify the feature family and expected gain (e.g., "[High Gain / Group Aggregation] Mean luxury spend by HomePlanet").
2. VALIDATION & LEAKAGE HINTS:
   - validation_hints must state the exact splitting constraint (e.g. "GroupKFold on group_id" or "StratifiedKFold on target").
   - leakage_risks must name any column with suspiciously high predictive power or train-test separation.
3. HONEST REPORTING: Base every claim solely on recorded findings. If a hypothesis was disproven, note it in issues.

WORKED EXAMPLE (Feature Lead):
- description: "[High Gain / String Decomposition] Extract Cabin deck and side; Decks B/C show >70% survival compared to 50% baseline."
- finding_id: "finding_002"

SITUATION:
{digest}

ALL FINDINGS:
{findings_text}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 2-3 sentences synthesizing data traits, major predictive signals, and primary validation risks.
- summary: 2-3 sentences summarizing the overall dataset structure, target behavior, and winning strategy.
- train_test_shift: summary of train vs test overlap, unseen categories, or adversarial drift.
- raw_columns_to_keep: list of high-signal raw string/ID/categorical columns that must remain accessible for downstream feature engineering.
- issues: list of data anomalies, outliers, and missingness issues.
- leakage_risks: list of columns or steps presenting data leakage risks.
- validation_hints: list of explicit guidance points for the ML Lead's validation split design.
- model_hints: list of model architectures that fit this data modality and target structure.
- feature_ideas: list of up to 10 objects {{"description": str, "finding_id": str}} ranking the best feature engineering leads."""


class ThinkDraft(BaseModel):
    thinking: str = ""
    open_questions: list[str] = []
    questions_to_ask: list[str] = []
    done: bool = False


class ReadDraft(BaseModel):
    thinking: str = ""
    findings: list[Finding] = []
    open_questions: list[str] = []


class SummaryDraft(BaseModel):
    thinking: str = ""
    summary: str = ""
    train_test_shift: str = ""
    raw_columns_to_keep: list[str] = []
    issues: list[str] = []
    leakage_risks: list[str] = []
    validation_hints: list[str] = []
    model_hints: list[str] = []
    feature_ideas: list[FeatureIdea] = []


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
        working_eda.train_test_shift = summary_res.train_test_shift
        working_eda.raw_columns_to_keep = summary_res.raw_columns_to_keep
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
