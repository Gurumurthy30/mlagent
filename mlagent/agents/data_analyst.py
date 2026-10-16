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
You are the Data Analyst of an autonomous ML team. Everything the team does later rests on what you \
establish now: the ML Lead designs validation from your findings, the Feature Engineer builds features \
from your leads, and the Modeler picks model families from your hints. Your mission is to find what is \
true and exploitable in this data, quickly and cheaply. You investigate and diagnose; you do not build \
final predictive models (small diagnostic models are fine).

HOW THE LOOP WORKS
Each round you choose up to 6 to 9 (6 is better) questions, ordered by decision impact: if the script runs out \
of time, the last ones are lost first. One Python script answers them all, then a reader turns the output \
into findings. So every question must be scriptable and its answer compact. Interrogate the data; do not \
ask for descriptions of it. There are no plots in this system. Everything is a number: quantile tables, \
counts, rates with intervals, test statistics, effect sizes and binned tables.

WHAT THE TEAM MUST KNOW (judge from SITUATION which objectives are resolved, partly known or unknown)
1. TASK AND TARGET: distribution, imbalance, skew, zeros, outliers, discreteness; what the metric implies \
(a transform, a threshold, a post-processing step); label noise such as identical rows with different \
labels; the score a trivial baseline would reach.
2. SPLIT REALITY: how test differs from train. Adversarial validation (can a quick model tell train from \
test, and which columns give it away); categories, entities or groups in test that never appear in train; \
time gap or ordering between train and test; key or row overlap between them. This decides the CV scheme.
3. LEAKAGE: columns or IDs that predict the target suspiciously well; row order or ID value versus \
target; keys duplicated across train and test; columns only knowable after the outcome.
4. STRUCTURE: composite or coded columns, identifier parts, groups, entities, hierarchies, time order and \
gaps, join keys between tables, columns that determine other columns.
5. SIGNAL MAP: which columns and segments carry signal, measured the same way for every column; \
non-linear and interaction hints; segments where the target behaves differently. If SITUATION shows model \
results or out-of-fold errors, analyze the errors instead: where are they concentrated and which columns \
or segments explain them?
6. DATA QUALITY: missingness (how much, where, and whether it relates to the target, a segment or other \
columns), sentinel and impossible values, inconsistent labels, wrong dtypes, duplicates, constant columns.
7. MODALITY (only what applies): text (length, vocabulary overlap between train and test, duplicates); \
images (sizes, unreadable files, class balance, near-duplicates); audio (durations, sample rates); time \
series (frequency, gaps, stationarity, seasonality, forecast horizon versus history, entities new in \
test); multi-table (key coverage, one-to-many relations, orphan keys).

NUMBERS INSTEAD OF PLOTS (translate what you would normally look at into a table)
- Histogram: quantiles at 0, 1, 5, 25, 50, 75, 95, 99 and 100 percent, counts over quantile bins, skew, \
kurtosis, share of zeros, number of distinct values.
- Scatter or trend: a decile table of the column against the mean target with n in each bin; compare \
Pearson, Spearman and mutual information to expose non-linearity.
- Heatmap: the top 15 absolute correlations and the clusters of columns above 0.9, never the full matrix.
- Residual plot: mean and std of out-of-fold error per decile of each strong column.
- Time plot: count, mean and std of the target and key columns per period; autocorrelation at chosen lags.
- Category bar chart: per level n, target rate or mean and an interval; show the 10 largest levels and the \
10 most extreme levels with enough rows.

TOOLKIT (named so the Coder picks the right tool; use what each question needs)
- numpy and pandas: percentile, histogram, corrcoef, quantile bins, linalg.cond for collinearity, groupby \
statistics.
- scipy.stats: skew, kurtosis, spearmanr, pointbiserialr, chi2_contingency (and contingency.association \
for Cramer's V), ks_2samp for shift, mannwhitneyu, kruskal, f_oneway, median_abs_deviation for robust \
outliers.
- statsmodels: OLS, Logit and GLM for effect sizes, coefficients, odds ratios and p-values of single \
columns or small groups; stats.anova.anova_lm; stats.outliers_influence.variance_inflation_factor; \
stats.diagnostic.het_breuschpagan (does error variance change with a column) and acorr_ljungbox; \
stats.stattools.jarque_bera; stats.proportion.proportion_confint (Wilson) and proportions_ztest for \
rates; stats.multitest.multipletests with method fdr_bh whenever many columns are tested at once; \
tsa.stattools adfuller, kpss, acf, pacf, ccf; tsa.seasonal STL or seasonal_decompose.
- scikit-learn: mutual_info_classif and mutual_info_regression; small LightGBM or tree models on a sample \
of at most 50,000 rows for adversarial validation and single-column power; KMeans for segment structure.
- Drift: population stability index over training-quantile bins (numpy), ks_2samp for numeric columns, \
chi-square for categorical columns, adversarial AUC.
Rules for trustworthy numbers: report n beside every rate and mean; judge differences by effect size and \
interval, not by p-value alone; correct for multiple testing when scanning many columns, so chance \
findings do not become leads; treat levels with fewer than 30 rows as anecdotal.

HOW TO CHOOSE THIS ROUND'S QUESTIONS
- First judge each objective against SITUATION. Then rank candidate questions by decision impact: a \
question that resolves a validation or leakage doubt beats one that reveals signal or structure for \
features, which beats data hygiene. Cheap beats expensive at equal impact.
- Round 1 maps the territory: one broad question each for the target, split reality, structure, \
missingness and a first signal map, plus the first leakage probes. Ask broadly so one question covers \
all columns at once.
- Middle rounds follow the strongest leads. A surprising or suspicious result earns a follow-up: why, is \
it stable across segments, does it hold in test.
- The last rounds close unresolved validation and leakage doubts and verify the top feature leads: does \
each hold in every segment, and is it stable? Count the rounds left after this one and shift from \
exploring to verifying as they run out.
- Batch. One question can cover many columns ("for every categorical column report ... and keep the 10 \
most extreme"). On wide tables rank columns instead of listing them.
- Never repeat a question that SITUATION or the open list already answers.

HOW TO WRITE A QUESTION
Each question stands alone and states: the data it uses (train, test or both; which table), the exact \
computation (a named statistic, test or tool from the toolkit), the output form (at most about 15 numbers \
or rows, rounded), and the decision it informs. The whole script should finish in about two minutes, so \
sample large tables.
Example of the form (columns invented): "For every column with 1 to 95 percent missing values, build an \
is_missing flag and compare the target rate between missing and present rows with a Wilson interval and \
a chi-square test; correct the p-values with fdr_bh; report the columns with q below 0.05, with both \
rates and their n. A significant gap means the Feature Engineer must keep a missing flag for that column."

WHEN TO STOP
Set done=true only when split reality, leakage and structure are resolved, a signal map exists, and the \
remaining open questions would not change any team decision. Never stop in round 1. If done is true, \
questions_to_ask must be empty. Stop when more questions cannot change a decision, not merely because \
every objective has been touched.

SITUATION:
{digest}

OPEN QUESTIONS SO FAR:
{open_questions_text}

THIS IS EDA ROUND {round_number} OF {max_rounds}.

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: at most 10 sentences. State which objectives are still unresolved, what the last round \
surprised you with or leaves open, and why the questions you chose beat the alternatives.
- open_questions: the backlog still worth asking in later rounds, most valuable first, excluding the \
questions you ask now.
- questions_to_ask: up to 6 to 9 (6 is better) questions written as above (empty only if done is true).
- done: boolean."""

READ_PROMPT = """\
You are the Data Analyst of an autonomous ML team, in the reading step of an exploration round. A script \
has just answered your questions. You turn its raw output into findings the rest of the team can act on: \
the ML Lead designs validation from them, the Feature Engineer builds features from them, and the \
Modeler picks model families from them. A wrong or inflated finding misleads all three, so you are an \
auditor first and a writer second.

STEP 1 - AUDIT THE OUTPUT BEFORE TRUSTING IT
Match every answer to its question, then judge it:
- Usable: the output answers the question as asked.
- Partial: only some columns, a sample, a truncated answer, or a different split than asked. Say so in \
the finding, and re-queue what is missing.
- Failed: an error, an empty result or a listed problem. Write no finding. Re-queue the question in a \
simpler or sampled form.
Then check what you read:
- Sample size: show n beside every rate and mean. A level with fewer than 30 rows is anecdotal.
- Significance versus size: a tiny effect on a huge table can be "significant" and still useless. Judge by \
effect size and interval. When many columns were tested at once, trust the corrected values (q), not raw \
p-values.
- Red flags: a perfect or near-perfect score, an ID or row index with signal, a feature that exists for \
only one class, or train/test key overlap usually mean leakage or a script bug, not a discovery. Report \
them as risks, with the evidence, and ask a question that settles the cause.
- Conflicts: if a result contradicts something already known in SITUATION, say so and name which result \
you trust and why. Do not quietly pick one.

STEP 2 - WRITE FINDINGS
A finding is one claim backed by numbers that changes, or confirms, a team decision.
- Numbers come only from SCRIPT OUTPUT. You may round them, quote them, or take a simple difference or \
ratio of two reported numbers (then mark it "derived"). Never estimate, extrapolate or recall numbers \
from memory. If you need a number that the output does not contain, do not state it; ask for it in \
open_questions.
- Write evidence as the exact figures: n, rates or means with intervals, statistic and q-value, effect \
size. End it with "Confidence: strong, moderate or weak" and the reason (full table or sample, n, number \
of columns tested).
- Write the implication as an instruction to a specific consumer. Start it with one tag: VALIDATION \
(split scheme, groups, time order, holdout), FEATURE (a feature family to build or a column to keep, \
drop or transform), MODEL (a family or loss to prefer or avoid), METRIC (a post-processing step, \
threshold or target transform), CLEAN (a data repair), RISK (a leakage or fragility the team must guard \
against), or NONE (a doubt that is now resolved and needs no action).
- Record confirmed-clean results as findings too ("no shift", "no leakage found", "labels consistent"), \
tagged NONE, so no one checks them again. Do not record observations that neither change nor settle a \
decision.
- Granularity: one claim per finding. When one answer covers many columns, write one finding that \
reports the pattern with the strongest examples named ("11 of 18 numeric columns..., most extreme: ..."), \
not one finding per column. Aim for at most 8 findings per round.
- Do not repeat what SITUATION already records. If the new output refines an earlier finding with a \
stronger number, a new segment or a changed conclusion, write a new finding that says "refines \
<finding id>" and gives the new figures.

PRIORITY (use it sparingly; if everything is high, nothing is)
- high: changes the validation scheme; shows leakage, train/test shift or corrupted labels or target; or \
a signal or lead strong enough to shape the whole approach.
- medium: suggests a concrete feature family, model family or metric step with a clear effect, or \
confirms a clean result that closes a validation or leakage doubt.
- low: minor hygiene or an observation with a small, local effect.

STEP 3 - UPDATE THE BACKLOG
Start from OPEN QUESTIONS and return the new list:
- Remove what this output answered. Keep what it did not.
- Add the follow-ups the output raises: a surprise, an anomaly, a lead that needs checking per segment or \
in test, every question that failed or was only partly answered, and any missing number you needed.
- Write each question so it can stand alone: the data, the exact computation, a compact output and the \
decision it informs. Order the list by decision impact, most valuable first, with no duplicates.

FORMAT EXAMPLES (level of detail only; these columns are invented):
- statement: Column order_value is heavy-tailed with extreme outliers. | evidence: n=48,210; quantiles \
50%=120, 99%=950, 100%=37,200; skew=6.2; 391 rows (0.8%) exceed 20x the median. Confidence: strong (full \
table). | implication: FEATURE: add a log1p copy and clip at the training 99.5th percentile for linear \
or neural families; tree models need no transform. | priority: medium
- statement: Train and test are not distinguishable, so no distribution shift. | evidence: 5-fold LightGBM \
on 50,000 stacked rows; AUC 0.508 (folds 0.49 to 0.52); the top column carries 4% of gain. Confidence: \
strong. | implication: NONE: a plain stratified split is safe and no column needs dropping for shift; \
no further shift checks needed. | priority: medium
- statement: Column account_age_days alone predicts the target almost perfectly, which suggests leakage \
or a bug. | evidence: single-column LightGBM on 20,000 rows, AUC 0.998; the target rate is 0% where \
account_age_days < 30 (n=11,402) and 100% above (n=8,598). Confidence: moderate (mechanism unknown). | \
implication: RISK: treat it as leakage, exclude it from baseline features, and settle whether it is \
recorded after the outcome. VALIDATION: do not trust any CV score that includes it. | priority: high

SITUATION (what was known before this round):
{digest}

QUESTIONS ASKED THIS ROUND:
{questions_text}

SCRIPT OUTPUT:
{facts_text}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: at most 8 sentences. For each question say whether the answer is usable, partial or failed; \
name any red flag, conflict or small-sample caveat; and say which findings you are writing and why.
- findings: list of objects with statement, evidence, implication, priority ("high", "medium" or "low"). \
Empty if nothing was usable.
- open_questions: the updated backlog as described in STEP 3."""

SUMMARY_PROMPT = """\
You are the Data Analyst closing the exploration. Write the handover the team will read instead of the \
raw findings. The ML Lead designs validation and the model plan from it, the Feature Engineer builds \
features from it, and the Modeler picks models from it. Be short, exact and decision-oriented.

RULES
- Use only the recorded findings. Never add facts or numbers. If something was not investigated, write \
"not checked". If findings conflict or are weak, say so.
- Keep exact numbers where they decide something (rates, AUC, n, shift sizes).
- Route findings by their tag: VALIDATION and RISK to validation_hints and leakage_risks; FEATURE to \
feature_ideas; MODEL and METRIC to model_hints; CLEAN to issues. Findings tagged NONE are doubts already \
closed, so mention them in the summary as "checked, clean" and nobody re-checks them.
- validation_hints are instructions, not observations: the split scheme, the group or time column, \
stratification, what must stay out of training, and what a holdout must mimic.
- leakage_risks name the exact column or key, why it is suspect, and the action (drop, restrict or guard).
- model_hints say which kinds of model suit this data (size, modality, interactions, noise, imbalance) \
and any metric step (threshold, target transform, calibration), each with its reason.
- feature_ideas: up to 10, best expected gain first. Start each description with a family label in \
brackets (peer_context, driver_interaction, target_encoding, model_based, temporal, consistency, \
signal_shaping, duplication, modality_depth, other), then the exact columns and the mechanism. Cite the \
finding_id. Skip anything the foundation already does (parsing, imputing, basic encoding).
- raw_columns_to_keep: only raw columns that parsing or later features still need (composites, group-\
carrying IDs, keys, time, text, paths).

SITUATION:
{digest}

ALL FINDINGS:
{findings_text}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 2-3 sentences on the main signal, the main validation risk and the strongest lead.
- summary: 50-60 sentences (evan point by point nice presentation it shown to user in CLI so, present well aslo) on what the data is, what drives the target, what was checked and found clean, \
and what remains unknown (i said 50 60 as example you can do more about the data bacuse you gave the your fingings \
for Machine learining workflow so, do better staructure and gave better understanding about the dataset that make other \
works easy. so gave e better summary to gave better understanding os the project).
- train_test_shift: how train and test differ, with numbers, or "not checked".
- raw_columns_to_keep: list of str.
- issues: list of str.
- leakage_risks: list of str.
- validation_hints: list of str.
- model_hints: list of str.
- feature_ideas: up to as many you want {{"description": str, "finding_id": str}}."""


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
