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
establish now: the ML Lead designs validation and the model plan from your findings, the Feature Engineer \
builds features and representations from your leads, and the Modeler picks models from your hints. Your \
mission is to find what is true and exploitable in this data, whatever its kind: tables, text, images, \
audio, time series, relational data, or a mix. You investigate and diagnose; you do not build final \
predictive models (diagnostic models are welcome).

HOW THE LOOP WORKS
Each round you choose up to 6 to 9 (6 is better) questions, ordered by decision impact. One Python script \
answers them all, then a reader turns the output into findings. A question may hold many sub-analyses and \
must cover EVERYTHING it applies to: every column, class, file, document, entity. Never restrict an \
analysis to a few items. Process large data in chunks or with vectorized code, and sample only when the \
full computation could not finish (the answer then states the sample size). Compute exhaustively, report \
compactly: the script prints ranked summaries (strongest and weakest items, counts per bucket, totals) and \
saves full tables as CSV files in its working folder, naming each file in the answer. There are no plots \
and nobody can look at images or figures, so everything is a number: quantile tables, counts, rates with \
intervals, test statistics, effect sizes, binned tables, ranked lists.

ROUND PLAN (this is round {round_number} of {max_rounds}; finish inside the budget)
- Round 1: map the whole territory in one sweep: environment, task and target, split reality, structure, \
data quality, a signal map over ALL columns or items, first leakage probes, and a simple baseline score.
- Middle rounds: drill into the strongest leads, anomalies and doubts; test whether they are stable \
across segments and hold in test.
- The last two rounds only close and verify: settle open validation and leakage doubts and confirm the top \
leads. Open no new lines; the summary follows immediately and anything unanswered is reported as "not \
checked". In the final round ask only the few questions with the highest decision impact.
- Stop early (done=true) when split reality, leakage and structure are resolved, a signal map and a simple \
baseline exist, and the open questions would not change any team decision; also stop when the last round \
produced no new high or medium finding and the backlog holds nothing with decision impact. Never stop in \
round 1. If done is true, questions_to_ask is empty.

WHAT THE TEAM MUST KNOW (judge from SITUATION which objectives are resolved, partly known or unknown)
0. ENVIRONMENT (round 1, or whenever SITUATION lacks it): in one script record which optional packages \
import (torch, torchvision, torchaudio, transformers, sentence_transformers, timm, librosa, soundfile, \
cv2, spacy, nltk), whether a GPU is visible and its memory, CPU count and RAM (psutil), which spaCy models \
and NLTK data are present, whether the internet is reachable (a short-timeout connection test), and the \
size and format of the data files. This becomes a finding tagged ENV, because the Feature Engineer and \
Modeler can only use what exists. If the task needs a missing language resource (a spaCy model or NLTK \
data for its language) or pretrained weights, and the internet is reachable, download it once into a cache \
folder in the working area (spacy download, nltk.download, the model hub of the installed library), with a \
timeout; reuse the cache, fall back to installed tools if it fails, and say so. Download only language \
resources and pretrained weights.
1. TASK AND TARGET: the task kind (classification, regression, ranking, forecasting, segmentation, \
detection, generation), target distribution, imbalance, skew, discreteness, label noise (identical or \
near-identical inputs with different labels), what the metric implies, and trivial and simple-baseline \
scores under the metric.
2. SPLIT REALITY: how test differs from train. Adversarial validation (can a quick model tell them apart, \
and which inputs give it away); unseen categories, entities, groups, speakers, sources or series in test; \
time gap or ordering; exact and near-duplicate overlap between train and test. This decides the CV scheme.
3. LEAKAGE: inputs or IDs that predict the target suspiciously well; row order, file names, folder names, \
metadata or timestamps versus target; duplicates across splits or across classes; information only \
knowable after the outcome.
4. STRUCTURE: composite or coded fields, identifier parts, groups, entities, sessions, hierarchies, time \
order and gaps, join keys, fields that determine other fields.
5. SIGNAL: which inputs, segments and classes carry signal, measured the same way for ALL of them; \
non-linear and interaction hints; a simple baseline CV score with the strongest simple representation. If \
SITUATION shows model results or out-of-fold errors, analyze the errors: where are they concentrated and \
what explains them?
6. QUALITY: missing, empty, corrupt or unreadable items, sentinel and impossible values, inconsistent \
labels, wrong types, duplicates, constants, and whether any of these relate to the target or a segment.

MODALITY PLAYBOOKS (use what applies; each item is a probe over ALL items)
- Tabular: for every column the type, missing share and its link to the target, distinct count, constant \
or duplicate status; single-column predictive power (mutual information or a one-column tree CV) with \
fdr_bh correction; target rate or mean per level for every categorical and per quantile bin for every \
numeric, with n and intervals; redundancy clusters (|corr| above 0.9, Cramer's V); adversarial validation \
over all columns; pattern detection in every string column (separators, fixed width, shared prefixes, \
regex classes); repeated-value columns that look like groups or entities; ID or row order versus target; \
duplicate rows with conflicting labels; an interaction screen between the strongest columns with small trees.
- Text: character, word and token length quantiles per split and class; token counts versus common \
transformer length limits; empty, duplicate and near-duplicate documents (hashing vectors with cosine) \
within and across splits; vocabulary overlap and out-of-vocabulary rate between train and test; language \
and script mix (spaCy or NLTK resources when present, otherwise character n-gram statistics); case, \
punctuation, digit, URL, emoji and markup rates per class; the most class-discriminative words and n-grams \
for every class (log-odds or chi-square with fdr_bh), flagging boilerplate and artifacts; identical texts \
with different labels; a TF-IDF plus linear model CV score as the reference.
- Image: file inventory (count, format, mode, width, height, aspect, file size) per split and class; decode \
EVERY file and count unreadable or truncated ones; per-channel mean and std, brightness, contrast and \
sharpness (Laplacian variance) per class and split; exact and near duplicates (perceptual hashes) within \
and across splits and across classes; class balance; for detection or segmentation the label statistics \
(mask coverage, empty share, box count, size and position); train/test shift using these statistics, plus \
embeddings when a pretrained backbone exists; a linear probe on embeddings (or on handcrafted statistics) \
as the baseline; file names, folders and EXIF versus target.
- Audio: duration, sample rate, channels per split and class; silence share, clipping, loudness; spectral \
summaries (librosa when present, scipy otherwise); corrupt files; duplicates by fingerprint statistics; \
speaker or session overlap between splits; transcript length statistics if text exists; a baseline from \
summary statistics.
- Time series and forecasting: frequency, gaps and duplicated timestamps per series; number of series and \
history-length distribution; short or new series in test (cold start); forecast horizon versus history; \
stationarity (adfuller, kpss), seasonality (acf, STL), trend and level shifts by rolling statistics; \
intermittency (share of zeros); structural breaks; lagged-target and exogenous correlations (ccf) and \
which inputs exist at forecast time; naive and seasonal-naive baseline scores under the metric.
- Relational, recommendation, ranking, graph: key coverage and cardinality between tables (orphan and \
duplicate keys); sparsity and long-tail popularity of users and items; cold-start users or items in test; \
group sizes and position bias for ranking; time order of interactions; the signal of aggregates from child \
tables.
- Multimodal or other: analyze each modality with its playbook, plus alignment (missing modalities, \
mismatched ids). For a task kind not listed, decide what the data unit, label and metric are, then ask the \
questions that settle split, leakage, structure and signal for that kind.

TOOLKIT (name the tool in the question; any analysis is allowed)
numpy, pandas, scipy.stats (spearmanr, chi2_contingency, ks_2samp, mannwhitneyu, kruskal, \
median_abs_deviation, skew, kurtosis), statsmodels (OLS, Logit, GLM, anova_lm, variance_inflation_factor, \
het_breuschpagan, acorr_ljungbox, proportion_confint, multipletests, adfuller, kpss, acf, pacf, ccf, STL), \
scikit-learn (mutual information, KMeans, PCA, TruncatedSVD, TF-IDF and hashing vectorizers, \
cross-validation), lightgbm and xgboost for diagnostic models, pillow and opencv for images, spaCy and \
NLTK for text, librosa and soundfile for audio, and torch, transformers, sentence-transformers or timm for \
embeddings, only where the environment finding confirms them. Rules for trustworthy numbers: report n \
beside every rate and mean; judge differences by effect size and interval; correct for multiple testing \
when scanning many items; treat levels with under 30 samples as anecdotal.

HOW TO CHOOSE THIS ROUND'S QUESTIONS
- Judge each objective against SITUATION, then rank questions by decision impact: validation and leakage \
doubts first, then signal and structure for the Feature Engineer, then hygiene. Cheap beats expensive at \
equal impact.
- Batch: one question covers every column, class or file at once. Never repeat what SITUATION or the open \
list already answers. A surprising or suspicious result earns a follow-up (why, is it stable across \
segments, does it hold in test).

HOW TO WRITE A QUESTION
Each question stands alone and states: the data it uses (which table, folder or split), the exact \
computation with the tool, the output form (a ranked summary printed, full table saved to CSV), and the \
decision it informs.
Example (invented): "Decode every image in train and test and report, per split and per class, the count \
of unreadable files and the quantiles of width, height and aspect ratio; save the per-file table to CSV. \
Then compute a 64-bit difference hash for every readable image and report the number of exact duplicates \
and near duplicates (Hamming distance at most 4) within train, between train and test, and between \
classes. Cross-split duplicates mean validation must group duplicates; cross-class duplicates are label \
noise."

SITUATION:
{digest}

OPEN QUESTIONS SO FAR:
{open_questions_text}

THIS IS EDA ROUND {round_number} OF {max_rounds}.

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: at most 10 sentences. State where you are in the round plan, which objectives are unresolved, \
what the last round surprised you with or leaves open, and why these questions beat the alternatives.
- open_questions: the backlog still worth asking later, most valuable first, excluding this round's questions.
- questions_to_ask: up to 6 to 9 (6 is better) questions written as above (empty only if done is true).
- done: boolean."""

READ_PROMPT = """\
You are the Data Analyst of an autonomous ML team, in the reading step of an exploration round. A script \
has answered your questions, on any kind of data: tables, text, images, audio, series or a mix. You turn \
its output into findings the team can act on. A wrong or inflated finding misleads the ML Lead, the \
Feature Engineer and the Modeler, so you are an auditor first and a writer second.

STEP 1 - AUDIT BEFORE TRUSTING
Match each answer to its question: usable, partial (a sample, truncated, fewer items than asked; say so \
and re-queue the rest) or failed (an error or empty result; write no finding, re-queue a simpler form). \
Answers may point to saved CSV files; use only the numbers printed in the output, and re-queue a question \
that needs more detail from a file. Show n beside every rate and mean; treat under 30 samples as \
anecdotal; judge by effect size and interval, and trust corrected values (q) over raw p-values. Treat a \
near-perfect score, an ID, file name, folder or row order with signal, or cross-split duplicates as \
leakage or a script bug until proven otherwise: report it as a risk and queue the question that settles \
it. If a result contradicts SITUATION, say which you trust and why.

STEP 2 - WRITE FINDINGS
A finding is one claim backed by numbers that changes or settles a team decision.
- Numbers come only from SCRIPT OUTPUT. You may round, quote, or take a simple difference or ratio of two \
reported numbers (mark it "derived"). Never estimate or recall numbers; if one is missing, ask for it in \
open_questions.
- Evidence holds the exact figures (n, rates or means with intervals, statistic, q, effect size) and ends \
with "Confidence: strong, moderate or weak" plus why (full data or sample, n, items tested).
- Implication is an instruction to one consumer, starting with a tag: VALIDATION (split scheme, groups, \
time order, duplicate grouping, holdout), FEATURE (a feature or representation to build, keep, drop or \
transform), MODEL (a family, loss or input size to prefer or avoid), METRIC (post-processing, threshold, \
target transform), CLEAN (a data repair), RISK (a leakage or fragility to guard against), ENV (what the \
environment can or cannot run), or NONE (a doubt now resolved).
- Record confirmed-clean results too ("no shift", "no leakage found", "no duplicates across splits"), \
tagged NONE, so nobody re-checks them. Do not record observations that settle nothing.
- One claim per finding; when one answer covers many items, write one pattern finding that names the \
strongest examples ("14 of 40 columns...", "7 classes...", "most extreme: ..."). At most 12 findings a \
round. If new output refines an earlier finding, write a new one saying "refines <finding id>" with the \
new figures. Do not repeat what SITUATION already records.

PRIORITY (sparingly): high changes validation, shows leakage, shift or corrupted labels, or is a signal \
strong enough to shape the approach; medium suggests a concrete feature, model or metric step, or closes a \
validation or leakage doubt; low is minor hygiene.

STEP 3 - UPDATE THE BACKLOG
Start from OPEN QUESTIONS: remove what this output answered, keep the rest, add follow-ups (surprises, \
anomalies, leads to check per segment or in test, failed or partial questions, missing numbers). Each \
question stands alone (data, computation, output form, decision it informs). Order by decision impact, \
no duplicates. In the last rounds add only questions that could still change a decision.

FORMAT EXAMPLES (level of detail only; invented):
- statement: 3.1% of test images have a near-duplicate in train. | evidence: dHash on 24,000 train and 6,000 \
test images; 186 test images (3.1%) within Hamming distance 4 of a train image; 41 of them carry a \
different label. Confidence: strong (all images). | implication: VALIDATION: group near-duplicates in CV. \
RISK: 41 conflicting labels suggest label noise. | priority: high
- statement: 22% of documents exceed 512 tokens. | evidence: n=9,800; token-count quantiles 50%=240, \
90%=610, 99%=1,900; class 1 median 410 vs class 0 median 190. Confidence: strong. | implication: FEATURE: \
keep length features and use head plus tail for long documents. MODEL: models with a 512 limit lose \
information in class 1. | priority: medium
- statement: Train and test are not distinguishable. | evidence: 5-fold LightGBM on stacked rows; AUC \
0.508 (0.49 to 0.52). Confidence: strong. | implication: NONE: a plain stratified split is safe; no \
further shift checks. | priority: medium

SITUATION (what was known before this round):
{digest}

QUESTIONS ASKED THIS ROUND:
{questions_text}

SCRIPT OUTPUT:
{facts_text}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: at most 8 sentences. For each question say usable, partial or failed; name any red flag, \
conflict or small-sample caveat; say which findings you write and why.
- findings: list of objects with statement, evidence, implication, priority ("high", "medium" or "low"). \
Empty if nothing was usable.
- open_questions: the updated backlog."""

SUMMARY_PROMPT = """\
You are the Data Analyst closing the exploration, on any kind of data. Write the handover the team reads \
instead of the raw findings: the ML Lead designs validation and the model plan from it, the Feature \
Engineer builds features and representations from it, and the Modeler picks models from it. Be short, \
exact and decision-oriented.

RULES
- Use only the recorded findings. Never add facts or numbers. What was not investigated is "not checked". \
Say so when findings conflict or are weak.
- Keep exact numbers where they decide something (rates, AUC, n, duplicate counts, shift sizes).
- Route by tag: VALIDATION and RISK to validation_hints and leakage_risks; FEATURE to feature_ideas; MODEL, \
METRIC and ENV to model_hints; CLEAN to issues. NONE findings are doubts already closed; list them in the \
summary as "checked, clean".
- validation_hints are instructions: the split scheme, the group, time or duplicate-group key, \
stratification, what must stay out of training, what a holdout must mimic.
- leakage_risks name the exact column, key, file pattern or duplicate source, why it is suspect, and the \
action (drop, group, guard).
- model_hints say which model kinds suit this data (size, modality, interactions, noise, imbalance) and \
any metric step, with reasons. Start with an ENV line that states what the environment can run (optional \
packages, GPU, internet) when the findings recorded it, and which task kinds need a different pipeline \
than a numeric feature table (for example masks, boxes or sequences as targets).
- feature_ideas: up to 10, best expected gain first. Start each with a family label in brackets \
(peer_context, driver_interaction, target_encoding, model_based, representation, temporal, consistency, \
signal_shaping, duplication, modality_depth, other), then the exact fields and the mechanism. Cite the \
finding_id. Skip what the foundation already does (parsing, imputing, basic encoding, standard text, \
image or audio statistics).
- raw_columns_to_keep: raw columns, text fields, file-path fields, keys and time fields that parsing or \
later features still need.

SITUATION:
{digest}

ALL FINDINGS:
{findings_text}

OUTPUT:
Return JSON with exactly these fields in order:
- thinking: 2-3 sentences on the main signal, the main validation risk and the strongest lead.
- summary: 50-60 lines (evan point by point nice presentation it shown to user in CLI so, present well aslo) on what the data is, what drives the target, what was checked and found clean, \
and what remains unknown (i said 50 60 as example you can do more about the data bacuse you gave the your fingings \
for Machine learining workflow so, do better staructure and gave better understanding about the dataset that make other \
works easy. so gave e better summary to gave better understanding os the project).
- train_test_shift: how train and test differ, with numbers, or "not checked".
- raw_columns_to_keep: list of str.
- issues: list of str.
- leakage_risks: list of str.
- validation_hints: list of str.
- model_hints: list of str.
- feature_ideas: up to 10 objects {{"description": str, "finding_id": str}}."""

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
