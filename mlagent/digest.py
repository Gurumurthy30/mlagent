"""Builds the context text each agent receives from the ledger. Plain code, no LLM involved.

Context is assembled from whole items, never cut in the middle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlagent.state import Ledger, FeatureRecord

DIGEST_CHAR_LIMITS: dict[str, int] = {
    "data_engineer": 8500,
    "data_analyst": 5000,
    "ml_lead": 4500,
    "feature_engineer": 16000,
    "modeler": 7000,
    "evaluator": 5500,
    "tuner": 5500,
    "reviewer": 6000,
    "submitter": 4000,
    "coder": 8000,
}

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def shorten_text(text: str, max_chars: int) -> str:
    """Shorten long text at a sentence or word boundary with an ellipsis, never mid-word."""
    if not text or len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rstrip()
    last_punct = max(truncated.rfind(". "), truncated.rfind(".\n"))
    if last_punct > max_chars // 2:
        return truncated[: last_punct + 1] + " ..."
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space] + " ..."
    return truncated + " ..."


def pack_section(header: str, items: list[str], budget: int) -> tuple[str, int]:
    """Pack whole items into a section up to budget.

    Guarantees at least 3 items or 400 characters if items exist.
    """
    if not items:
        return "", 0
    header_line = f"{header}:" if header else ""
    packed: list[str] = []
    current_chars = len(header_line) + 1 if header_line else 0
    min_count = min(3, len(items))

    for i, item in enumerate(items):
        item_len = len(item) + 1
        if i < min_count or (current_chars < 400 and i < len(items)):
            packed.append(item)
            current_chars += item_len
        elif current_chars + item_len <= budget:
            packed.append(item)
            current_chars += item_len
        else:
            break

    if not packed:
        return "", 0
    lines = ([header_line] if header_line else []) + packed
    text = "\n".join(lines)
    return text, len(text)


def assemble_sections(sections: list[tuple[str, list[str]]], total_budget: int) -> str:
    """Assemble sections with budget flowing from unused sections to later ones."""
    active = [(h, it) for h, it in sections if it]
    if not active:
        return ""
    remaining_budget = total_budget
    built_texts: list[str] = []
    for i, (header, items) in enumerate(active):
        remaining_sections = len(active) - i
        section_budget = max(remaining_budget // remaining_sections, 400)
        text, used = pack_section(header, items, section_budget)
        if text:
            built_texts.append(text)
            remaining_budget = max(remaining_budget - used - 2, 0)
    return "\n\n".join(built_texts)


def build_task_brief_items(ledger: Ledger) -> list[str]:
    items = [
        f"USER TASK: {ledger.user_task}",
        f"USER METRICS: {ledger.user_metrics_text or 'not given'}",
    ]
    brief = ledger.brief
    if brief.summary:
        items.append(f"Brief summary: {brief.summary}")
        items.append(f"Modality: {brief.modality}, Task family: {brief.task_family}")
        items.append(f"Target: {brief.target or 'none'}, ID column: {brief.id_column or 'none'}")
        dir_text = "higher is better" if brief.higher_is_better else "lower is better"
        items.append(f"Primary metric: {brief.primary_metric} ({dir_text})")
        if brief.extra_metrics:
            items.append(f"Extra metrics: {', '.join(brief.extra_metrics)}")
        if brief.deliverable_format:
            items.append(f"Deliverable format: {brief.deliverable_format}")
        for f in brief.files[:10]:
            cols_note = ""
            if f.columns:
                col_sample = ", ".join(f.columns[:10])
                if len(f.columns) > 10:
                    col_sample += f", ... ({len(f.columns)} total)"
                cols_note = f" | columns: [{col_sample}]"
            items.append(f"- file: {f.path} ({f.kind}, {f.role}, {f.size_mb:.1f} MB){cols_note}")
        if len(brief.files) > 10:
            items.append(f"- ... and {len(brief.files) - 10} more files")
        if brief.open_questions:
            items.append(f"Open questions: {'; '.join(brief.open_questions)}")
    return items


def build_eda_items(ledger: Ledger) -> list[str]:
    eda = ledger.eda
    items: list[str] = []
    if eda.summary:
        items.append(f"EDA Summary: {eda.summary}")
    sorted_findings = sorted(eda.findings, key=lambda f: PRIORITY_ORDER.get(f.priority, 1))
    for f in sorted_findings[:12]:
        items.append(f"- {f.finding_id} [{f.priority}] {f.statement} -> {f.implication}")
    if eda.issues:
        items.append(f"Data issues: {'; '.join(eda.issues)}")
    if eda.leakage_risks:
        items.append(f"Leakage risks: {'; '.join(eda.leakage_risks)}")
    if eda.validation_hints:
        items.append(f"Validation hints: {'; '.join(eda.validation_hints)}")
    if eda.model_hints:
        items.append(f"Model hints: {'; '.join(eda.model_hints)}")
    open_ideas = ledger.open_feature_ideas()
    if open_ideas:
        idea_strs = [f"{i.description} ({i.finding_id})" if i.finding_id else i.description for i in open_ideas[:8]]
        items.append(f"Open feature ideas: {'; '.join(idea_strs)}")
    if eda.open_questions:
        items.append(f"Open questions: {'; '.join(eda.open_questions)}")
    return items


def build_plan_items(ledger: Ledger) -> list[str]:
    plan = ledger.plan
    if not plan.validation_scheme and not plan.baseline_description:
        return []
    items: list[str] = []
    val_text = plan.validation_scheme
    if plan.validation_reason:
        val_text += f" ({plan.validation_reason})"
    items.append(f"Validation scheme: {val_text}")
    if plan.splits_path:
        items.append(f"Splits path: {plan.splits_path}")
    if plan.baseline_description:
        items.append(f"Baseline: {plan.baseline_description}")
    items.append(f"Target score: {plan.target_score if plan.target_score is not None else 'none'}")
    items.append(f"Use GPU: {'yes' if plan.use_gpu else 'no'}")
    if plan.risks:
        items.append(f"Risks: {'; '.join(plan.risks)}")
    if plan.playbook:
        items.append(f"Playbook: {shorten_text(plan.playbook, 800)}")
    return items


def build_features_items(ledger: Ledger) -> list[str]:
    if not ledger.features:
        return []
    items = [f"Current features version: v{ledger.current_feature_version}"]
    families: dict[str, list[FeatureRecord]] = {}
    for f in ledger.features:
        fam = f.family or "general"
        families.setdefault(fam, []).append(f)
    for fam, feats in families.items():
        kept = [f for f in feats if f.kept]
        dropped = [f for f in feats if not f.kept]
        kept_strs = [
            f"{f.name} ({f.score_gain:+.4f})" if f.score_gain is not None else f.name
            for f in kept
        ]
        dropped_strs = [
            f"{f.name} ({f.score_gain:+.4f})" if f.score_gain is not None else f.name
            for f in dropped
        ]
        items.append(
            f"- family '{fam}': kept {len(kept)} ({', '.join(kept_strs) or 'none'}), "
            f"dropped {len(dropped)} ({', '.join(dropped_strs) or 'none'})"
        )
    return items


def build_runs_items(ledger: Ledger) -> list[str]:
    if not ledger.runs:
        return []
    items: list[str] = []
    best_run = ledger.best_approved_run()
    if best_run:
        items.append(f"BEST APPROVED RUN: {best_run.headline}")
    else:
        items.append("BEST APPROVED RUN: none yet")

    # Latest runs headlines
    items.append(f"RECENT RUNS (total {len(ledger.runs)}):")
    for run in ledger.runs[-8:]:
        items.append(f"- {run.headline}")

    # Rejections
    rejections = [r for r in ledger.runs if r.verdict == "reject"][-4:]
    if rejections:
        items.append("RECENT EVALUATOR REJECTIONS:")
        for r in rejections:
            items.append(f"- {r.experiment_id} ({r.model_family}): {'; '.join(r.verdict_reasons)}")
    return items


def build_history_items(ledger: Ledger) -> list[str]:
    items: list[str] = []
    cards = ledger.round_cards()
    if cards:
        items.append("ROUND HISTORY:")
        for c in cards[-6:]:
            items.append(f"- {c}")
    dead_ends = ledger.dead_ends()
    if dead_ends:
        items.append("KNOWN DEAD ENDS:")
        for d in dead_ends[:8]:
            items.append(f"- {d}")
    if ledger.reviews:
        latest = ledger.reviews[-1]
        items.append(
            f"LATEST REVIEW (round {latest.iteration}): chose {latest.focus}. {latest.reason}"
        )
        for instr in latest.instructions:
            items.append(f"  * {instr}")
    return items


def build_digest(agent_name: str, ledger: Ledger, char_limit: int | None = None) -> str:
    """Build the prompt digest for a specific agent."""
    limit = char_limit if char_limit is not None else DIGEST_CHAR_LIMITS.get(agent_name, 6000)

    budget_items = [
        f"BUDGET: Round {ledger.iteration} | script time limit 20 min | EDA rounds {ledger.eda.eda_loops_used}"
    ]
    machine_items = [f"COMPUTE: use_gpu={ledger.plan.use_gpu}"]

    if agent_name == "reviewer":
        # Targeted reviewer digest per section 4
        sections = [
            ("PROJECT BRIEF", build_task_brief_items(ledger)),
            ("EDA FINDINGS", build_eda_items(ledger)),
            ("FEATURES", build_features_items(ledger)),
            ("RUNS", build_runs_items(ledger)),
            ("HISTORY", build_history_items(ledger)),
            ("BUDGET", budget_items),
        ]
    elif agent_name == "data_engineer":
        sections = [("TASK", [f"USER TASK: {ledger.user_task}", f"USER METRICS: {ledger.user_metrics_text or 'not given'}"])]
    elif agent_name == "data_analyst":
        sections = [
            ("PROJECT BRIEF", build_task_brief_items(ledger)),
            ("EDA FINDINGS", build_eda_items(ledger)),
            ("BUDGET", budget_items),
        ]
    elif agent_name == "ml_lead":
        sections = [
            ("PROJECT BRIEF", build_task_brief_items(ledger)),
            ("EDA FINDINGS", build_eda_items(ledger)),
            ("BUDGET", budget_items),
            ("MACHINE", machine_items),
        ]
    elif agent_name == "feature_engineer":
        sections = [
            ("TASK & BRIEF", build_task_brief_items(ledger)),
            ("PLAN", build_plan_items(ledger)),
            ("HISTORY", build_history_items(ledger)),
            ("FEATURES", build_features_items(ledger)),
            ("EDA FINDINGS", build_eda_items(ledger)),
            ("RUNS", build_runs_items(ledger)),
            ("BUDGET", budget_items),
        ]
    elif agent_name == "modeler":
        sections = [
            ("TASK & BRIEF", build_task_brief_items(ledger)),
            ("PLAN", build_plan_items(ledger)),
            ("HISTORY", build_history_items(ledger)),
            ("RUNS", build_runs_items(ledger)),
            ("FEATURES", build_features_items(ledger)),
            ("EDA HINTS", build_eda_items(ledger)),
            ("BUDGET", budget_items),
            ("MACHINE", machine_items),
        ]
    elif agent_name == "evaluator":
        sections = [
            ("PLAN", build_plan_items(ledger)),
            ("TASK & BRIEF", build_task_brief_items(ledger)),
            ("RUNS", build_runs_items(ledger)),
            ("EDA LEAKAGE RISKS", [f"- {r}" for r in ledger.eda.leakage_risks]),
            ("HISTORY", build_history_items(ledger)),
            ("BUDGET", budget_items),
        ]
    elif agent_name == "tuner":
        sections = [
            ("PLAN", build_plan_items(ledger)),
            ("RUNS", build_runs_items(ledger)),
            ("HISTORY", build_history_items(ledger)),
            ("BUDGET", budget_items),
            ("MACHINE", machine_items),
        ]
    elif agent_name == "submitter":
        sections = [
            ("TASK & BRIEF", build_task_brief_items(ledger)),
            ("PLAN", build_plan_items(ledger)),
            ("RUNS", build_runs_items(ledger)),
            ("BUDGET", budget_items),
        ]
    else:  # coder or general
        sections = [
            ("PLAN", build_plan_items(ledger)),
            ("RUNS", build_runs_items(ledger)),
            ("BUDGET", budget_items),
            ("MACHINE", machine_items),
        ]

    return assemble_sections(sections, limit)


def build_report_facts(ledger: Ledger, char_limit: int | None = None) -> str:
    """Build the comprehensive facts document for the final report (~14,000 chars).

    Priority order:
    1. final result and best run
    2. brief
    3. EDA summary and high-priority findings
    4. kept features
    5. round cards
    6. Evaluator rejections and what they caught
    7. dead ends
    8. medium findings
    9. runs table
    """
    total_budget = char_limit if char_limit is not None else 14000
    best_run = ledger.best_approved_run()

    # 1. Final result and best run
    final_items = [
        f"Final predictions path: {ledger.final.predictions_path or 'none'}",
        f"Ensemble: {ledger.final.ensemble_description or 'single best model'}",
    ]
    if best_run:
        final_items.append(f"Best run: {best_run.headline}")
        if best_run.model_path:
            final_items.append(f"Best model path: {best_run.model_path}")
        if best_run.validation_scores:
            scores_str = ", ".join(f"{s:.4f}" for s in best_run.validation_scores)
            final_items.append(f"Validation split scores: [{scores_str}]")
        if best_run.holdout_score is not None:
            final_items.append(f"Holdout score: {best_run.holdout_score:.4f}")
        if best_run.extra_scores:
            extra_str = ", ".join(f"{k}: {v:.4f}" for k, v in best_run.extra_scores.items())
            final_items.append(f"Extra metrics: {extra_str}")
    else:
        final_items.append("Best run: none")

    # 2. Brief
    brief_items = build_task_brief_items(ledger)

    # 3. EDA summary and high-priority findings
    eda_high_items: list[str] = []
    if ledger.eda.summary:
        eda_high_items.append(f"EDA Summary: {ledger.eda.summary}")
    high_findings = [f for f in ledger.eda.findings if f.priority == "high"]
    for f in high_findings:
        eda_high_items.append(
            f"- {f.finding_id} [high] {f.statement} -> {f.implication} (evidence: {f.evidence})"
        )
    if ledger.eda.leakage_risks:
        eda_high_items.append(f"Leakage risks: {'; '.join(ledger.eda.leakage_risks)}")

    # 4. Kept features
    kept_features = [f for f in ledger.features if f.kept]
    kept_items = [
        f"- {f.name} ({f.family or 'general'}): gain {f.score_gain:+.4f if f.score_gain is not None else 'n/a'} | finding: {f.finding_id or 'none'} | reason: {f.reason}"
        for f in kept_features
    ]
    if not kept_items:
        kept_items = ["No features kept"]

    # 5. Round cards
    round_cards = ledger.round_cards()
    card_items = [f"- {c}" for c in round_cards] or ["No rounds finished"]

    # 6. Evaluator rejections
    rejections = [r for r in ledger.runs if r.verdict == "reject"]
    rej_items = [
        f"- {r.experiment_id} ({r.model_family}): rejected because {'; '.join(r.verdict_reasons)}"
        for r in rejections
    ] or ["No rejected runs"]

    # 7. Dead ends
    dead_ends = ledger.dead_ends()
    dead_items = [f"- {d}" for d in dead_ends] or ["None"]

    # 8. Medium findings
    med_findings = [f for f in ledger.eda.findings if f.priority == "medium"]
    med_items = [
        f"- {f.finding_id} [medium] {f.statement} -> {f.implication}"
        for f in med_findings
    ]

    # 9. Runs table
    table_lines = [
        "| Run | Model | Features | Score | Gain vs Best | Status | Summary |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in ledger.runs:
        score_val = f"{r.score_mean:.4f}" if r.score_mean is not None else "n/a"
        gain_val = f"{r.gain_vs_best:+.4f}" if r.gain_vs_best is not None else "-"
        summary_short = shorten_text(r.summary or "no summary", 80)
        table_lines.append(
            f"| {r.experiment_id} | {r.model_family} | v{r.feature_version} | {score_val} | {gain_val} | {r.status} | {summary_short} |"
        )

    sections = [
        ("FINAL RESULT & BEST RUN", final_items),
        ("PROJECT BRIEF", brief_items),
        ("EDA SUMMARY & HIGH-PRIORITY FINDINGS", eda_high_items),
        ("KEPT FEATURES", kept_items),
        ("ROUND CARDS", card_items),
        ("EVALUATOR REJECTIONS", rej_items),
        ("DEAD ENDS", dead_items),
        ("ADDITIONAL FINDINGS", med_items),
        ("EXPERIMENT RUNS TABLE", table_lines),
    ]

    return assemble_sections(sections, total_budget)


def build_progress_text(ledger: Ledger) -> str:
    """A short readable summary for the user, saved as memory/progress.md."""
    best_run = ledger.best_approved_run()
    best_text = best_run.headline if best_run else "none yet"
    last_review = ledger.reviews[-1] if ledger.reviews else None

    lines = [
        "# mlagent progress",
        "",
        f"- status: {ledger.status} (phase: {ledger.phase}, focus: {ledger.focus})",
        f"- experiment rounds finished: {ledger.iteration}",
        f"- runs: {len(ledger.runs)}, approved: {len(ledger.approved_runs())}",
        f"- best approved run: {best_text}",
        f"- stop reason: {ledger.stop_reason or 'not stopped yet'}",
        "",
        "## Task",
        ledger.user_task,
        "",
        f"Metrics asked for: {ledger.user_metrics_text or 'not given'}",
    ]
    if last_review:
        lines += [
            "",
            "## Latest review",
            f"Focus {last_review.focus}: {last_review.reason}",
            f"Lesson: {last_review.lesson or 'none'}",
        ]
    if ledger.runs:
        lines += ["", "## Runs"]
        for run in ledger.runs:
            lines.append(f"- {run.headline}")
    return "\n".join(lines) + "\n"
