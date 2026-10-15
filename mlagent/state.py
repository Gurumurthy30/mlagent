"""The ledger: the memory of one run. It is saved as memory/ledger.json after every step.

It holds small, validated facts and file paths. File contents never go in here.
Each section has one writer agent. Ids always come from code, never from the LLM.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

SCHEMA_VERSION = 2

Priority = Literal["high", "medium", "low"]
Verdict = Literal["approve", "reject"]
ReviewFocus = Literal["features", "model", "tune", "data", "finish"]
LoopFocus = Literal["start", "features", "model", "tune", "data", "finish"]
Phase = Literal[
    "data_engineer",
    "data_analyst",
    "ml_lead",
    "experiment_loop",
    "submitter",
    "submit",
    "report",
    "done",
]
RunStatus = Literal["running", "done", "failed"]
StopReason = Literal[
    "target", "plateau", "budget", "crashes", "llm_unavailable", "agent_failed", "reviewer_finish"
]


class FileInfo(BaseModel):
    path: str
    size_mb: float = 0.0
    kind: str = ""  # table, image, audio, text, document, model, other
    role: str = ""  # train, test, sample_submission, description, other
    columns: list[str] = []  # table files only, at most 60 names


class ProjectBrief(BaseModel):
    summary: str = ""
    files: list[FileInfo] = []
    modality: str = ""  # tabular, image, text, audio, time_series, ...
    task_family: str = ""  # classification, regression, clustering, forecasting, ...
    target: str = ""
    id_column: str = ""
    primary_metric: str = ""
    higher_is_better: bool = True
    extra_metrics: list[str] = []
    deliverable_format: str = ""  # what the predictions file must look like
    has_large_files: bool = False
    open_questions: list[str] = []


class Finding(BaseModel):
    finding_id: str = ""
    statement: str
    evidence: str = ""
    implication: str = ""
    priority: Priority = "medium"


class FeatureIdea(BaseModel):
    description: str
    finding_id: str = ""


class EdaReport(BaseModel):
    summary: str = ""  # 2 to 3 sentences, written at the end of EDA
    open_questions: list[str] = []
    done: bool = False
    findings: list[Finding] = []
    issues: list[str] = []
    feature_ideas: list[FeatureIdea] = []
    leakage_risks: list[str] = []
    validation_hints: list[str] = []  # time or group structure, imbalance, ...
    model_hints: list[str] = []
    eda_loops_used: int = 0


class ExperimentPlan(BaseModel):
    validation_scheme: str = ""  # kfold, stratified_kfold, group_kfold, time_split, ...
    validation_reason: str = ""
    splits_path: str = ""
    baseline_description: str = ""
    target_score: float | None = None
    use_gpu: bool = False
    risks: list[str] = []
    playbook: str = ""  # hints for this modality and task, written by the ML Lead


class FeatureRecord(BaseModel):
    feature_id: str
    name: str
    family: str = ""  # free short label
    reason: str = ""
    finding_id: str = ""
    score_gain: float | None = None
    kept: bool = False
    feature_version: int = 0  # the features_vN.py file that first included it
    iteration: int = 0


class ExperimentRun(BaseModel):
    experiment_id: str
    focus: LoopFocus = "start"
    model_family: str = ""
    model_params: dict = {}
    summary: str = ""  # one plain sentence saying what the run tried and why
    feature_version: int = 0
    script_path: str = ""
    artifact_dir: str = ""
    validation_scores: list[float] = []  # one score per validation split
    score_mean: float | None = None
    score_std: float | None = None
    holdout_score: float | None = None
    extra_scores: dict[str, float] = {}
    model_path: str = ""
    iteration: int = 0  # the experiment-loop round that made this run
    error: str = ""  # not empty means the run crashed
    verdict: Verdict | None = None
    verdict_reasons: list[str] = []
    checks: dict = {}
    gain_vs_best: float | None = None
    counted_gain: bool = False

    @property
    def validation_predictions_path(self) -> str:
        if not self.artifact_dir:
            return ""
        return str(Path(self.artifact_dir) / "val_predictions.csv")

    @property
    def test_predictions_path(self) -> str:
        if not self.artifact_dir:
            return ""
        return str(Path(self.artifact_dir) / "test_predictions.csv")

    @property
    def crashed(self) -> bool:
        return bool(self.error)

    @property
    def approved(self) -> bool:
        return self.verdict == "approve" and not self.crashed and self.score_mean is not None

    @property
    def status(self) -> str:
        if self.crashed:
            return "crashed"
        if self.verdict is None:
            return "not evaluated"
        if self.verdict == "approve":
            return "approved"
        return "rejected"

    @property
    def headline(self) -> str:
        if self.crashed:
            last_err = self.error.strip().splitlines()[-1] if self.error.strip() else "crashed"
            return f"{self.experiment_id} | {self.model_family or 'unknown'} | {self.summary or 'no summary'} | crashed: {last_err}"
        if self.score_mean is not None:
            score_text = f"{self.score_mean:.4f}"
            if self.score_std is not None and self.score_std > 0:
                score_text += f"±{self.score_std:.4f}"
            if self.gain_vs_best is not None:
                sign = "+" if self.gain_vs_best > 0 else ""
                cg = ", counted gain" if self.counted_gain else ""
                score_text += f" ({sign}{self.gain_vs_best:.4f}{cg})"
        else:
            score_text = "no score"
        return f"{self.experiment_id} | {self.model_family or 'unknown'} | {self.summary or 'no summary'} | {score_text} | {self.status}"


class Review(BaseModel):
    iteration: int
    best_experiment_id: str = ""
    assessment: str = ""  # what improved, what did not, where the bottleneck is
    focus: ReviewFocus
    reason: str = ""
    instructions: list[str] = []  # up to 5 points for the next agent
    lesson: str = ""  # one sentence: what this round taught the team
    question: str = ""  # only when focus is "data"
    guard_note: str = ""  # filled by code when it replaced the Reviewer's choice
    reviewed_focus: LoopFocus = "start"  # the focus of the round this review is about


class FinalResult(BaseModel):
    best_experiment_id: str = ""
    ensemble_description: str = ""
    predictions_path: str = ""
    check_messages: list[str] = []
    report_path: str = ""
    headline: str = ""  # the report's "In brief" section, extracted by code


class Ledger(BaseModel):
    schema_version: int = SCHEMA_VERSION

    # Written by the CLI
    user_task: str = ""  # the user's own words, kept as typed
    user_metrics_text: str = ""
    created_at: str = ""

    # Written by the agents
    brief: ProjectBrief = ProjectBrief()
    eda: EdaReport = EdaReport()
    plan: ExperimentPlan = ExperimentPlan()
    features: list[FeatureRecord] = []
    current_feature_version: int = 0
    runs: list[ExperimentRun] = []
    reviews: list[Review] = []
    final: FinalResult = FinalResult()

    # Written by the graph
    phase: Phase = "data_engineer"
    focus: LoopFocus = "start"
    iteration: int = 0  # experiment-loop rounds finished (one Reviewer decision each)
    data_reentries_used: int = 0
    last_tuned_iteration: int = -1  # round index of the last tune round, -1 when never
    last_features_changed: bool = True
    status: RunStatus = "running"
    stop_reason: StopReason | None = None
    error_message: str = ""

    def next_experiment_id(self) -> str:
        return f"experiment_{len(self.runs) + 1:03d}"

    def approved_runs(self) -> list[ExperimentRun]:
        return [run for run in self.runs if run.approved]

    def best_approved_run(self) -> ExperimentRun | None:
        approved = self.approved_runs()
        if not approved:
            return None
        if self.brief.higher_is_better:
            return max(
                approved,
                key=lambda run: run.score_mean if run.score_mean is not None else float("-inf"),
            )
        return min(
            approved,
            key=lambda run: run.score_mean if run.score_mean is not None else float("inf"),
        )

    def trailing_crashes(self) -> int:
        count = 0
        for run in reversed(self.runs):
            if not run.crashed:
                break
            count += 1
        return count

    def runs_this_round(self) -> list[ExperimentRun]:
        return [run for run in self.runs if run.iteration == self.iteration]

    def unevaluated_runs(self) -> list[ExperimentRun]:
        return [
            run
            for run in self.runs_this_round()
            if not run.crashed and run.verdict is None
        ]

    def round_had_gain(self, n: int) -> bool:
        return any(run.counted_gain for run in self.runs if run.iteration == n)

    def round_cards(self) -> list[str]:
        cards: list[str] = []
        for review in self.reviews:
            n = review.iteration
            focus = review.reviewed_focus
            detail = ""
            if focus == "features":
                round_feats = [f for f in self.features if f.iteration == n]
                kept = [f for f in round_feats if f.kept]
                if round_feats:
                    first_str = ""
                    if kept:
                        gain_str = (
                            f" +{kept[0].score_gain:.4f}"
                            if kept[0].score_gain is not None and kept[0].score_gain >= 0
                            else (f" {kept[0].score_gain:.4f}" if kept[0].score_gain is not None else "")
                        )
                        first_str = f" ({kept[0].name}{gain_str})"
                    detail = f"kept {len(kept)} of {len(round_feats)}{first_str}"
                else:
                    detail = "no features tested"
            elif focus == "model":
                round_runs = [r for r in self.runs if r.iteration == n]
                detail = round_runs[0].summary if round_runs and round_runs[0].summary else (round_runs[0].model_family if round_runs else "model run")
            elif focus == "tune":
                round_runs = [r for r in self.runs if r.iteration == n]
                detail = round_runs[0].model_family if round_runs else "tuning"
            elif focus == "data":
                q = review.question.strip()
                detail = (q[:40] + "...") if len(q) > 40 else (q or "data question")
            else:
                detail = focus

            # Best score progression
            earlier_runs = [r for r in self.runs if r.iteration < n and r.approved]
            upto_runs = [r for r in self.runs if r.iteration <= n and r.approved]
            higher = self.brief.higher_is_better
            earlier_best = (
                (max(earlier_runs, key=lambda r: r.score_mean) if higher else min(earlier_runs, key=lambda r: r.score_mean))
                if earlier_runs
                else None
            )
            upto_best = (
                (max(upto_runs, key=lambda r: r.score_mean) if higher else min(upto_runs, key=lambda r: r.score_mean))
                if upto_runs
                else None
            )

            if earlier_best and earlier_best.score_mean is not None and upto_best and upto_best.score_mean is not None:
                score_str = f"best {earlier_best.score_mean:.4f} -> {upto_best.score_mean:.4f}"
            elif upto_best and upto_best.score_mean is not None:
                score_str = f"best {upto_best.score_mean:.4f}"
            else:
                score_str = "no score"

            cg = "yes" if self.round_had_gain(n) else "no"
            lesson = review.lesson or "none"
            cards.append(
                f"Round {n} [{focus}] {detail} | {score_str} | counted gain: {cg} | lesson: {lesson}"
            )
        return cards

    def dead_ends(self) -> list[str]:
        items: list[str] = []
        # 1. Crashed or rejected setups with first reason
        for run in self.runs:
            if run.crashed:
                first_err = run.error.strip().splitlines()[0] if run.error.strip() else "crash"
                items.append(f"{run.model_family or run.experiment_id}: crashed ({first_err})")
            elif run.verdict == "reject":
                reason = run.verdict_reasons[0] if run.verdict_reasons else "rejected"
                items.append(f"{run.model_family or run.experiment_id}: rejected ({reason})")

        # 2. Feature families with 2+ dropped ideas and none kept
        families: dict[str, list[FeatureRecord]] = {}
        for feat in self.features:
            if feat.family:
                families.setdefault(feat.family, []).append(feat)
        for fam, feats in families.items():
            dropped = [f for f in feats if not f.kept]
            kept = [f for f in feats if f.kept]
            if len(dropped) >= 2 and len(kept) == 0:
                items.append(f"feature family '{fam}': all {len(dropped)} ideas dropped")

        # 3. Lesson of rounds with no counted gain
        for review in self.reviews:
            if not self.round_had_gain(review.iteration) and review.lesson.strip():
                items.append(f"round {review.iteration} lesson: {review.lesson.strip()}")

        # Deduplicate preserving order, cap at 15
        deduped: list[str] = []
        seen = set()
        for item in items:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
                if len(deduped) >= 15:
                    break
        return deduped

    def open_feature_ideas(self) -> list[FeatureIdea]:
        existing_keys = {
            (f.finding_id, f.name.strip().lower())
            for f in self.features
            if f.finding_id
        }
        existing_reasons = {f.reason.strip().lower() for f in self.features if f.reason}
        open_ideas: list[FeatureIdea] = []
        for idea in self.eda.feature_ideas:
            key = (idea.finding_id, idea.description.strip().lower())
            desc_lower = idea.description.strip().lower()
            if key not in existing_keys and desc_lower not in existing_reasons:
                open_ideas.append(idea)
        return open_ideas
