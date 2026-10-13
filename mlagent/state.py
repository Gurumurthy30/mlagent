"""The ledger: the memory of one run. It is saved as memory/ledger.json after every step.

It holds small, validated facts and file paths. File contents never go in here.
Each section has one writer agent (see the plan). Ids always come from code, never from the LLM.
"""

from typing import Literal

from pydantic import BaseModel

SCHEMA_VERSION = 1

ChecklistStatus = Literal["open", "covered", "not_applicable"]
Priority = Literal["high", "medium", "low"]
Verdict = Literal["approve", "reject"]
ReviewFocus = Literal["features", "model", "tune", "data", "finish"]
LoopFocus = Literal["start", "features", "model", "tune", "data", "finish"]
Phase = Literal["data_engineer", "data_analyst", "ml_lead", "experiment_loop", "submitter", "done"]
RunStatus = Literal["running", "done", "failed"]
StopReason = Literal[
    "target", "plateau", "budget", "crashes", "llm_unavailable", "agent_failed", "reviewer_finish"
]


class FileInfo(BaseModel):
    path: str
    size_mb: float = 0.0
    kind: str = ""  # table, image, audio, text, document, model, other
    role: str = ""  # train, test, sample_submission, description, other


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
    finding_id: str
    statement: str
    evidence: str = ""
    implication: str = ""
    priority: Priority = "medium"


class FeatureIdea(BaseModel):
    description: str
    finding_id: str = ""


class EdaReport(BaseModel):
    checklist: dict[str, ChecklistStatus] = {}  # checklist item -> coverage state
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
    primary_metric: str = ""
    higher_is_better: bool = True
    extra_metrics: list[str] = []
    baseline_description: str = ""
    target_score: float | None = None
    use_gpu: bool = False
    risks: list[str] = []
    playbook: str = ""  # hints for this modality and task, written by the ML Lead


class FeatureRecord(BaseModel):
    feature_id: str
    name: str
    reason: str = ""
    finding_id: str = ""
    score_gain: float | None = None
    kept: bool = False
    feature_version: int = 0  # the features_vN.py file that first included it


class ExperimentRun(BaseModel):
    experiment_id: str
    focus: LoopFocus = "start"
    model_family: str = ""
    model_params: dict = {}
    description: str = ""
    feature_version: int = 0
    script_path: str = ""
    artifact_dir: str = ""
    validation_scores: list[float] = []  # one score per validation split
    score_mean: float | None = None
    score_std: float | None = None
    holdout_score: float | None = None
    extra_scores: dict[str, float] = {}
    validation_predictions_path: str = ""
    test_predictions_path: str = ""
    model_path: str = ""
    iteration: int = 0  # the experiment-loop round that made this run
    error: str = ""  # not empty means the run crashed

    @property
    def crashed(self) -> bool:
        return bool(self.error)


class Evaluation(BaseModel):
    experiment_id: str
    verdict: Verdict
    reasons: list[str] = []
    checks: dict = {}


class Review(BaseModel):
    iteration: int
    best_experiment_id: str = ""
    assessment: str = ""  # what improved, what did not, where the bottleneck is
    focus: ReviewFocus
    reason: str = ""
    instructions: list[str] = []  # up to 5 points for the next agent
    question: str = ""  # only when focus is "data"
    guard_note: str = ""  # filled by code when it replaced the Reviewer's choice
    reviewed_focus: LoopFocus = "start"  # the focus of the round this review is about
    counted_gain: bool = False  # did that round produce a counted gain over the best run


class FinalResult(BaseModel):
    best_experiment_id: str = ""
    ensemble_description: str = ""
    predictions_path: str = ""
    check_messages: list[str] = []
    report_path: str = ""


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
    evaluations: list[Evaluation] = []
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

    def next_feature_id(self) -> str:
        return f"feature_{len(self.features) + 1:03d}"

    def next_finding_id(self) -> str:
        return f"finding_{len(self.eda.findings) + 1:03d}"

    def find_run(self, experiment_id: str) -> ExperimentRun | None:
        for run in self.runs:
            if run.experiment_id == experiment_id:
                return run
        return None

    def find_evaluation(self, experiment_id: str) -> Evaluation | None:
        for evaluation in reversed(self.evaluations):
            if evaluation.experiment_id == experiment_id:
                return evaluation
        return None

    def approved_runs(self) -> list[ExperimentRun]:
        approved_ids = {e.experiment_id for e in self.evaluations if e.verdict == "approve"}
        return [
            run
            for run in self.runs
            if run.experiment_id in approved_ids and not run.crashed and run.score_mean is not None
        ]

    def best_approved_run(self) -> ExperimentRun | None:
        approved = self.approved_runs()
        if not approved:
            return None
        if self.plan.higher_is_better:
            return max(approved, key=lambda run: run.score_mean)
        return min(approved, key=lambda run: run.score_mean)

    def recent_crashes(self, count: int) -> list[ExperimentRun]:
        crashed_runs = [run for run in self.runs if run.crashed]
        return crashed_runs[-count:]
