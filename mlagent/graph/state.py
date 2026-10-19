"""Master state schemas and Champion-vs-Challenger tracking for mlagent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel as _PydanticBaseModel
from pydantic import Field


# Base model with compatibility helpers
class BaseModel(_PydanticBaseModel):
    """Pydantic v2 compatible BaseModel wrapper."""

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().model_json_schema(*args, **kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str | bytes, *args: Any, **kwargs: Any) -> Any:
        return super().model_validate_json(json_data, *args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        return super().model_dump_json(*args, **kwargs)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().model_dump(*args, **kwargs)


# State schema version identifier
SCHEMA_VERSION = 3

# Data modality types
Modality = Literal[
    "tabular",
    "text",
    "image",
    "audio",
    "time_series",
    "multimodal",
]

# Supported machine learning task families
TaskFamily = Literal[
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
    "clustering",
    "time_series_forecasting",
    "ranking",
    "anomaly_detection",
]

# Supported validation partitioning schemes
ValidationScheme = Literal[
    "stratified_kfold",
    "kfold",
    "group_kfold",
    "stratified_group_kfold",
    "time_series_split",
    "purged_time_series_split",
    "unsupervised_stability_split",
]

# Priority level for findings and hypotheses
Priority = Literal["high", "medium", "low"]

# Evaluation verdict
Verdict = Literal["approve", "reject"]

# Strategic routes dispatched by ML Lead
StrategistRoute = Literal[
    "fe_and_model",
    "model_only",
    "tune_champion",
    "revisit_eda",
    "finalize",
]

# Model complexity tier for progressive exploration
ComplexityTier = Literal[
    "tier_1_simple_baseline",
    "tier_2_ensemble_or_regularized",
    "tier_3_gradient_boosting_or_nn",
]

# Generalization audit classification
OverfittingStatus = Literal[
    "healthy",
    "mild_overfit",
    "severe_overfit",
    "underfitting",
]

# Active lifecycle phase
Phase = Literal[
    "data_engineer",
    "eda_analyst",
    "ml_lead",
    "feature_engineer",
    "modeler",
    "tuner",
    "evaluator",
    "finalizer",
    "summarizer",
    "done",
]

# Overall project run status
RunStatus = Literal["running", "done", "failed"]

# Reason for termination
StopReason = Literal[
    "target_reached",
    "hypothesis_plateau",
    "max_experiment_loops_reached",
    "consecutive_crashes",
    "llm_unavailable",
    "agent_failed",
    "strategist_finalize",
]


# Deterministic single-column profile statistics
class ColumnProfile(BaseModel):
    """Basic column statistics computed deterministically before calling LLM."""

    name: str
    dtype: str
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    sample_values: list[str] = Field(default_factory=list)
    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None
    std_val: float | None = None


# Deterministic single-file metadata profile
class FileProfile(BaseModel):
    """File inventory metadata extracted directly by Python profiler."""

    path: str
    size_mb: float = 0.0
    kind: Literal["table", "image_dir", "audio_dir", "text_corpus", "document", "other"] = "table"
    inferred_role: Literal[
        "train", "test", "sample_submission", "description", "auxiliary", "other"
    ] = "other"
    row_count: int = 0
    col_count: int = 0
    duplicate_row_count: int = 0
    columns: list[ColumnProfile] = Field(default_factory=list)


# Complete folder scan profile passed to Data Engineer
class DatasetProfile(BaseModel):
    """Deterministic directory scan summary."""

    project_folder: str
    files: list[FileProfile] = Field(default_factory=list)
    train_minus_test_columns: list[str] = Field(default_factory=list)
    shared_columns: list[str] = Field(default_factory=list)
    readme_excerpt: str = ""


# Node 1: Data Engineer output schema (Stage 1-2 Problem Definition & Scoping)
class ProjectScope(BaseModel):
    """Output schema for Data Engineer problem framing and scoping."""

    thinking: str = Field(description="Analysis of user requirements and dataset files.")
    cli_summary: str = Field(description="Crisp 1-3 line summary formatted for terminal display.")
    enhanced_problem_statement: str = Field(
        description="Actionable formulation of what to predict and optimize."
    )
    modality: Modality
    task_family: TaskFamily
    target_column: str = Field(
        default="", description="Target column in training data, or empty if unsupervised."
    )
    id_columns: list[str] = Field(default_factory=list)
    group_columns: list[str] = Field(
        default_factory=list, description="Entity or customer group identifier columns."
    )
    time_columns: list[str] = Field(
        default_factory=list, description="Timestamp or temporal ordering columns."
    )
    text_columns: list[str] = Field(default_factory=list)
    primary_metric: str = Field(description="Primary optimization metric.")
    higher_is_better: bool = True
    secondary_metrics: list[str] = Field(
        default_factory=list, description="Diagnostic companion metrics."
    )
    deliverable_format: str = Field(
        description="Expected predictions output format and column names."
    )
    data_sufficiency_and_risks: list[str] = Field(default_factory=list)
    initial_eda_questions: list[str] = Field(
        default_factory=list, description="Top questions to seed EDA."
    )


# Locked project brief stored inside Ledger
class ProjectBrief(BaseModel):
    """Locked project scope stored in Ledger.brief."""

    profile: DatasetProfile | None = None
    enhanced_problem_statement: str = ""
    modality: Modality = "tabular"
    task_family: TaskFamily = "binary_classification"
    target_column: str = ""
    id_columns: list[str] = Field(default_factory=list)
    group_columns: list[str] = Field(default_factory=list)
    time_columns: list[str] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    primary_metric: str = "accuracy"
    higher_is_better: bool = True
    secondary_metrics: list[str] = Field(default_factory=list)
    deliverable_format: str = ""
    data_sufficiency_and_risks: list[str] = Field(default_factory=list)


# Node 2a: EDA plan step schema
class EdaQuestionPlan(BaseModel):
    """Output schema for EDA Analyst question planning."""

    thinking: str = Field(description="Assessment of current findings and open data questions.")
    cli_summary: str = Field(description="Crisp 1-3 line summary formatted for terminal display.")
    questions_to_test: list[str] = Field(
        description="1 to 5 quantitative questions for this EDA script."
    )


# Single quantitative finding draft
class EdaFindingDraft(BaseModel):
    """Quantitative pattern or anomaly observed during EDA."""

    statement: str = Field(description="What pattern or anomaly was found.")
    quantitative_evidence: str = Field(
        description="Exact numbers, distributions, or statistical values."
    )
    modeling_implication: str = Field(
        description="Impact on preprocessing, split strategy, or feature design."
    )
    priority: Priority = "medium"


# Stored finding with unique identifier
class EdaFinding(EdaFindingDraft):
    """Stored finding with deterministic ID."""

    finding_id: str = ""
    eda_round: int = 0


# Node 2b: EDA read step schema
class EdaFindingsSummary(BaseModel):
    """Output schema for EDA Analyst result interpretation."""

    thinking: str = Field(description="Interpretation of script outputs.")
    cli_summary: str = Field(description="Crisp 1-3 line summary formatted for terminal display.")
    new_findings: list[EdaFindingDraft] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    ready_to_build_blueprint: bool = Field(
        description="True if key data distributions and risks are resolved."
    )


# Per-column preprocessing specification
class ColumnSpec(BaseModel):
    """Stage 3 data cleaning and encoding rule for a column."""

    column_name: str
    dtype_role: Literal[
        "numeric_continuous",
        "numeric_discrete",
        "categorical_low_cardinality",
        "categorical_high_cardinality",
        "boolean",
        "datetime",
        "free_text",
        "identifier_or_drop",
    ]
    missing_strategy: Literal[
        "none",
        "drop_rows",
        "impute_mean",
        "impute_median",
        "impute_mode",
        "impute_domain_constant",
        "group_conditional_impute",
        "missing_indicator_plus_median",
    ] = "none"
    missing_constant_value: str | None = None
    format_normalization: str = Field(
        default="", description="String or date standardization instruction."
    )
    outlier_strategy: Literal[
        "none",
        "keep_as_genuine_signal",
        "clip_iqr_or_percentile",
        "log1p_or_boxcox_transform",
        "drop_corrupted_rows",
    ] = "none"
    encoding_strategy: Literal[
        "none",
        "binary_0_1",
        "one_hot_encoding",
        "ordinal_encoding",
        "oof_target_encoding",
        "frequency_encoding",
        "tfidf_or_embedding",
    ] = "none"
    scaling_strategy: Literal[
        "none",
        "standard_scaler",
        "robust_scaler",
        "minmax_scaler",
    ] = "none"
    drop_column: bool = False
    drop_reason: str = ""


# Multi-column domain consistency rule
class ConsistencyRule(BaseModel):
    """Multi-column domain consistency or sanity rule."""

    rule_name: str
    condition_description: str
    action_description: str
    supporting_finding_id: str = ""


# Candidate feature idea proposal
class FeatureIdea(BaseModel):
    """Candidate feature creation or pruning idea."""

    idea_id: str = ""
    name: str = Field(description="Short snake_case feature identifier.")
    category: Literal[
        "date_time_decomposition",
        "domain_ratio_or_difference",
        "group_aggregation",
        "nonlinear_transform",
        "polynomial_or_interaction",
        "text_statistic_or_embedding",
        "redundancy_pruning",
    ]
    input_columns: list[str] = Field(default_factory=list)
    transformation_logic: str = Field(description="Exact mathematical or extraction formula.")
    supporting_finding_id: str = ""
    priority: Priority = "medium"
    status: Literal["untested", "kept", "dropped"] = "untested"
    tested_in_iteration: int | None = None
    measured_cv_gain: float | None = None


# Node 2c: Final blueprint emitted by EDA Analyst
class EdaBlueprint(BaseModel):
    """Output schema for EDA Analyst blueprint handoff."""

    thinking: str = Field(
        description="Synthesis of dataset profile, cleaning needs, and risk areas."
    )
    cli_summary: str = Field(description="Crisp 1-3 line summary formatted for terminal display.")
    concise_executive_summary: str = Field(
        description="Overview of dataset structure under 600 characters."
    )
    duplicate_handling_policy: Literal[
        "none", "drop_exact_duplicates", "deduplicate_by_id_keep_latest"
    ] = "none"
    target_distribution_and_imbalance: str = Field(
        description="Target distribution, class skew, or balance summary."
    )
    recommended_validation_scheme: ValidationScheme
    recommended_split_column: str = Field(
        default="", description="Group or time column for splitting."
    )
    redundant_or_collinear_columns_to_drop: list[str] = Field(default_factory=list)
    leakage_risks_to_guard: list[str] = Field(default_factory=list)
    column_preprocessing_plan: list[ColumnSpec] = Field(default_factory=list)
    domain_consistency_rules: list[ConsistencyRule] = Field(default_factory=list)
    feature_hypotheses_backlog: list[FeatureIdea] = Field(default_factory=list)


# Stored EDA state in Ledger
class EdaState(BaseModel):
    """Complete EDA and Preprocessing state in Ledger.eda."""

    eda_loops_used: int = 0
    done: bool = False
    open_questions: list[str] = Field(default_factory=list)
    findings: list[EdaFinding] = Field(default_factory=list)
    concise_executive_summary: str = ""
    duplicate_handling_policy: Literal[
        "none", "drop_exact_duplicates", "deduplicate_by_id_keep_latest"
    ] = "none"
    target_distribution_and_imbalance: str = ""
    recommended_validation_scheme: ValidationScheme = "stratified_kfold"
    recommended_split_column: str = ""
    redundant_or_collinear_columns_to_drop: list[str] = Field(default_factory=list)
    leakage_risks_to_guard: list[str] = Field(default_factory=list)
    column_preprocessing_plan: list[ColumnSpec] = Field(default_factory=list)
    domain_consistency_rules: list[ConsistencyRule] = Field(default_factory=list)
    feature_hypotheses: list[FeatureIdea] = Field(default_factory=list)


# Node 3a: Round 0 validation and baseline plan
class ValidationPlan(BaseModel):
    """Output schema for ML Lead validation locking and baseline configuration."""

    thinking: str = Field(
        description="Rationale for split scheme, holdout partition, and baseline model."
    )
    cli_summary: str = Field(description="Crisp 1-3 line summary formatted for terminal display.")
    validation_scheme: ValidationScheme
    n_splits: int = Field(default=5, ge=3, le=10)
    holdout_fraction: float = Field(default=0.15, ge=0.0, le=0.30)
    split_column: str = ""
    time_column: str = ""
    gap: int = Field(
        default=0, ge=0, description="Temporal gap between train and validation folds."
    )
    embargo: int = Field(default=0, ge=0, description="Embargo samples following validation fold.")
    forecast_horizon: int = Field(
        default=0, ge=0, description="Number of future periods evaluated."
    )
    planned_model_progression: list[str] = Field(description="Simple-to-complex algorithm roadmap.")
    baseline_hypothesis_statement: str = Field(
        description="Baseline hypothesis to verify pipeline."
    )
    baseline_model_family: str = Field(description="Simple Tier-1 starting algorithm.")
    target_post_processing: str = Field(default="none")


# Node 3b: Iterative hypothesis routing plan
class HypothesisPlan(BaseModel):
    """Output schema for ML Lead hypothesis strategist in round >= 1."""

    thinking: str = Field(
        description="Evaluation of current Champion vs Challenger and next focus selection."
    )
    cli_summary: str = Field(description="Crisp 1-3 line summary formatted for terminal display.")
    previous_hypothesis_verdict: Literal[
        "validated_new_champion",
        "refuted_regression",
        "overfit_rejected",
        "script_error",
        "baseline_established",
    ]
    bottleneck_diagnosis: str = Field(description="Current primary model or data bottleneck.")
    lesson_learned: str = Field(description="Key fact established in the previous round.")
    next_route: StrategistRoute
    hypothesis_statement: str = Field(description="Hypothesis being evaluated in this iteration.")
    selected_feature_idea_ids: list[str] = Field(default_factory=list)
    custom_feature_instructions: list[str] = Field(default_factory=list)
    target_model_family: str = Field(default="")
    model_complexity_tier: ComplexityTier = "tier_2_ensemble_or_regularized"
    model_instructions: list[str] = Field(default_factory=list)
    eda_revisit_question: str = Field(default="")


# Stored hypothesis execution record
class HypothesisRecord(BaseModel):
    """Historical record of an executed hypothesis."""

    hypothesis_id: str
    iteration: int
    route: StrategistRoute
    base_feature_version: int
    hypothesis_statement: str
    selected_feature_idea_ids: list[str] = Field(default_factory=list)
    custom_feature_instructions: list[str] = Field(default_factory=list)
    target_model_family: str = ""
    model_complexity_tier: ComplexityTier = "tier_1_simple_baseline"
    model_instructions: list[str] = Field(default_factory=list)
    eda_revisit_question: str = ""
    bottleneck_diagnosis: str = ""
    lesson_learned: str = ""
    outcome: Literal[
        "pending",
        "validated_new_champion",
        "refuted_regression",
        "overfit_rejected",
        "script_error",
    ] = "pending"


# Single fold-isolated feature operation specification
class FeatureOp(BaseModel):
    """Executable feature transformation or selection operation."""

    op_name: str
    operation_type: Literal[
        "stage3_preprocessing",
        "date_decomposition",
        "domain_interaction_or_ratio",
        "group_aggregation",
        "nonlinear_or_frequency",
        "text_representation",
        "prune_redundant_or_noisy",
    ]
    input_columns: list[str] = Field(default_factory=list)
    output_or_pruned_columns: list[str] = Field(default_factory=list)
    implementation_rule: str = Field(
        description="Formula or rule fit strictly inside training fold."
    )
    linked_idea_id: str = ""


# Node 4: Feature Engineer plan output
class FeaturePlan(BaseModel):
    """Output schema for Feature Engineer."""

    thinking: str = Field(
        description="Explanation of feature engineering and leak-free fold isolation."
    )
    cli_summary: str = Field(description="Crisp 1-3 line summary formatted for terminal display.")
    parent_feature_version: int = Field(description="Champion feature version being inherited.")
    operations_to_apply: list[FeatureOp] = Field(default_factory=list)
    columns_to_drop_or_prune: list[str] = Field(default_factory=list)
    requires_numeric_scaling: bool = False
    fold_isolation_check: str = Field(description="Confirmation of fold-isolated fitting.")


# Stored feature version manifest
class FeatureVersionRecord(BaseModel):
    """Stored manifest of a generated feature version."""

    feature_version: int
    parent_feature_version: int = 0
    iteration: int = 0
    hypothesis_id: str = ""
    script_path: str = ""
    output_feature_columns: list[str] = Field(default_factory=list)
    added_columns: list[str] = Field(default_factory=list)
    pruned_columns: list[str] = Field(default_factory=list)
    operations: list[FeatureOp] = Field(default_factory=list)
    screening_cv_score: float | None = None
    verified_improvement: bool = False


# Node 5: Modeler training plan output
class ModelPlan(BaseModel):
    """Output schema for Modeler."""

    thinking: str = Field(
        description="Selection of algorithm, hyperparameter defaults, and post-processing."
    )
    cli_summary: str = Field(description="Crisp 1-3 line summary formatted for terminal display.")
    model_family: str
    complexity_tier: ComplexityTier
    feature_version_to_use: int
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    imbalance_or_loss_strategy: str = "default"
    post_processing_strategy: str = "none"


# Hyperparameter search dimension
class HyperparameterRange(BaseModel):
    """Search bounds for hyperparameter tuning."""

    param_name: str
    param_type: Literal["float_log", "float_uniform", "int", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    choices: list[str | int | float | bool] = Field(default_factory=list)


# Node 6: Hyperparameter Tuner output
class TuningPlan(BaseModel):
    """Output schema for Hyperparameter Tuner."""

    thinking: str = Field(description="Diagnostic tuning search space tailored to train-vs-CV gap.")
    cli_summary: str = Field(description="Crisp 1-3 line summary formatted for terminal display.")
    source_experiment_id: str
    model_family: str
    feature_version: int
    search_method: Literal["optuna_tpe", "randomized_search", "grid_search"] = "optuna_tpe"
    n_trials: int = Field(default=25, ge=5, le=100)
    search_space: list[HyperparameterRange] = Field(default_factory=list)
    early_stopping_rounds: int = 50


# Priority 2: Task-Specific Diagnostic Reports
class ClassificationReport(BaseModel):
    """Classification diagnostic metrics."""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float | None = None
    pr_auc: float | None = None
    per_class_f1: dict[str, float] = Field(default_factory=dict)
    dominant_error_mode: str = ""


class RegressionReport(BaseModel):
    """Regression diagnostic metrics."""

    mae: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    mape: float | None = None
    residual_mean: float = 0.0
    residual_skew: float = 0.0
    worst_decile_rmse: float | None = None


class ClusteringReport(BaseModel):
    """Clustering diagnostic metrics."""

    n_clusters: int = 0
    silhouette_score: float = 0.0
    davies_bouldin_score: float = 0.0
    calinski_harabasz_score: float = 0.0
    cluster_size_distribution: dict[str, float] = Field(default_factory=dict)


class MultilabelReport(BaseModel):
    """Multilabel classification diagnostic metrics."""

    subset_accuracy: float = 0.0
    hamming_loss: float = 0.0
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    per_label_f1: dict[str, float] = Field(default_factory=dict)


class RankingReport(BaseModel):
    """Ranking diagnostic metrics."""

    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    map_at_10: float = 0.0
    mean_reciprocal_rank: float = 0.0


class ForecastingReport(BaseModel):
    """Time series forecasting diagnostic metrics."""

    horizon_mae: list[float] = Field(default_factory=list)
    wape: float = 0.0
    smape: float = 0.0
    prediction_interval_coverage_80: float | None = None


class AnomalyReport(BaseModel):
    """Anomaly detection diagnostic metrics."""

    precision_at_k: float = 0.0
    roc_auc: float | None = None
    pr_auc: float | None = None
    detected_anomaly_ratio: float = 0.0


# High-error subgroup segment
class ErrorSlice(BaseModel):
    """Subgroup identified with elevated error rate."""

    slice_condition: str
    sample_count: int = 0
    slice_metric_score: float = 0.0
    degradation_vs_overall: float = 0.0


# Node 7: Independent Evaluator output
class EvaluationVerdict(BaseModel):
    """Output schema for Independent Evaluator audit."""

    thinking: str = Field(
        description="Generalization audit across Train, CV, and Holdout partitions."
    )
    cli_summary: str = Field(description="Crisp 1-3 line summary formatted for terminal display.")
    experiment_id: str
    verdict: Verdict
    rejection_reasons: list[str] = Field(default_factory=list)
    overfitting_status: OverfittingStatus
    train_vs_cv_gap: float = 0.0
    cv_vs_holdout_gap: float | None = None
    beats_current_champion: bool = False
    primary_metric_gain_vs_champion: float = 0.0
    secondary_metrics_comparison: str = ""
    worst_error_slices: list[ErrorSlice] = Field(default_factory=list)
    actionable_feedback_for_ml_lead: str = ""


# Model execution record stored in Ledger.runs
class ExperimentRun(BaseModel):
    """Complete record of a trained model experiment."""

    experiment_id: str
    iteration: int
    hypothesis_id: str = ""
    route_type: StrategistRoute = "fe_and_model"
    model_family: str = ""
    complexity_tier: ComplexityTier = "tier_1_simple_baseline"
    model_params: dict[str, Any] = Field(default_factory=dict)
    feature_version: int = 1
    script_path: str = ""
    artifact_dir: str = ""
    model_path: str = ""
    summary: str = ""

    # Primary CV and holdout scores
    train_score_mean: float | None = None
    cv_fold_scores: list[float] = Field(default_factory=list)
    cv_mean: float | None = None
    cv_std: float | None = None
    holdout_score: float | None = None
    secondary_metrics: dict[str, float] = Field(default_factory=dict)

    # Diagnostic sub-reports by task family
    classification_report: ClassificationReport | None = None
    regression_report: RegressionReport | None = None
    clustering_report: ClusteringReport | None = None
    multilabel_report: MultilabelReport | None = None
    ranking_report: RankingReport | None = None
    forecasting_report: ForecastingReport | None = None
    anomaly_report: AnomalyReport | None = None

    # Error analysis and feature importances
    top_feature_importances: dict[str, float] = Field(default_factory=dict)
    worst_error_slices: list[ErrorSlice] = Field(default_factory=list)

    # Audit outcome and champion status
    error_traceback: str = ""
    verdict: Verdict | None = None
    verdict_reasons: list[str] = Field(default_factory=list)
    overfitting_status: OverfittingStatus = "healthy"
    gain_vs_champion: float | None = None
    promoted_to_champion: bool = False
    evaluator_feedback: str = ""

    @property
    def crashed(self) -> bool:
        return bool(self.error_traceback)

    @property
    def approved(self) -> bool:
        return self.verdict == "approve" and not self.crashed and self.cv_mean is not None


# Priority 2: Production drift monitoring specification with numeric rules
class DriftSpec(BaseModel):
    """Production drift monitoring rule with numeric thresholds."""

    feature_or_target: str
    baseline_statistic: str
    drift_metric: Literal[
        "psi", "ks_test", "chi_square", "missing_rate_delta", "prediction_drift"
    ] = "psi"
    warning_threshold: float = Field(
        default=0.10, description="Numeric threshold triggering alert."
    )
    retrain_threshold: float = Field(
        default=0.25, description="Numeric threshold triggering model retraining."
    )
    action: str = "trigger_retraining"


# Node 8: Finalizer delivery output
class DeliveryPlan(BaseModel):
    """Output schema for Finalizer delivery agent."""

    thinking: str = Field(
        description="Review of approved candidate models and final deliverable packaging."
    )
    cli_summary: str = Field(description="Crisp 1-3 line summary formatted for terminal display.")
    delivery_strategy: Literal[
        "single_champion_model", "weighted_oof_ensemble", "stacking_ensemble"
    ]
    selected_experiment_ids: list[str] = Field(description="Eligible approved experiment IDs.")
    executive_summary: str = Field(description="Final technical project summary.")
    known_limitations_and_failure_modes: list[str] = Field(default_factory=list)
    monitoring_and_maintenance_plan: list[DriftSpec] = Field(default_factory=list)


# Stored final deliverable record
class FinalDeliverable(BaseModel):
    """Final package deliverables and production monitoring specification."""

    delivery_strategy: str = "single_champion_model"
    champion_experiment_id: str = ""
    ensemble_experiment_ids: list[str] = Field(default_factory=list)
    ensemble_weights: dict[str, float] = Field(default_factory=dict)
    final_cv_score: float | None = None
    final_holdout_score: float | None = None
    final_secondary_metrics: dict[str, float] = Field(default_factory=dict)
    predictions_path: str = ""
    serialized_pipeline_path: str = ""
    report_path: str = ""
    deployment_checks_passed: bool = False
    check_messages: list[str] = Field(default_factory=list)
    monitoring_plan: list[DriftSpec] = Field(default_factory=list)


# Master state ledger holding project memory
class Ledger(BaseModel):
    """Master project state serialized to memory/ledger.json after each node."""

    schema_version: int = SCHEMA_VERSION

    # User inputs
    user_task: str = ""
    user_metrics_text: str = ""
    created_at: str = ""

    # Stage 1-2 Project Scope
    scope: ProjectScope | None = None
    brief: ProjectBrief = Field(default_factory=ProjectBrief)

    # Stage 3-4 EDA State
    eda: EdaState = Field(default_factory=EdaState)

    # Stages 5-7 Experiment Strategy and History
    validation_plan: ValidationPlan | None = None
    splits_path: str = ""
    hypotheses: list[HypothesisRecord] = Field(default_factory=list)
    feature_versions: list[FeatureVersionRecord] = Field(default_factory=list)
    runs: list[ExperimentRun] = Field(default_factory=list)

    # Champion-vs-Challenger linear ratchet state
    champion_experiment_id: str = ""
    champion_feature_version: int = 0
    latest_candidate_feature_version: int = 0

    # Stage 8-9 Final Delivery
    final: FinalDeliverable = Field(default_factory=FinalDeliverable)

    # Orchestration state
    phase: Phase = "data_engineer"
    current_route: StrategistRoute = "fe_and_model"
    iteration: int = 0
    eda_revisits_used: int = 0
    last_tuned_iteration: int = -1
    status: RunStatus = "running"
    stop_reason: StopReason | None = None
    error_message: str = ""

    # Generate unique deterministic experiment identifier
    def next_experiment_id(self, offset: int = 0) -> str:
        return f"experiment_{len(self.runs) + offset + 1:03d}"

    # Generate unique finding identifier
    def next_finding_id(self, offset: int = 0) -> str:
        return f"finding_{len(self.eda.findings) + offset + 1:03d}"

    # Generate unique feature idea identifier
    def next_idea_id(self, offset: int = 0) -> str:
        return f"idea_{len(self.eda.feature_hypotheses) + offset + 1:03d}"

    # Generate unique hypothesis identifier
    def next_hypothesis_id(self) -> str:
        return f"hyp_{len(self.hypotheses) + 1:03d}"

    # Retrieve current verified Champion run
    def champion_run(self, exclude_experiment_id: str = "") -> ExperimentRun | None:
        if self.champion_experiment_id and self.champion_experiment_id != exclude_experiment_id:
            for run in self.runs:
                if run.experiment_id == self.champion_experiment_id and run.approved:
                    return run
        approved = [r for r in self.runs if r.approved and r.experiment_id != exclude_experiment_id]
        if not approved:
            return None
        if self.brief.higher_is_better:
            return max(
                approved, key=lambda r: r.cv_mean if r.cv_mean is not None else float("-inf")
            )
        return min(approved, key=lambda r: r.cv_mean if r.cv_mean is not None else float("inf"))

    @property
    def champion_score(self) -> float | None:
        champ = self.champion_run()
        return champ.cv_mean if champ else None

    @property
    def champion_model_type(self) -> str:
        champ = self.champion_run()
        return champ.model_family if champ else ""

    # Return baseline feature version for current round
    def active_base_feature_version(self) -> int:
        if self.champion_feature_version > 0:
            return self.champion_feature_version
        return self.latest_candidate_feature_version

    # Retrieve specific feature version record
    def get_feature_version_record(self, version: int) -> FeatureVersionRecord | None:
        for fv in self.feature_versions:
            if fv.feature_version == version:
                return fv
        return None

    # Promote newly evaluated run to Champion if it beats current Champion
    def update_champion_after_evaluation(self, evaluated_run: ExperimentRun) -> bool:
        if not evaluated_run.approved or evaluated_run.cv_mean is None:
            return False
        current_champ = self.champion_run(exclude_experiment_id=evaluated_run.experiment_id)
        if current_champ is None or current_champ.cv_mean is None:
            self.champion_experiment_id = evaluated_run.experiment_id
            self.champion_feature_version = evaluated_run.feature_version
            evaluated_run.gain_vs_champion = 0.0
            evaluated_run.promoted_to_champion = True
            return True

        raw_diff = evaluated_run.cv_mean - current_champ.cv_mean
        evaluated_run.gain_vs_champion = raw_diff if self.brief.higher_is_better else -raw_diff
        better = (
            evaluated_run.cv_mean > current_champ.cv_mean
            if self.brief.higher_is_better
            else evaluated_run.cv_mean < current_champ.cv_mean
        )
        if better and evaluated_run.overfitting_status != "severe_overfit":
            self.champion_experiment_id = evaluated_run.experiment_id
            self.champion_feature_version = evaluated_run.feature_version
            evaluated_run.promoted_to_champion = True
            return True
        return False

    # Filter untested feature hypotheses
    def untried_feature_hypotheses(self) -> list[FeatureIdea]:
        tested_ids = {
            op.linked_idea_id
            for fv in self.feature_versions
            for op in fv.operations
            if op.linked_idea_id
        }
        open_list = [
            h
            for h in self.eda.feature_hypotheses
            if h.status == "untested" and h.idea_id not in tested_ids
        ]
        order = {"high": 0, "medium": 1, "low": 2}
        return sorted(open_list, key=lambda h: order.get(h.priority, 1))

    # Summarize history including best run and promotion events
    def round_summary_table(self) -> list[str]:
        rows: list[str] = []
        for hyp in self.hypotheses:
            it = hyp.iteration
            round_runs = [r for r in self.runs if r.iteration == it]
            if round_runs:
                # Pick best run for iteration
                best_r = (
                    max(
                        round_runs,
                        key=lambda r: r.cv_mean if r.cv_mean is not None else float("-inf"),
                    )
                    if self.brief.higher_is_better
                    else min(
                        round_runs,
                        key=lambda r: r.cv_mean if r.cv_mean is not None else float("inf"),
                    )
                )
                score_str = f"CV={best_r.cv_mean:.4f}" if best_r.cv_mean is not None else "crashed"
                hold_str = (
                    f", Holdout={best_r.holdout_score:.4f}"
                    if best_r.holdout_score is not None
                    else ""
                )
                champ_tag = " [NEW CHAMPION]" if best_r.promoted_to_champion else ""
                rows.append(
                    f"Round {it} ({hyp.route} | {best_r.model_family} on feat_v{best_r.feature_version}): "
                    f"{score_str}{hold_str} [{best_r.overfitting_status}]{champ_tag} | Lesson: {hyp.lesson_learned or hyp.hypothesis_statement}"
                )
            else:
                rows.append(f"Round {it} ({hyp.route}): {hyp.outcome} | {hyp.hypothesis_statement}")
        return rows
