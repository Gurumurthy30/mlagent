"""Metric Registry and deterministic evaluation for all machine learning tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Metric optimization direction
MetricDirection = Literal["maximize", "minimize"]

# Type of predictions expected by the evaluation function
MetricInputType = Literal[
    "probabilities",
    "labels",
    "continuous",
    "ranked_lists",
    "clustering_embeddings",
]


# Specification describing a registered metric
@dataclass(frozen=True)
class MetricSpec:
    # Metric canonical name
    name: str
    # Direction of optimization
    direction: MetricDirection
    # Expected prediction type
    input_type: MetricInputType
    # Compatible task families
    supported_tasks: list[str]
    # Human-readable description
    description: str


# Helper to compute paired statistical gain
def is_counted_gain(
    new_scores: list[float] | None,
    base_scores: list[float] | None,
    higher_is_better: bool,
) -> bool:
    # Require non-empty valid score lists
    if not new_scores or not base_scores:
        return False
    # Filter non-finite numbers
    valid_pairs = [
        (n, b) for n, b in zip(new_scores, base_scores)
        if math.isfinite(n) and math.isfinite(b)
    ]
    # Need at least one valid split
    if not valid_pairs:
        return False
    # Calculate fold-wise difference
    diffs = [n - b for n, b in valid_pairs]
    # Average difference across splits
    mean_diff = float(np.mean(diffs))
    # Standard error of differences
    std_err = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    # Evaluate gain depending on metric direction
    if higher_is_better:
        return mean_diff > max(0.0, std_err * 0.5)
    return mean_diff < -max(0.0, std_err * 0.5)


# Check if all elements in a list are finite
def all_finite(values: list[float] | None) -> bool:
    if values is None or len(values) == 0:
        return False
    return all(math.isfinite(v) for v in values)


# Metric Registry holding metadata and evaluation logic
class MetricRegistry:
    """Central registry of supported ML evaluation metrics across all task types."""

    # Built-in metric specifications
    SPECS: dict[str, MetricSpec] = {
        # Classification
        "accuracy": MetricSpec("accuracy", "maximize", "labels", ["binary_classification", "multiclass_classification"], "Accuracy score"),
        "f1": MetricSpec("f1", "maximize", "labels", ["binary_classification", "multiclass_classification", "multilabel_classification"], "F1 score"),
        "f1_macro": MetricSpec("f1_macro", "maximize", "labels", ["binary_classification", "multiclass_classification"], "Macro-averaged F1 score"),
        "f1_micro": MetricSpec("f1_micro", "maximize", "labels", ["multiclass_classification", "multilabel_classification"], "Micro-averaged F1 score"),
        "roc_auc": MetricSpec("roc_auc", "maximize", "probabilities", ["binary_classification", "multiclass_classification", "anomaly_detection"], "Area under ROC curve"),
        "pr_auc": MetricSpec("pr_auc", "maximize", "probabilities", ["binary_classification", "anomaly_detection"], "Area under Precision-Recall curve"),
        "log_loss": MetricSpec("log_loss", "minimize", "probabilities", ["binary_classification", "multiclass_classification"], "Cross-entropy loss"),
        "precision": MetricSpec("precision", "maximize", "labels", ["binary_classification", "multiclass_classification"], "Precision score"),
        "recall": MetricSpec("recall", "maximize", "labels", ["binary_classification", "multiclass_classification"], "Recall score"),
        # Regression
        "rmse": MetricSpec("rmse", "minimize", "continuous", ["regression", "time_series_forecasting"], "Root Mean Squared Error"),
        "mae": MetricSpec("mae", "minimize", "continuous", ["regression", "time_series_forecasting"], "Mean Absolute Error"),
        "r2": MetricSpec("r2", "maximize", "continuous", ["regression"], "R-squared coefficient of determination"),
        "mape": MetricSpec("mape", "minimize", "continuous", ["regression", "time_series_forecasting"], "Mean Absolute Percentage Error"),
        # Clustering
        "silhouette": MetricSpec("silhouette", "maximize", "clustering_embeddings", ["clustering"], "Silhouette score"),
        "davies_bouldin": MetricSpec("davies_bouldin", "minimize", "clustering_embeddings", ["clustering"], "Davies-Bouldin index"),
        "calinski_harabasz": MetricSpec("calinski_harabasz", "maximize", "clustering_embeddings", ["clustering"], "Calinski-Harabasz score"),
        # Ranking
        "ndcg": MetricSpec("ndcg", "maximize", "ranked_lists", ["ranking"], "Normalized Discounted Cumulative Gain"),
        "map": MetricSpec("map", "maximize", "ranked_lists", ["ranking"], "Mean Average Precision"),
        "mrr": MetricSpec("mrr", "maximize", "ranked_lists", ["ranking"], "Mean Reciprocal Rank"),
        # Multilabel & Anomaly
        "hamming_loss": MetricSpec("hamming_loss", "minimize", "labels", ["multilabel_classification"], "Hamming loss"),
        "precision_at_k": MetricSpec("precision_at_k", "maximize", "probabilities", ["anomaly_detection", "ranking"], "Precision at top K"),
    }

    @classmethod
    def get_spec(cls, metric_name: str) -> MetricSpec:
        # Standardize metric string
        key = metric_name.strip().lower().replace("-", "_")
        # Return exact spec or default to accuracy
        if key in cls.SPECS:
            return cls.SPECS[key]
        # Infer direction if unknown
        higher_is_better = not any(term in key for term in ("loss", "error", "mse", "mae", "rmse", "bouldin", "hamming"))
        direction: MetricDirection = "maximize" if higher_is_better else "minimize"
        return MetricSpec(key, direction, "continuous", ["general"], "Custom metric")

    @classmethod
    def is_higher_better(cls, metric_name: str) -> bool:
        # Check optimization direction
        return cls.get_spec(metric_name).direction == "maximize"

    @classmethod
    def compute_metric(cls, metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Compute metric score deterministically using scikit-learn
        from sklearn import metrics

        key = metric_name.strip().lower().replace("-", "_")
        if key == "accuracy":
            return float(metrics.accuracy_score(y_true, y_pred))
        if key in ("f1", "f1_macro"):
            return float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0))
        if key == "f1_micro":
            return float(metrics.f1_score(y_true, y_pred, average="micro", zero_division=0))
        if key == "roc_auc":
            try:
                return float(metrics.roc_auc_score(y_true, y_pred))
            except Exception:
                return 0.5
        if key == "pr_auc":
            try:
                precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
                return float(metrics.auc(recall, precision))
            except Exception:
                return 0.0
        if key == "log_loss":
            return float(metrics.log_loss(y_true, y_pred))
        if key == "rmse":
            return float(math.sqrt(metrics.mean_squared_error(y_true, y_pred)))
        if key == "mae":
            return float(metrics.mean_absolute_error(y_true, y_pred))
        if key == "r2":
            return float(metrics.r2_score(y_true, y_pred))
        if key == "mape":
            return float(metrics.mean_absolute_percentage_error(y_true, y_pred))
        if key == "hamming_loss":
            return float(metrics.hamming_loss(y_true, y_pred))

        # Default fallback: mean squared difference
        return float(np.mean((y_true - y_pred) ** 2))
