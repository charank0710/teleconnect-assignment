"""Metric calculation and plotting helpers for model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .utils import adjusted_r2_score, ensure_directory


sns.set_theme(style="whitegrid")


def classification_metrics(y_true, y_pred, y_prob=None) -> dict[str, float]:
    """Compute standard classification metrics."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def regression_metrics(y_true, y_pred, n_features: int) -> dict[str, float]:
    """Compute standard regression metrics including adjusted R-squared."""

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "adjusted_r2": adjusted_r2_score(r2, len(y_true), n_features),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path | str | None = None) -> None:
    """Plot a confusion matrix."""

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    if save_path is not None:
        ensure_directory(Path(save_path).parent)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_roc_curve(y_true, y_prob, title: str, save_path: Path | str | None = None) -> None:
    """Plot the ROC curve."""

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path is not None:
        ensure_directory(Path(save_path).parent)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, title: str, save_path: Path | str | None = None) -> None:
    """Plot actual versus predicted values for regression."""

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=30, alpha=0.7)
    min_value = min(np.min(y_true), np.min(y_pred))
    max_value = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    if save_path is not None:
        ensure_directory(Path(save_path).parent)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_residual_distribution(residuals, title: str, save_path: Path | str | None = None) -> None:
    """Plot the residual distribution for a regression model."""

    plt.figure(figsize=(6, 5))
    sns.histplot(residuals, kde=True, bins=30, color="#2563eb")
    plt.axvline(0, color="black", linestyle="--")
    plt.title(title)
    plt.xlabel("Residual")
    if save_path is not None:
        ensure_directory(Path(save_path).parent)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_feature_importance(feature_names: Iterable[str], importances: Iterable[float], title: str, save_path: Path | str | None = None) -> None:
    """Plot a horizontal feature importance bar chart."""

    importance_frame = pd.DataFrame({"feature": list(feature_names), "importance": list(importances)}).sort_values("importance", ascending=False).head(20)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance_frame, x="importance", y="feature", color="#0f766e")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("")
    if save_path is not None:
        ensure_directory(Path(save_path).parent)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Convert a list of metric rows into a tidy comparison table."""

    return pd.DataFrame(rows)

