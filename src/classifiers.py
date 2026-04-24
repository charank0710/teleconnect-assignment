"""Classifier benchmarking utilities for churn prediction."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .evaluation import classification_metrics
from .utils import save_pickle


def build_classifier_configs(random_state: int = 42) -> dict[str, tuple[Pipeline, dict]]:
    """Return the classifier pipelines and parameter grids used in benchmarking."""

    return {
        "Logistic Regression": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state)),
                ]
            ),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            Pipeline(steps=[("model", DecisionTreeClassifier(random_state=random_state, class_weight="balanced"))]),
            {"model__max_depth": [3, 5, 8, None], "model__min_samples_split": [2, 5, 10]},
        ),
        "Random Forest": (
            Pipeline(steps=[("model", RandomForestClassifier(random_state=random_state, class_weight="balanced"))]),
            {"model__n_estimators": [150, 250], "model__max_depth": [None, 8, 12], "model__min_samples_leaf": [1, 2, 4]},
        ),
        "Gradient Boosting": (
            Pipeline(steps=[("model", GradientBoostingClassifier(random_state=random_state))]),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1], "model__min_samples_leaf": [1, 3, 5]},
        ),
        "SVM": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", SVC(probability=True, class_weight="balanced", random_state=random_state)),
                ]
            ),
            {"model__C": [0.5, 1.0, 2.0], "model__kernel": ["rbf", "linear"], "model__gamma": ["scale", "auto"]},
        ),
    }


def benchmark_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    save_path: Path | str | None = None,
) -> tuple[pd.DataFrame, Pipeline, dict]:
    """Tune and compare classification models, returning a leaderboard and the best estimator."""

    configs = build_classifier_configs(random_state=random_state)
    rows: list[dict] = []
    best_model = None
    best_row = None
    best_score = -np.inf
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for model_name, (pipeline, param_grid) in configs.items():
        start_time = time.time()
        search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        validation_predictions = search.best_estimator_.predict(X_validation)
        if hasattr(search.best_estimator_, "predict_proba"):
            validation_probability = search.best_estimator_.predict_proba(X_validation)[:, 1]
        else:
            validation_probability = None
        validation_metrics = classification_metrics(y_validation, validation_predictions, validation_probability)
        test_predictions = search.best_estimator_.predict(X_test)
        if hasattr(search.best_estimator_, "predict_proba"):
            test_probability = search.best_estimator_.predict_proba(X_test)[:, 1]
        else:
            test_probability = None
        test_metrics = classification_metrics(y_test, test_predictions, test_probability)
        elapsed = time.time() - start_time

        row = {
            "model": model_name,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_f1": validation_metrics["f1"],
            "validation_roc_auc": validation_metrics.get("roc_auc", np.nan),
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics.get("roc_auc", np.nan),
            "training_time": elapsed,
            "best_params": search.best_params_,
        }
        rows.append(row)

        if validation_metrics["f1"] > best_score:
            best_score = validation_metrics["f1"]
            best_model = search.best_estimator_
            best_row = row

    result_frame = pd.DataFrame(rows).sort_values(["validation_f1", "validation_roc_auc"], ascending=False).reset_index(drop=True)
    if best_model is None or best_row is None:
        raise RuntimeError("No classifier could be trained.")

    final_model = best_model.fit(pd.concat([X_train, X_validation]), pd.concat([y_train, y_validation]))
    if save_path is not None:
        save_pickle(final_model, save_path)
    return result_frame, final_model, best_row


def extract_tree_feature_importance(model, feature_names: list[str]) -> pd.Series:
    """Extract feature importances from tree-based classifiers."""

    estimator = getattr(model, "named_steps", {}).get("model", model)
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("The supplied model does not expose feature importances.")
    return pd.Series(estimator.feature_importances_, index=feature_names).sort_values(ascending=False)

