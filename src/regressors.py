"""Regressor benchmarking utilities for monthly revenue forecasting."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .evaluation import regression_metrics
from .utils import save_pickle


def build_regressor_configs(random_state: int = 42) -> dict[str, tuple[Pipeline, dict]]:
    """Return the regressor pipelines and parameter grids used in benchmarking."""

    return {
        "Linear Regression": (
            Pipeline(steps=[("scaler", StandardScaler()), ("model", LinearRegression())]),
            {"scaler": [StandardScaler(), MinMaxScaler()]},
        ),
        "Ridge": (
            Pipeline(steps=[("scaler", StandardScaler()), ("model", Ridge(random_state=random_state))]),
            {"model__alpha": [0.1, 1.0, 10.0, 30.0]},
        ),
        "Lasso": (
            Pipeline(steps=[("scaler", StandardScaler()), ("model", Lasso(max_iter=10000, random_state=random_state))]),
            {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
        ),
        "ElasticNet": (
            Pipeline(steps=[("scaler", StandardScaler()), ("model", ElasticNet(max_iter=10000, random_state=random_state))]),
            {"model__alpha": [0.001, 0.01, 0.1], "model__l1_ratio": [0.2, 0.5, 0.8]},
        ),
        "Decision Tree": (
            Pipeline(steps=[("model", DecisionTreeRegressor(random_state=random_state))]),
            {"model__max_depth": [4, 8, None], "model__min_samples_split": [2, 5, 10]},
        ),
        "Random Forest": (
            Pipeline(steps=[("model", RandomForestRegressor(random_state=random_state, n_jobs=-1))]),
            {"model__n_estimators": [150, 250], "model__max_depth": [None, 10, 15], "model__min_samples_leaf": [1, 2, 4]},
        ),
        "SVR": (
            Pipeline(steps=[("scaler", StandardScaler()), ("model", SVR())]),
            {"model__C": [1.0, 5.0, 10.0], "model__epsilon": [0.1, 0.2], "model__kernel": ["rbf", "linear"]},
        ),
    }


def benchmark_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    save_path: Path | str | None = None,
    return_details: bool = False,
) -> tuple[pd.DataFrame, Pipeline, dict] | tuple[pd.DataFrame, Pipeline, dict, list[dict]]:
    """Tune and compare regression models, returning a leaderboard and the best estimator."""

    configs = build_regressor_configs(random_state=random_state)
    rows: list[dict] = []
    details: list[dict] = []
    best_model = None
    best_row = None
    best_score = np.inf
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for model_name, (pipeline, param_grid) in configs.items():
        start_time = time.time()
        search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(X_train, y_train)
        validation_predictions = search.best_estimator_.predict(X_validation)
        validation_metrics = regression_metrics(y_validation, validation_predictions, n_features=X_train.shape[1])
        test_predictions = search.best_estimator_.predict(X_test)
        test_metrics = regression_metrics(y_test, test_predictions, n_features=X_train.shape[1])
        elapsed = time.time() - start_time

        row = {
            "model": model_name,
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_r2": validation_metrics["r2"],
            "validation_adjusted_r2": validation_metrics["adjusted_r2"],
            "test_mae": test_metrics["mae"],
            "test_mse": test_metrics["mse"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_adjusted_r2": test_metrics["adjusted_r2"],
            "training_time": elapsed,
            "best_params": search.best_params_,
        }
        rows.append(row)
        details.append(
            {
                "model": model_name,
                "estimator": search.best_estimator_,
                "validation_predictions": validation_predictions,
                "test_predictions": test_predictions,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "best_params": search.best_params_,
            }
        )

        if validation_metrics["rmse"] < best_score:
            best_score = float(validation_metrics["rmse"])
            best_model = search.best_estimator_
            best_row = row

    result_frame = pd.DataFrame(rows).sort_values(["validation_rmse", "validation_mae"]).reset_index(drop=True)
    if best_model is None or best_row is None:
        raise RuntimeError("No regressor could be trained.")

    final_model = best_model.fit(pd.concat([X_train, X_validation]), pd.concat([y_train, y_validation]))
    if save_path is not None:
        save_pickle(final_model, save_path)
    if return_details:
        return result_frame, final_model, best_row, details
    return result_frame, final_model, best_row


def extract_linear_coefficients(model, feature_names: list[str]) -> pd.Series:
    """Extract coefficient magnitudes from a linear regression pipeline."""

    estimator = getattr(model, "named_steps", {}).get("model", model)
    if not hasattr(estimator, "coef_"):
        raise ValueError("The supplied model does not expose coefficients.")
    coefficients = np.asarray(estimator.coef_).ravel()
    return pd.Series(coefficients, index=feature_names).sort_values(key=np.abs, ascending=False)

