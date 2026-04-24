from __future__ import annotations

import numpy as np

from src.evaluation import classification_metrics, regression_metrics


def test_classification_metrics_are_computed() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = classification_metrics(y_true, y_pred, y_prob)

    assert metrics['accuracy'] == 0.75
    assert metrics['precision'] > 0
    assert metrics['roc_auc'] > 0.5


def test_regression_metrics_include_adjusted_r2() -> None:
    y_true = np.array([10.0, 12.0, 14.0, 16.0])
    y_pred = np.array([9.5, 12.5, 13.5, 15.5])

    metrics = regression_metrics(y_true, y_pred, n_features=2)

    assert metrics['mae'] > 0
    assert 'adjusted_r2' in metrics
