from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import create_derived_features, handle_missing_values, treat_outliers_iqr


def test_handle_missing_values_fills_total_charges() -> None:
    frame = pd.DataFrame(
        {
            'TotalCharges': [' ', '120.5', np.nan],
            'MonthlyCharges': [20.0, 40.0, 60.0],
            'tenure': [1, 2, 3],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
        }
    )

    cleaned = handle_missing_values(frame)

    assert cleaned['TotalCharges'].isna().sum() == 0
    assert cleaned['TotalCharges'].dtype.kind in {'f', 'i'}


def test_create_derived_features_adds_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            'TotalCharges': [200.0, 600.0],
            'MonthlyCharges': [50.0, 75.0],
            'tenure': [4, 8],
            'PhoneService': ['Yes', 'No'],
            'MultipleLines': ['Yes', 'No phone service'],
            'InternetService': ['DSL', 'No'],
            'OnlineSecurity': ['Yes', 'No internet service'],
            'OnlineBackup': ['No', 'No internet service'],
            'DeviceProtection': ['Yes', 'No internet service'],
            'TechSupport': ['No', 'No internet service'],
            'StreamingTV': ['Yes', 'No internet service'],
            'StreamingMovies': ['No', 'No internet service'],
            'Contract': ['One year', 'Two year'],
        }
    )

    engineered = create_derived_features(frame)

    assert {'AvgMonthlySpend', 'ServiceCount', 'ContractValue'}.issubset(engineered.columns)
    assert engineered.loc[0, 'ServiceCount'] > engineered.loc[1, 'ServiceCount']


def test_treat_outliers_iqr_caps_large_values() -> None:
    frame = pd.DataFrame({'value': [1, 2, 3, 1000]})

    capped = treat_outliers_iqr(frame, numeric_columns=['value'])

    assert capped['value'].max() < 1000
