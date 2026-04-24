"""Feature engineering and preprocessing helpers for Telco churn data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except Exception:  # pragma: no cover - optional dependency fallback
    SMOTE = None
    RandomUnderSampler = None


def handle_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean common Telco missing value patterns and numeric coercions."""

    cleaned = frame.copy()
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["TotalCharges"].median())

    object_columns = cleaned.select_dtypes(include=["object"]).columns
    for column in object_columns:
        cleaned[column] = cleaned[column].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].mode(dropna=True).iloc[0])
    return cleaned


def treat_outliers_iqr(frame: pd.DataFrame, numeric_columns: list[str] | None = None) -> pd.DataFrame:
    """Cap numeric outliers using the IQR rule."""

    cleaned = frame.copy()
    if numeric_columns is None:
        numeric_columns = cleaned.select_dtypes(include=["number"]).columns.tolist()

    for column in numeric_columns:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        cleaned[column] = cleaned[column].clip(lower_bound, upper_bound)
    return cleaned


def encode_label_series(series: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    """Label-encode a single categorical series."""

    encoder = LabelEncoder()
    encoded = pd.Series(encoder.fit_transform(series.astype(str)), index=series.index, name=series.name)
    return encoded, encoder


def encode_target(series: pd.Series) -> pd.Series:
    """Convert a binary churn target into integers."""

    lowered = series.astype(str).str.lower()
    if set(lowered.unique()).issubset({"yes", "no"}):
        return lowered.map({"no": 0, "yes": 1}).astype(int)
    return pd.Series(LabelEncoder().fit_transform(series.astype(str)), index=series.index, name=series.name)


def create_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features that capture spend, breadth of services, and contract value."""

    engineered = frame.copy()
    if {"TotalCharges", "tenure", "MonthlyCharges"}.issubset(engineered.columns):
        tenure_safe = engineered["tenure"].replace(0, np.nan)
        engineered["AvgMonthlySpend"] = np.where(
            tenure_safe.isna(),
            engineered["MonthlyCharges"],
            engineered["TotalCharges"] / tenure_safe,
        )
        engineered["AvgMonthlySpend"] = engineered["AvgMonthlySpend"].replace([np.inf, -np.inf], np.nan).fillna(engineered["MonthlyCharges"])

    if set(SERVICE_COLUMNS).issubset(engineered.columns):
        service_count = pd.Series(0, index=engineered.index, dtype=float)
        for column in SERVICE_COLUMNS:
            service_count += engineered[column].astype(str).str.contains("Yes", case=False).astype(float)
        engineered["ServiceCount"] = service_count

    if {"Contract", "MonthlyCharges", "tenure"}.issubset(engineered.columns):
        contract_term = engineered["Contract"].map({"Month-to-month": 1, "One year": 12, "Two year": 24}).fillna(1)
        remaining_months = (contract_term - engineered["tenure"].clip(lower=0)).clip(lower=0)
        engineered["ContractValue"] = engineered["MonthlyCharges"] * remaining_months

    return engineered


def build_tabular_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scaler: str = "standard",
) -> ColumnTransformer:
    """Create a tabular preprocessing pipeline with imputation, scaling, and one-hot encoding."""

    scaler_step: Any
    if scaler == "standard":
        scaler_step = StandardScaler()
    elif scaler == "minmax":
        scaler_step = MinMaxScaler()
    elif scaler in {None, "none"}:
        scaler_step = "passthrough"
    else:
        raise ValueError("Scaler must be 'standard', 'minmax', or None.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_step),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def prepare_features(
    frame: pd.DataFrame,
    target_column: str | None = None,
    drop_columns: list[str] | None = None,
    scaler: str = "standard",
) -> tuple[pd.DataFrame, pd.Series | None, ColumnTransformer, list[str]]:
    """Create a model-ready feature matrix and optional target series."""

    working_frame = create_derived_features(handle_missing_values(frame))
    if drop_columns is None:
        drop_columns = []

    target = None
    if target_column is not None and target_column in working_frame.columns:
        target = encode_target(working_frame[target_column]) if working_frame[target_column].dtype == object else working_frame[target_column].copy()

    feature_frame = working_frame.drop(columns=[column for column in [target_column, *drop_columns] if column in working_frame.columns])
    numeric_features = feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in feature_frame.columns if column not in numeric_features]

    preprocessor = build_tabular_preprocessor(numeric_features, categorical_features, scaler=scaler)
    transformed = preprocessor.fit_transform(feature_frame)
    feature_names = preprocessor.get_feature_names_out().tolist()
    transformed_frame = pd.DataFrame(transformed, columns=feature_names, index=feature_frame.index)
    return transformed_frame, target, preprocessor, feature_names


def fit_feature_preprocessor(
    frame: pd.DataFrame,
    drop_columns: list[str] | None = None,
    scaler: str = "standard",
) -> tuple[ColumnTransformer, list[str]]:
    """Fit a preprocessing pipeline on a feature frame and return the transformer and feature names."""

    working_frame = create_derived_features(handle_missing_values(frame))
    if drop_columns is None:
        drop_columns = []

    feature_frame = working_frame.drop(columns=[column for column in drop_columns if column in working_frame.columns])
    numeric_features = feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in feature_frame.columns if column not in numeric_features]
    preprocessor = build_tabular_preprocessor(numeric_features, categorical_features, scaler=scaler)
    preprocessor.fit(feature_frame)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return preprocessor, feature_names


def transform_features(
    frame: pd.DataFrame,
    preprocessor: ColumnTransformer,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Transform a feature frame using a fitted preprocessing pipeline."""

    working_frame = create_derived_features(handle_missing_values(frame))
    if drop_columns is None:
        drop_columns = []

    feature_frame = working_frame.drop(columns=[column for column in drop_columns if column in working_frame.columns])
    feature_names = preprocessor.get_feature_names_out().tolist()
    transformed = preprocessor.transform(feature_frame)
    return pd.DataFrame(transformed, columns=feature_names, index=feature_frame.index)


def prepare_train_val_test_features(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_column: str,
    drop_columns: list[str] | None = None,
    scaler: str = "standard",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, ColumnTransformer, list[str]]:
    """Prepare split feature matrices with the preprocessing fitted on the training data only."""

    if drop_columns is None:
        drop_columns = []

    target_train = encode_target(train_frame[target_column]) if train_frame[target_column].dtype == object else train_frame[target_column].copy()
    target_validation = encode_target(validation_frame[target_column]) if validation_frame[target_column].dtype == object else validation_frame[target_column].copy()
    target_test = encode_target(test_frame[target_column]) if test_frame[target_column].dtype == object else test_frame[target_column].copy()

    preprocessor, feature_names = fit_feature_preprocessor(train_frame.drop(columns=[target_column]), drop_columns=drop_columns, scaler=scaler)
    train_features = transform_features(train_frame.drop(columns=[target_column]), preprocessor, drop_columns=drop_columns)
    validation_features = transform_features(validation_frame.drop(columns=[target_column]), preprocessor, drop_columns=drop_columns)
    test_features = transform_features(test_frame.drop(columns=[target_column]), preprocessor, drop_columns=drop_columns)
    return (
        train_features,
        target_train,
        validation_features,
        target_validation,
        test_features,
        target_test,
        preprocessor,
        feature_names,
    )


def feature_selection_correlation(frame: pd.DataFrame, threshold: float = 0.8) -> list[str]:
    """Return numeric columns that are not highly correlated with others."""

    correlations = frame.corr(numeric_only=True).abs()
    upper_triangle = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    return [column for column in upper_triangle.columns if not any(upper_triangle[column] > threshold)]


def feature_selection_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "classification",
    n_features_to_select: int | None = None,
) -> list[str]:
    """Use recursive feature elimination to keep the strongest predictors."""

    if task == "classification":
        estimator = LogisticRegression(max_iter=2000)
    else:
        estimator = LinearRegression()

    if n_features_to_select is None:
        n_features_to_select = max(5, X.shape[1] // 2)

    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
    selector.fit(X, y)
    return X.columns[selector.support_].tolist()


def feature_importance_tree(X: pd.DataFrame, y: pd.Series, task: str = "classification") -> pd.Series:
    """Fit a simple tree model and return sorted feature importances."""

    if task == "classification":
        model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    else:
        model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)


def feature_selection_mutual_information(X: pd.DataFrame, y: pd.Series, task: str = "classification") -> pd.Series:
    """Rank features by mutual information."""

    if task == "classification":
        scores = mutual_info_classif(X, y, random_state=42)
    else:
        scores = mutual_info_regression(X, y, random_state=42)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def handle_class_imbalance(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "smote",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Balance a classification training set using SMOTE or random undersampling."""

    if method == "smote":
        if SMOTE is None:
            raise ImportError("imbalanced-learn is required for SMOTE.")
        sampler = SMOTE(random_state=random_state)
    elif method in {"undersample", "random_undersample"}:
        if RandomUnderSampler is None:
            raise ImportError("imbalanced-learn is required for random undersampling.")
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError("method must be 'smote' or 'undersample'.")

    resampled_X, resampled_y = sampler.fit_resample(X, y)
    return pd.DataFrame(resampled_X, columns=X.columns), pd.Series(resampled_y, name=y.name)

