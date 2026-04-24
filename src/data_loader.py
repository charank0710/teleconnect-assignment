"""Data loading and synthetic fallback generation for Telco churn data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import PROJECT_ROOT, ensure_directory, set_random_seed


RAW_DATA_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def _choice(rng: np.random.Generator, values: list[str], probabilities: list[float], size: int) -> np.ndarray:
    """Sample categorical values with probabilities."""

    return rng.choice(values, size=size, p=probabilities)


def generate_synthetic_telco_data(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """Generate a Telco-shaped dataset for offline execution and testing."""

    set_random_seed(random_state)
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(0, 73, size=n_samples)
    gender = _choice(rng, ["Male", "Female"], [0.49, 0.51], n_samples)
    senior = rng.binomial(1, 0.16, size=n_samples)
    partner = _choice(rng, ["Yes", "No"], [0.48, 0.52], n_samples)
    dependents = np.where(partner == "Yes", _choice(rng, ["Yes", "No"], [0.36, 0.64], n_samples), _choice(rng, ["Yes", "No"], [0.18, 0.82], n_samples))
    phone_service = _choice(rng, ["Yes", "No"], [0.9, 0.1], n_samples)
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        _choice(rng, ["Yes", "No"], [0.42, 0.58], n_samples),
    )
    internet_service = _choice(rng, ["Fiber optic", "DSL", "No"], [0.44, 0.38, 0.18], n_samples)

    def _service_flag(yes_probability: float) -> np.ndarray:
        return np.where(
            internet_service == "No",
            "No internet service",
            _choice(rng, ["Yes", "No"], [yes_probability, 1 - yes_probability], n_samples),
        )

    online_security = _service_flag(0.46)
    online_backup = _service_flag(0.52)
    device_protection = _service_flag(0.51)
    tech_support = _service_flag(0.43)
    streaming_tv = _service_flag(0.58)
    streaming_movies = _service_flag(0.59)
    contract = _choice(rng, ["Month-to-month", "One year", "Two year"], [0.55, 0.25, 0.20], n_samples)
    paperless = _choice(rng, ["Yes", "No"], [0.6, 0.4], n_samples)
    payment_method = _choice(
        rng,
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        [0.34, 0.21, 0.23, 0.22],
        n_samples,
    )

    service_multiplier = (
        (internet_service == "Fiber optic").astype(float) * 28
        + (internet_service == "DSL").astype(float) * 12
        + (phone_service == "Yes").astype(float) * 8
        + (streaming_tv == "Yes").astype(float) * 10
        + (streaming_movies == "Yes").astype(float) * 10
        + (online_security == "Yes").astype(float) * 7
        + (tech_support == "Yes").astype(float) * 8
    )
    monthly_charges = np.clip(
        18
        + service_multiplier
        + senior * 4
        + (paperless == "Yes").astype(float) * 1.5
        + rng.normal(0, 4.5, size=n_samples),
        18,
        120,
    ).round(2)
    total_charges = np.where(tenure == 0, 0.0, monthly_charges * tenure + rng.normal(0, 24, size=n_samples)).round(2)

    churn_logit = (
        1.35 * (contract == "Month-to-month").astype(float)
        + 0.95 * (internet_service == "Fiber optic").astype(float)
        - 0.03 * tenure
        - 0.55 * (partner == "Yes").astype(float)
        - 0.55 * (dependents == "Yes").astype(float)
        - 0.70 * (tech_support == "Yes").astype(float)
        - 0.50 * (online_security == "Yes").astype(float)
        + 0.012 * monthly_charges
        + 0.40 * (payment_method == "Electronic check").astype(float)
        - 1.4
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    churn = np.where(rng.random(n_samples) < churn_probability, "Yes", "No")

    frame = pd.DataFrame(
        {
            "customerID": [f"CUST-{index:05d}" for index in range(1, n_samples + 1)],
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

    missing_indices = rng.choice(frame.index, size=max(1, n_samples // 70), replace=False)
    frame.loc[missing_indices, "TotalCharges"] = " "
    return frame


def load_telco_data(data_path: Path | str | None = None, fallback_to_synthetic: bool = True) -> pd.DataFrame:
    """Load the Telco dataset from disk, optionally falling back to synthetic data."""

    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "raw" / RAW_DATA_FILENAME

    csv_path = Path(data_path)
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
    elif fallback_to_synthetic:
        frame = generate_synthetic_telco_data()
        synthetic_path = PROJECT_ROOT / "data" / "raw" / "generated_telco_customer_churn.csv"
        ensure_directory(synthetic_path.parent)
        frame.to_csv(synthetic_path, index=False)
    else:
        raise FileNotFoundError(f"Could not find dataset at {csv_path}")

    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    return frame


def validate_telco_schema(frame: pd.DataFrame) -> list[str]:
    """Return the missing required columns, if any."""

    required_columns = {
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }
    return sorted(required_columns.difference(frame.columns))


def split_train_val_test(
    frame: pd.DataFrame,
    target_column: str,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataset into 70/15/15 train, validation, and test partitions."""

    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("Split sizes must be between 0 and 1.")

    target_values = frame[target_column] if stratify else None
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=test_size + validation_size,
        random_state=random_state,
        stratify=target_values,
    )
    if stratify:
        temp_target = temp_frame[target_column]
    else:
        temp_target = None
    validation_fraction = validation_size / (test_size + validation_size)
    validation_frame, test_frame = train_test_split(
        temp_frame,
        test_size=1 - validation_fraction,
        random_state=random_state,
        stratify=temp_target,
    )
    return train_frame.reset_index(drop=True), validation_frame.reset_index(drop=True), test_frame.reset_index(drop=True)

