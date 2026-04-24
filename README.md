# TeleConnect ML Assignment

Customer churn prediction and monthly revenue forecasting for a telecom business, implemented as a reproducible supervised learning project. The repository includes exploratory analysis, feature engineering, model benchmarking, interpretation, and business recommendations.

## Dataset

Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The project is built for the Telco Customer Churn dataset. If the Kaggle CSV is not present in `data/raw/`, the loader falls back to a synthetic Telco-shaped dataset so the notebooks and tests can still run end-to-end.

## Installation

```bash
git clone https://github.com/your-username/teleconnect-ml-assignment.git
cd teleconnect-ml-assignment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## How to Run

1. Place the Kaggle CSV in `data/raw/` as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
2. Open and run `notebooks/01_EDA.ipynb` to generate the exploratory analysis outputs.
3. Run `notebooks/02_preprocessing.ipynb` to build the feature matrices and comparison datasets.
4. Run `notebooks/03_classification.ipynb` to benchmark the churn classifiers.
5. Run `notebooks/04_regression.ipynb` to benchmark the revenue regressors.
6. Run `notebooks/05_interpretation.ipynb` to generate SHAP, PDP, and business recommendations.

You can also import the reusable functions directly from `src/` for custom experimentation.

## Results Summary

On the synthetic fallback data, the strongest classification and regression baselines are tree-based models, which is consistent with the feature interactions in the Telco domain. After you run the notebooks on the real Kaggle dataset, the reports in `reports/` will contain the exact metrics, plots, and selected best models.

| Task | Best Model (placeholder) | Key Metric (placeholder) | Insight |
| --- | --- | --- | --- |
| Churn Classification | Random Forest / Gradient Boosting | F1, ROC-AUC | Contract type and support services are dominant churn drivers. |
| Revenue Forecasting | Random Forest Regressor | RMSE, R2 | Service bundle mix explains most variance in monthly charges. |

## Project Structure

- `data/` stores raw and processed data plus the data dictionary.
- `notebooks/` contains the five assignment notebooks.
- `src/` contains reusable preprocessing, training, and evaluation utilities.
- `models/` stores the saved best estimator artifacts.
- `reports/` stores figures and narrative result summaries.
- `tests/` contains unit tests for core preprocessing and evaluation helpers.

## Tech Stack

- `pandas` and `numpy` for data manipulation.
- `scikit-learn` for preprocessing, model training, tuning, and metrics.
- `imbalanced-learn` for SMOTE and random undersampling.
- `matplotlib` and `seaborn` for visualization.
- `shap` for model interpretation.
- `pytest` for lightweight validation.
