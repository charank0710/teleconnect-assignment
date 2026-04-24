# Data Dictionary

## Source

Kaggle Telco Customer Churn dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Columns

- `customerID`: unique identifier for each customer
- `gender`: customer gender
- `SeniorCitizen`: binary senior citizen flag
- `Partner`: whether the customer has a partner
- `Dependents`: whether the customer has dependents
- `tenure`: months with the company
- `PhoneService`: phone subscription flag
- `MultipleLines`: multiple line subscription status
- `InternetService`: DSL, fiber optic, or no internet
- `OnlineSecurity`: online security subscription status
- `OnlineBackup`: online backup subscription status
- `DeviceProtection`: device protection subscription status
- `TechSupport`: tech support subscription status
- `StreamingTV`: streaming TV subscription status
- `StreamingMovies`: streaming movies subscription status
- `Contract`: contract type
- `PaperlessBilling`: paperless billing flag
- `PaymentMethod`: payment method
- `MonthlyCharges`: monthly revenue per customer
- `TotalCharges`: cumulative charges
- `Churn`: target label for churn classification

## Notes

- If the real Kaggle file is unavailable, the project automatically generates a synthetic Telco-shaped dataset with the same schema.
- Processed features are written to `data/processed/` when the notebooks or helper functions are executed.
