# Credit Scoring Model

A machine learning project to predict individual creditworthiness using real financial data from 150,000 borrowers.

## Results

| Model | ROC-AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | **0.8635** | 0.2154 | 0.7701 | 0.3367 |
| Random Forest | 0.8616 | 0.2800 | 0.6384 | 0.3892 |
| Decision Tree | 0.8408 | 0.2504 | 0.6379 | 0.3597 |

Best model: **Logistic Regression** with ROC-AUC of 0.8635

## Project Overview
Built an end-to-end binary classification pipeline to predict whether a borrower will default on a loan within 2 years.

## Dataset

- Source: [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
- 150,000 borrower records
- Target variable: `SeriousDlqin2yrs` (1 = defaulted, 0 = good standing)
- Default rate: 6.68% (heavily imbalanced)

## Pipeline

1. **Data cleaning** — handled 29,731 missing income values, removed age outliers, capped credit utilization
2. **Feature engineering** — created 6 new features including DebtToIncomeRatio, PaymentReliability, HighUtilization flag
3. **Class balancing** — applied SMOTE to fix 93/7 class imbalance
4. **Model training** — trained Logistic Regression, Decision Tree, and Random Forest
5. **Evaluation** — compared models using Precision, Recall, F1-Score, and ROC-AUC

## Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

# Download dataset from Kaggle link above, save as credit_data.csv

# Run the model
python credit_scoring_model.py
```

## Output

The script prints stage-by-stage results and saves a full evaluation chart as `credit_scoring_results.png`.

## Author

Aysha-Sohail — Computer Science Student
