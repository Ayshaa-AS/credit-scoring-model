# ============================================================
#  CREDIT SCORING MODEL — Complete Pipeline
#  Dataset: cs-training.csv (Kaggle "Give Me Some Credit")
#  Save this file and cs-training.csv in the same folder.
#  Run: python credit_scoring_model.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# STAGE 1 — LOAD DATA
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  STAGE 1: Loading Data")
print("="*55)

df = pd.read_csv('credit_data.csv')

# The Kaggle dataset has an unnamed index column — drop it
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nClass distribution:\n{df['SeriousDlqin2yrs'].value_counts()}")
print(f"\nDefault rate: {df['SeriousDlqin2yrs'].mean()*100:.2f}%")


# ─────────────────────────────────────────────
# STAGE 2 — PREPROCESSING
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  STAGE 2: Preprocessing")
print("="*55)

# 2a. Check missing values
print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")

 #2b. Fill missing values
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

# 2b2. Fill any remaining NaN in ALL columns
df = df.fillna(df.median(numeric_only=True))

# 2c. Remove extreme outliers in age
df = df[(df['age'] >= 18) & (df['age'] <= 100)]

# 2d. Cap RevolvingUtilization at 1 (values > 1 are data errors)
df['RevolvingUtilizationOfUnsecuredLines'] = df[
    'RevolvingUtilizationOfUnsecuredLines'
].clip(upper=1.0)

print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")
print(f"\nDataset shape after cleaning: {df.shape}")


# ─────────────────────────────────────────────
# STAGE 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  STAGE 3: Feature Engineering")
print("="*55)

# Debt-to-income ratio
df['DebtToIncomeRatio'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)

# Payment reliability (inverse of late payments)
df['TotalLatePayments'] = (
    df['NumberOfTime30-59DaysPastDueNotWorse'] +
    df['NumberOfTime60-89DaysPastDueNotWorse'] +
    df['NumberOfTimes90DaysLate']
)
df['PaymentReliability'] = 1 / (df['TotalLatePayments'] + 1)

# High credit utilization flag (>70% is risky)
df['HighUtilization'] = (
    df['RevolvingUtilizationOfUnsecuredLines'] > 0.7
).astype(int)

# Credit lines per dependent
df['LinesPerDependent'] = df['NumberOfOpenCreditLinesAndLoans'] / (
    df['NumberOfDependents'] + 1
)

# Age buckets (young / middle / senior)
df['AgeBucket'] = pd.cut(
    df['age'],
    bins=[17, 30, 50, 100],
    labels=[0, 1, 2]
).astype(int)

print("New features created:")
new_features = [
    'DebtToIncomeRatio', 'TotalLatePayments',
    'PaymentReliability', 'HighUtilization',
    'LinesPerDependent', 'AgeBucket'
]
for f in new_features:
    print(f"  ✓ {f}")


# ─────────────────────────────────────────────
# STAGE 4 — TRAIN / TEST SPLIT + SCALING
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  STAGE 4: Train/Test Split & Scaling")
print("="*55)

TARGET = 'SeriousDlqin2yrs'
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # keep class balance in both splits
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Apply SMOTE only on training data to fix class imbalance
print("\nApplying SMOTE to balance training data...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE — class distribution:\n{pd.Series(y_train_res).value_counts()}")


# ─────────────────────────────────────────────
# STAGE 5 — MODEL TRAINING
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  STAGE 5: Model Training")
print("="*55)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=8,
        class_weight='balanced',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1          # use all CPU cores
    ),
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    trained_models[name] = model
    print(f"  ✓ {name} trained")


# ─────────────────────────────────────────────
# STAGE 6 — EVALUATION
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  STAGE 6: Evaluation")
print("="*55)

results = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    report = classification_report(
        y_test, y_pred,
        target_names=['Good (0)', 'Default (1)'],
        output_dict=True
    )

    results[name] = {
        'auc': auc,
        'precision': report['Default (1)']['precision'],
        'recall':    report['Default (1)']['recall'],
        'f1':        report['Default (1)']['f1-score'],
        'y_pred':    y_pred,
        'y_prob':    y_prob,
    }

    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Precision : {results[name]['precision']:.4f}")
    print(f"  Recall    : {results[name]['recall']:.4f}")
    print(f"  F1-Score  : {results[name]['f1']:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=['Good (0)', 'Default (1)']
    ))

# Summary table
print("\n" + "="*55)
print("  RESULTS SUMMARY")
print("="*55)
summary = pd.DataFrame(results).T[['auc', 'precision', 'recall', 'f1']]
summary.columns = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
print(summary.round(4).to_string())
best_model_name = summary['ROC-AUC'].idxmax()
print(f"\n★ Best model by ROC-AUC: {best_model_name}")


# ─────────────────────────────────────────────
# STAGE 7 — VISUALIZATIONS
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  STAGE 7: Generating Visualizations...")
print("="*55)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Credit Scoring Model — Full Evaluation Report', fontsize=16, fontweight='bold')

# --- Plot 1: Class Distribution (original) ---
ax = axes[0, 0]
counts = df[TARGET].value_counts()
ax.bar(['Good (0)', 'Default (1)'], counts.values,
       color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.5)
ax.set_title('Class Distribution (Original Data)', fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(counts.values):
    ax.text(i, v + 200, f'{v:,}', ha='center', fontsize=10)

# --- Plot 2: ROC Curves (all 3 models) ---
ax = axes[0, 1]
colors = ['#3498db', '#e67e22', '#9b59b6']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=color, lw=2)
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Plot 3: Metrics Bar Chart ---
ax = axes[0, 2]
metrics_df = summary[['Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
x = np.arange(len(metrics_df))
width = 0.2
metric_colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
for i, col in enumerate(metrics_df.columns):
    ax.bar(x + i*width, metrics_df[col], width, label=col,
           color=metric_colors[i], alpha=0.85)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(metrics_df.index, rotation=12, fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_title('Metrics Comparison', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# --- Plot 4: Confusion Matrix (best model) ---
ax = axes[1, 0]
best_model = trained_models[best_model_name]
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Good', 'Default'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_model_name}', fontweight='bold')

# --- Plot 5: Feature Importances (Random Forest) ---
ax = axes[1, 1]
rf = trained_models['Random Forest']
importances = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)
top10.sort_values().plot(kind='barh', ax=ax, color='#3498db', edgecolor='white')
ax.set_title('Top 10 Feature Importances (Random Forest)', fontweight='bold')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.3)

# --- Plot 6: Correlation Heatmap (top features) ---
ax = axes[1, 2]
top_cols = importances.nlargest(8).index.tolist() + [TARGET]
corr = df[top_cols].corr()
sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0,
            annot=True, fmt='.2f', linewidths=0.5,
            annot_kws={'size': 7})
ax.set_title('Correlation Heatmap (Top Features)', fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0,  labelsize=7)

plt.tight_layout()
plt.savefig('credit_scoring_results.png', dpi=150, bbox_inches='tight')
print("\n  ✓ Saved: credit_scoring_results.png")
plt.show()

print("\n" + "="*55)
print("  DONE! All stages complete.")
print(f"  Best model: {best_model_name}")
print(f"  Best ROC-AUC: {summary['ROC-AUC'].max():.4f}")
print("="*55 + "\n")
