import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
)

os.makedirs('results', exist_ok=True)

# ── Load ─────────────────────────────────────────────────────
print("Loading model and test data...")
model   = pickle.load(open('model/lightgbm_model.pkl', 'rb'))
test_df = pd.read_csv('dataset/test_features.csv')

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# ── Feature alignment (model column order must match training) ─
trained_cols = model.feature_name_
missing = set(trained_cols) - set(X_test.columns)
extra   = set(X_test.columns) - set(trained_cols)
for col in missing:
    X_test[col] = 0
X_test = X_test.drop(columns=list(extra))
X_test = X_test[trained_cols]
print(f"Features aligned: {X_test.shape[1]} ✓")

# ── Predict ──────────────────────────────────────────────────
y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)

print("\n" + "="*40)
print("           EVALUATION RESULTS")
print("="*40)
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")
print(f"  ROC-AUC   : {roc_auc*100:.2f}%")
print("="*40)

# ── Inference speed ──────────────────────────────────────────
start = time.perf_counter()
model.predict(X_test.iloc[:500])
elapsed   = time.perf_counter() - start
ms_sample = (elapsed / 500) * 1000
print(f"\n  Inference speed : {ms_sample:.3f} ms / sample")
size_mb = os.path.getsize('model/lightgbm_model.pkl') / (1024 * 1024)
print(f"  Model size      : {size_mb:.2f} MB")

# ════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════

# ── 1. Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            ax=ax)
ax.set_title('Confusion Matrix — LightGBM', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)
plt.close()
print("\n✓ Saved: results/confusion_matrix.png")

# ── 2. Metrics bar chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
values  = [accuracy*100, precision*100, recall*100, f1*100, roc_auc*100]
colors  = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
bars = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=1.2)
ax.set_ylim(90, 101)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('results/metrics.png', dpi=150)
plt.close()
print("✓ Saved: results/metrics.png")

# ── 3. ROC Curve ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color='#3498db', lw=2,
        label=f'LightGBM (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('results/roc_curve.png', dpi=150)
plt.close()
print("✓ Saved: results/roc_curve.png")

# ── 4. Feature importance (top 20) ───────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=model.feature_name_)
top20    = feat_imp.nlargest(20)

fig, ax = plt.subplots(figsize=(9, 6))
top20.sort_values().plot(kind='barh', ax=ax, color='#3498db')
ax.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=150)
plt.close()
print("✓ Saved: results/feature_importance.png")

print("\n✓ All evaluation plots saved to results/")