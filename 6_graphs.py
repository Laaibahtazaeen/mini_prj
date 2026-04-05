"""
6_graphs.py — Standalone comparison charts generator.
Run AFTER training: python 6_graphs.py
Saves 6 high-quality PNG charts to results/
"""

import os, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

os.makedirs('results', exist_ok=True)

DARK   = '#0f0f1a'
CARD   = '#1a1a2e'
PURPLE = '#7c3aed'
CYAN   = '#06b6d4'
GREEN  = '#10b981'
RED    = '#ef4444'
GOLD   = '#f59e0b'
WHITE  = '#e8e8f0'
MUTED  = '#6b7280'

def styled_fig(w=9, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=WHITE, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a45')
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    return fig, ax

# ── Load model + data ────────────────────────────────────────
print("Loading model and test data...")
model    = pickle.load(open('model/lightgbm_model.pkl', 'rb'))
test_df  = pd.read_csv('dataset/test_features.csv')
train_df = pd.read_csv('dataset/train_features.csv')

X_test  = test_df.drop('label', axis=1)
y_test  = test_df['label']
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

# Align features
cols = model.feature_name_
for c in cols:
    if c not in X_test.columns:  X_test[c]  = 0
    if c not in X_train.columns: X_train[c] = 0
X_test  = X_test[cols]
X_train = X_train[cols]

# ── LightGBM predictions ─────────────────────────────────────
print("Computing LightGBM predictions...")
y_pred_lgb  = model.predict(X_test)
y_prob_lgb  = model.predict_proba(X_test)[:, 1]

lgb_acc  = accuracy_score(y_test, y_pred_lgb)  * 100
lgb_prec = precision_score(y_test, y_pred_lgb) * 100
lgb_rec  = recall_score(y_test, y_pred_lgb)    * 100
lgb_f1   = f1_score(y_test, y_pred_lgb)        * 100
lgb_auc  = roc_auc_score(y_test, y_prob_lgb)   * 100

# ── Logistic Regression baseline ────────────────────────────
print("Training Logistic Regression baseline (takes ~1 min)...")
# Use only hand-crafted features (first ~60 columns) for speed
hc_cols = [c for c in cols if not c.startswith('ng_') and not c.startswith('tf_')]
lr = LogisticRegression(max_iter=500, n_jobs=-1, random_state=42)
lr.fit(X_train[hc_cols], y_train)
y_pred_lr = lr.predict(X_test[hc_cols])
y_prob_lr = lr.predict_proba(X_test[hc_cols])[:, 1]

lr_acc  = accuracy_score(y_test, y_pred_lr)  * 100
lr_prec = precision_score(y_test, y_pred_lr) * 100
lr_rec  = recall_score(y_test, y_pred_lr)    * 100
lr_f1   = f1_score(y_test, y_pred_lr)        * 100
lr_auc  = roc_auc_score(y_test, y_prob_lr)   * 100

print(f"\nLightGBM  Acc={lgb_acc:.2f}%  F1={lgb_f1:.2f}%  AUC={lgb_auc:.2f}%")
print(f"LR Base   Acc={lr_acc:.2f}%   F1={lr_f1:.2f}%  AUC={lr_auc:.2f}%")


# ════════════════════════════════════════════════════════════
# CHART 1 — Metrics comparison grouped bar
# ════════════════════════════════════════════════════════════
fig, ax = styled_fig(11, 6)

metrics   = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
lgb_vals  = [lgb_acc, lgb_prec, lgb_rec, lgb_f1, lgb_auc]
lr_vals   = [lr_acc,  lr_prec,  lr_rec,  lr_f1,  lr_auc]
x = np.arange(len(metrics))
w = 0.36

b1 = ax.bar(x - w/2, lr_vals,  w, label='Logistic Regression (baseline)',
            color=MUTED,   alpha=0.85, zorder=3)
b2 = ax.bar(x + w/2, lgb_vals, w, label='LightGBM (your model)',
            color=PURPLE,  alpha=0.95, zorder=3)

ax.set_ylim(min(min(lr_vals), min(lgb_vals)) - 3, 101)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold', pad=16)
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
ax.grid(axis='y', color='#2a2a45', linewidth=0.8, zorder=0)
ax.legend(fontsize=10, facecolor=CARD, edgecolor='#2a2a45', labelcolor=WHITE)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=9, color=WHITE, fontweight='bold')

plt.tight_layout()
plt.savefig('results/chart1_metrics_comparison.png', dpi=160, bbox_inches='tight')
plt.close()
print("✓ Saved: results/chart1_metrics_comparison.png")


# ════════════════════════════════════════════════════════════
# CHART 2 — Confusion Matrix
# ════════════════════════════════════════════════════════════
fig, ax = styled_fig(7, 6)

cm = confusion_matrix(y_test, y_pred_lgb)
labels = ['Legitimate', 'Phishing']

cmap = sns.color_palette([CARD, PURPLE], as_cmap=True)
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            ax=ax, linewidths=2, linecolor=DARK,
            annot_kws={'size': 14, 'weight': 'bold', 'color': WHITE})

ax.set_title('Confusion Matrix — LightGBM', fontsize=14, fontweight='bold', pad=14)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.tick_params(colors=WHITE)

plt.tight_layout()
plt.savefig('results/chart2_confusion_matrix.png', dpi=160, bbox_inches='tight')
plt.close()
print("✓ Saved: results/chart2_confusion_matrix.png")


# ════════════════════════════════════════════════════════════
# CHART 3 — ROC Curve
# ════════════════════════════════════════════════════════════
fig, ax = styled_fig(8, 6)

fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_prob_lgb)
fpr_lr,  tpr_lr,  _ = roc_curve(y_test, y_prob_lr)

ax.plot(fpr_lr,  tpr_lr,  color=MUTED,  lw=2, linestyle='--',
        label=f'Logistic Regression  (AUC = {lr_auc/100:.4f})')
ax.plot(fpr_lgb, tpr_lgb, color=CYAN,   lw=2.5,
        label=f'LightGBM  (AUC = {lgb_auc/100:.4f})')
ax.plot([0, 1], [0, 1], color='#3a3a5a', lw=1.2, linestyle=':')

ax.fill_between(fpr_lgb, tpr_lgb, alpha=0.08, color=CYAN)
ax.set_xlim([-0.01, 1.0]); ax.set_ylim([0.0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold', pad=14)
ax.legend(fontsize=10, facecolor=CARD, edgecolor='#2a2a45', labelcolor=WHITE)
ax.grid(color='#2a2a45', linewidth=0.6)

plt.tight_layout()
plt.savefig('results/chart3_roc_curve.png', dpi=160, bbox_inches='tight')
plt.close()
print("✓ Saved: results/chart3_roc_curve.png")


# ════════════════════════════════════════════════════════════
# CHART 4 — Feature Importance (top 20)
# ════════════════════════════════════════════════════════════
fig, ax = styled_fig(10, 7)

feat_imp = pd.Series(model.feature_importances_, index=model.feature_name_)
top20    = feat_imp.nlargest(20).sort_values()

colors_bar = [CYAN if not n.startswith(('ng_', 'tf_')) else PURPLE for n in top20.index]
top20.plot(kind='barh', ax=ax, color=colors_bar, edgecolor='none', zorder=3)

ax.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold', pad=14)
ax.set_xlabel('Feature Importance Score', fontsize=12)
ax.grid(axis='x', color='#2a2a45', linewidth=0.8, zorder=0)

legend_patches = [
    mpatches.Patch(color=CYAN,   label='Hand-crafted feature'),
    mpatches.Patch(color=PURPLE, label='N-gram / TF-IDF feature'),
]
ax.legend(handles=legend_patches, fontsize=9,
          facecolor=CARD, edgecolor='#2a2a45', labelcolor=WHITE)

plt.tight_layout()
plt.savefig('results/chart4_feature_importance.png', dpi=160, bbox_inches='tight')
plt.close()
print("✓ Saved: results/chart4_feature_importance.png")


# ════════════════════════════════════════════════════════════
# CHART 5 — Confidence distribution
# ════════════════════════════════════════════════════════════
fig, ax = styled_fig(10, 6)

prob_phish = y_prob_lgb[y_test == 1]
prob_legit = y_prob_lgb[y_test == 0]

ax.hist(prob_legit, bins=60, alpha=0.75, color=GREEN,
        label='Legitimate URLs', density=True, zorder=3)
ax.hist(prob_phish, bins=60, alpha=0.75, color=RED,
        label='Phishing URLs',   density=True, zorder=3)
ax.axvline(0.5, color=WHITE, linestyle='--', linewidth=1.5, label='Decision boundary (0.5)')

ax.set_xlabel('Predicted Phishing Probability', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Model Confidence Distribution', fontsize=14, fontweight='bold', pad=14)
ax.legend(fontsize=10, facecolor=CARD, edgecolor='#2a2a45', labelcolor=WHITE)
ax.grid(color='#2a2a45', linewidth=0.6, zorder=0)

plt.tight_layout()
plt.savefig('results/chart5_confidence_distribution.png', dpi=160, bbox_inches='tight')
plt.close()
print("✓ Saved: results/chart5_confidence_distribution.png")


# ════════════════════════════════════════════════════════════
# CHART 6 — Speed / Size / Time comparison
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle('System Performance Comparison', fontsize=14,
             fontweight='bold', color=WHITE, y=1.02)

start = time.perf_counter()
model.predict(X_test.iloc[:500])
lgb_ms = (time.perf_counter() - start) / 500 * 1000

start = time.perf_counter()
lr.predict(X_test[hc_cols].iloc[:500])
lr_ms = (time.perf_counter() - start) / 500 * 1000

lgb_mb = os.path.getsize('model/lightgbm_model.pkl') / (1024*1024)
lr_mb  = 0.5  # LR is tiny — estimated

subplots_data = [
    ('Inference Speed\n(ms/sample, lower=better)', [lr_ms, lgb_ms], '%.3f ms'),
    ('Model Size\n(MB, lower=better)',              [lr_mb, lgb_mb], '%.1f MB'),
    ('Features Used\n(count)',                       [len(hc_cols), len(cols)], '%d'),
]

bar_colors = [MUTED, PURPLE]
bar_labels = ['Logistic Reg.', 'LightGBM']

for ax, (title, vals, fmt) in zip(axes, subplots_data):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values(): spine.set_edgecolor('#2a2a45')
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)

    bars = ax.bar(bar_labels, vals, color=bar_colors, edgecolor='none', width=0.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', color='#2a2a45', linewidth=0.7, zorder=0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                fmt % val, ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=WHITE)
    ax.set_xticklabels(bar_labels, fontsize=10, color=WHITE)

plt.tight_layout()
plt.savefig('results/chart6_system_performance.png', dpi=160, bbox_inches='tight')
plt.close()
print("✓ Saved: results/chart6_system_performance.png")

print("\n🎉  All 6 charts saved to results/")
print(f"\n{'='*45}")
print(f"  LightGBM  Accuracy : {lgb_acc:.2f}%")
print(f"  LightGBM  F1 Score : {lgb_f1:.2f}%")
print(f"  LightGBM  ROC-AUC  : {lgb_auc:.2f}%")
print(f"  Baseline  Accuracy : {lr_acc:.2f}%")
print(f"  Baseline  F1 Score : {lr_f1:.2f}%")
print(f"{'='*45}")