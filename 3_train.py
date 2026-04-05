import pickle
import pandas as pd
import lightgbm as lgb

# ── Load features ────────────────────────────────────────────
print("Loading features...")
train_df = pd.read_csv('dataset/train_features.csv')
val_df   = pd.read_csv('dataset/val_features.csv')

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_val   = val_df.drop('label', axis=1)
y_val   = val_df['label']

print(f"Train : {len(X_train):,} samples  |  {X_train.shape[1]} features")
print(f"Val   : {len(X_val):,} samples")

# ── Model ────────────────────────────────────────────────────
model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=10,
    num_leaves=80,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced',  
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# ── Train ────────────────────────────────────────────────────
print("\nTraining LightGBM — this takes ~5-10 minutes...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=200),
    ],
)

# ── Save ─────────────────────────────────────────────────────
pickle.dump(model, open('model/lightgbm_model.pkl', 'wb'))
print(f"\n✓ Model saved  |  Best iteration: {model.best_iteration_}")