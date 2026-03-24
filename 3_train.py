# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import pickle

# # Load features
# print("Loading features...")
# train_df = pd.read_csv('dataset/train_features.csv')
# val_df   = pd.read_csv('dataset/val_features.csv')

# # Split features and labels
# X_train = train_df.drop('label', axis=1)
# y_train = train_df['label']
# X_val   = val_df.drop('label', axis=1)
# y_val   = val_df['label']

# print("Training size:",   len(X_train))
# print("Validation size:", len(X_val))

# # Build LightGBM model
# model = lgb.LGBMClassifier(
#     n_estimators=1000,
#     learning_rate=0.05,
#     max_depth=8,
#     num_leaves=50,
#     min_child_samples=20,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     n_jobs=-1
# )

# # Train model
# print("\nTraining LightGBM model...")
# model.fit(
#     X_train, y_train,
#     eval_set=[(X_val, y_val)],
#     callbacks=[
#         lgb.early_stopping(50),
#         lgb.log_evaluation(100)
#     ]
# )

# # Save model
# pickle.dump(model, open('model/lightgbm_model.pkl', 'wb'))
# print("\nModel saved successfully!")


import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

# Load features
print("Loading features...")
train_df = pd.read_csv('dataset/train_features.csv')
val_df   = pd.read_csv('dataset/val_features.csv')

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_val   = val_df.drop('label', axis=1)
y_val   = val_df['label']

print("Training size:",   len(X_train))
print("Validation size:", len(X_val))

# Improved LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=2000,       # increased from 1000
    learning_rate=0.01,      # slower learning = better accuracy
    max_depth=10,            # deeper trees
    num_leaves=100,          # more leaves
    min_child_samples=10,    # reduced for better learning
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,           # regularization
    reg_lambda=0.1,          # regularization
    random_state=42,
    n_jobs=-1
)

# Train model
print("\nTraining Improved LightGBM model...")
print("Please wait this will take 5-10 minutes...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(100)
    ]
)

# Save model
pickle.dump(model, open('model/lightgbm_model.pkl', 'wb'))
print("\nModel saved successfully!")