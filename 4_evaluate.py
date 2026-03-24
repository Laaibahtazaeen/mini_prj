import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix)

# Load model and test data
print("Loading model and test data...")
model    = pickle.load(open('model/lightgbm_model.pkl', 'rb'))
test_df  = pd.read_csv('dataset/test_features.csv')

X_test   = test_df.drop('label', axis=1)
y_test   = test_df['label']

# Predict
y_pred   = model.predict(X_test)

# Metrics
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

# Print results
print("\n========== YOUR RESULTS ==========")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print("===================================")

# Compare with base paper
print("\n===== COMPARISON WITH BASE PAPER =====")
print(f"Base Paper CNN:      98.74%")
print(f"Your LightGBM:       {accuracy*100:.2f}%")
if accuracy*100 > 98.74:
    print("YOU BEAT THE BASE PAPER!")
else:
    print("Very close! Good result!")
print("=======================================")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            cmap='Blues')
plt.title('Confusion Matrix - LightGBM')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
plt.show()
print("\nConfusion matrix saved!")