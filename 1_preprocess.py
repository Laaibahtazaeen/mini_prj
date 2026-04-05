

import os
import pandas as pd

# ── Directory setup ──
os.makedirs('dataset', exist_ok=True)
os.makedirs('model',   exist_ok=True)

def load_txt_file(filepath):
    urls, labels = [], []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                label = 1 if parts[0].strip().lower() == 'phishing' else 0
                url   = parts[1].strip()
                if url:
                    labels.append(label)
                    urls.append(url)
    return pd.DataFrame({'url': urls, 'label': labels})

# ── Load ───────
print("Loading datasets...")
train_df = load_txt_file('dataset/small_dataset/train.txt')
test_df  = load_txt_file('dataset/small_dataset/test.txt')
val_df   = load_txt_file('dataset/small_dataset/val.txt')

# ── Clean ───
for df in [train_df, test_df, val_df]:
    df.drop_duplicates(subset='url', inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

# ── Stats ────
print(f"\nTrain  : {len(train_df):,} rows  "
      f"| phishing={train_df['label'].sum():,}  "
      f"| legit={len(train_df)-train_df['label'].sum():,}")
print(f"Val    : {len(val_df):,} rows")
print(f"Test   : {len(test_df):,} rows")

# ── Save ────
train_df.to_csv('dataset/train.csv', index=False)
test_df.to_csv('dataset/test.csv',   index=False)
val_df.to_csv('dataset/val.csv',     index=False)

print("\n✓ Preprocessing done — CSV files saved to dataset/")
print(train_df.head())