import pandas as pd

def load_txt_file(filepath):
    urls = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    labels.append(1 if parts[0].strip() == 'phishing' else 0)
                    urls.append(parts[1].strip())
    return pd.DataFrame({'url': urls, 'label': labels})

# Load all files
print("Loading datasets...")
train_df = load_txt_file('dataset/small_dataset/train.txt')
test_df  = load_txt_file('dataset/small_dataset/test.txt')
val_df   = load_txt_file('dataset/small_dataset/val.txt')

# Print info
print("Train size:", len(train_df))
print("Phishing in train:", len(train_df[train_df['label']==1]))
print("Legitimate in train:", len(train_df[train_df['label']==0]))
print("\nTest size:", len(test_df))
print("Val size:", len(val_df))

# Clean data
for df in [train_df, test_df, val_df]:
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df = df[df['url'].str.strip() != '']

# Save as CSV
train_df.to_csv('dataset/train.csv', index=False)
test_df.to_csv('dataset/test.csv',   index=False)
val_df.to_csv('dataset/val.csv',     index=False)

print("\nDone! CSV files saved!")
print("\nSample data:")
print(train_df.head())