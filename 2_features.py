import os
import re
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


NGRAM_FEATURES  = 150   
TFIDF_FEATURES  = 150  
SUSPICIOUS_WORDS = [
    'login', 'verify', 'secure', 'account', 'update', 'banking',
    'confirm', 'signin', 'password', 'credit', 'paypal', 'ebay',
    'amazon', 'apple', 'microsoft', 'google', 'free', 'lucky',
    'bonus', 'winner', 'click', 'here', 'redirect', 'link',
    'webscr', 'cmd', 'dispatch', 'session',
]
SHORTENERS = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 't.co', 'tiny.cc']


def extract_features(url: str) -> dict:
   
    f = {}

    # ── Basic counts ─────────────────────────────────────────
    f['url_length']      = len(url)
    f['num_dots']        = url.count('.')
    f['num_hyphens']     = url.count('-')
    f['num_underscores'] = url.count('_')
    f['num_slashes']     = url.count('/')
    f['num_digits']      = sum(c.isdigit() for c in url)
    f['num_special']     = sum(not c.isalnum() for c in url)
    f['num_letters']     = sum(c.isalpha() for c in url)
    f['num_params']      = url.count('?')
    f['num_fragments']   = url.count('#')
    f['num_equals']      = url.count('=')
    f['num_ampersand']   = url.count('&')
    f['num_percent']     = url.count('%')
    f['num_at']          = url.count('@')
    f['num_tilde']       = url.count('~')
    f['num_comma']       = url.count(',')
    f['num_plus']        = url.count('+')
    f['num_asterisk']    = url.count('*')
    f['num_exclamation'] = url.count('!')
    f['num_dollar']      = url.count('$')

    # ── Protocol ─────────────────────────────────────────────
    f['has_https']        = 1 if url.startswith('https') else 0
    f['has_http']         = 1 if url.startswith('http')  else 0

    # ── Suspicious patterns ───────────────────────────────────
    f['has_ip']           = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    f['has_at']           = 1 if '@' in url else 0
    f['has_double_slash'] = 1 if '//' in url[7:] else 0
    f['has_port']         = 1 if re.search(r':\d+', url) else 0
    f['has_encoded']      = 1 if '%' in url else 0
    f['has_shortener']    = 1 if any(s in url for s in SHORTENERS) else 0

    # ── Domain features ───────────────────────────────────────
    try:
        domain              = url.split('/')[2]
        f['domain_length']  = len(domain)
        f['num_subdomains'] = domain.count('.')
        f['has_www']        = 1 if domain.startswith('www') else 0
        f['domain_digits']  = sum(c.isdigit() for c in domain)
        f['domain_hyphens'] = domain.count('-')
    except IndexError:
        f['domain_length']  = 0
        f['num_subdomains'] = 0
        f['has_www']        = 0
        f['domain_digits']  = 0
        f['domain_hyphens'] = 0

    # ── Path ─────────────────────────────────────────────────
    f['url_depth']   = url.count('/')
    f['path_length'] = len(url.split('?')[0])

    # ── Ratios ───────────────────────────────────────────────
    n = len(url) + 1   # +1 to avoid division by zero
    f['digit_ratio']   = f['num_digits']  / n
    f['letter_ratio']  = f['num_letters'] / n
    f['special_ratio'] = f['num_special'] / n
    f['dot_ratio']     = f['num_dots']    / n
    f['hyphen_ratio']  = f['num_hyphens'] / n

    # ── Suspicious words ─────────────────────────────────────
    url_lower = url.lower()
    f['has_suspicious']   = 1 if any(w in url_lower for w in SUSPICIOUS_WORDS) else 0
    f['suspicious_count'] = sum(1 for w in SUSPICIOUS_WORDS if w in url_lower)

    # ── Entropy ──────────────────────────────────────────────
    prob              = [float(url.count(c)) / len(url) for c in set(url)]
    f['url_entropy']  = -sum(p * math.log(p + 1e-10) for p in prob)

    # ── Word length stats (BUG FIX: guard against empty list) ─
    words = [w for w in re.split(r'\W+', url) if len(w) > 0]
    if words:
        f['longest_word']    = max(len(w) for w in words)
        f['avg_word_length'] = float(np.mean([len(w) for w in words]))
    else:
        f['longest_word']    = 0
        f['avg_word_length'] = 0.0   # ← was NaN before, which broke model

    return f


# ── Load data ────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv('dataset/train.csv')
test_df  = pd.read_csv('dataset/test.csv')
val_df   = pd.read_csv('dataset/val.csv')

print(f"  Train={len(train_df):,}  Test={len(test_df):,}  Val={len(val_df):,}")

# ── Hand-crafted features ────────────────────────────────────
print("Extracting hand-crafted features...")
train_feat_df = pd.DataFrame([extract_features(u) for u in train_df['url']])
val_feat_df   = pd.DataFrame([extract_features(u) for u in val_df['url']])
test_feat_df  = pd.DataFrame([extract_features(u) for u in test_df['url']])

# ── N-gram features ──────────────────────────────────────────
print(f"Extracting N-gram features (max={NGRAM_FEATURES})...")
ngram_vec  = CountVectorizer(analyzer='char', ngram_range=(2, 4),
                              max_features=NGRAM_FEATURES)
train_ng   = ngram_vec.fit_transform(train_df['url']).toarray()
val_ng     = ngram_vec.transform(val_df['url']).toarray()
test_ng    = ngram_vec.transform(test_df['url']).toarray()

ng_cols        = [f'ng_{i}' for i in range(NGRAM_FEATURES)]
train_ngram_df = pd.DataFrame(train_ng, columns=ng_cols)
val_ngram_df   = pd.DataFrame(val_ng,   columns=ng_cols)
test_ngram_df  = pd.DataFrame(test_ng,  columns=ng_cols)

# ── TF-IDF features ──────────────────────────────────────────
print(f"Extracting TF-IDF features (max={TFIDF_FEATURES})...")
tfidf_vec  = TfidfVectorizer(analyzer='char', ngram_range=(2, 4),
                              max_features=TFIDF_FEATURES, sublinear_tf=True)
train_tf   = tfidf_vec.fit_transform(train_df['url']).toarray()
val_tf     = tfidf_vec.transform(val_df['url']).toarray()
test_tf    = tfidf_vec.transform(test_df['url']).toarray()

tf_cols        = [f'tf_{i}' for i in range(TFIDF_FEATURES)]
train_tfidf_df = pd.DataFrame(train_tf, columns=tf_cols)
val_tfidf_df   = pd.DataFrame(val_tf,   columns=tf_cols)
test_tfidf_df  = pd.DataFrame(test_tf,  columns=tf_cols)

# ── Combine ──────────────────────────────────────────────────
print("Combining all features...")
train_final = pd.concat([train_feat_df, train_ngram_df, train_tfidf_df], axis=1)
val_final   = pd.concat([val_feat_df,   val_ngram_df,   val_tfidf_df],   axis=1)
test_final  = pd.concat([test_feat_df,  test_ngram_df,  test_tfidf_df],  axis=1)

train_final['label'] = train_df['label'].values
val_final['label']   = val_df['label'].values
test_final['label']  = test_df['label'].values

# ── Save features ────────────────────────────────────────────
train_final.to_csv('dataset/train_features.csv', index=False)
val_final.to_csv('dataset/val_features.csv',     index=False)
test_final.to_csv('dataset/test_features.csv',   index=False)

# ── Save vectorizers ─────────────────────────────────────────
os.makedirs('model', exist_ok=True)
pickle.dump(ngram_vec,  open('model/ngram_vectorizer.pkl', 'wb'))
pickle.dump(tfidf_vec,  open('model/tfidf_vectorizer.pkl', 'wb'))

print(f"\n✓ Feature extraction done!")
print(f"  Total features : {train_final.shape[1] - 1}")
print(f"  Train shape    : {train_final.shape}")
print(f"  Val shape      : {val_final.shape}")
print(f"  Test shape     : {test_final.shape}")