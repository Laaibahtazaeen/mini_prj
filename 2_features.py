# # import pandas as pd
# # import numpy as np
# # import re
# # import pickle
# # from sklearn.feature_extraction.text import CountVectorizer

# # # Load datasets
# # print("Loading data...")
# # train_df = pd.read_csv('dataset/train.csv')
# # test_df  = pd.read_csv('dataset/test.csv')
# # val_df   = pd.read_csv('dataset/val.csv')

# # def extract_features(url):
# #     features = {}
    
# #     # Basic URL features
# #     features['url_length']      = len(url)
# #     features['num_dots']        = url.count('.')
# #     features['num_hyphens']     = url.count('-')
# #     features['num_underscores'] = url.count('_')
# #     features['num_slashes']     = url.count('/')
# #     features['num_digits']      = sum(c.isdigit() for c in url)
# #     features['num_special']     = sum(not c.isalnum() for c in url)
# #     features['num_letters']     = sum(c.isalpha() for c in url)
    
# #     # Protocol features
# #     features['has_https']       = 1 if url.startswith('https') else 0
# #     features['has_http']        = 1 if url.startswith('http') else 0
    
# #     # Has IP address
# #     features['has_ip']          = 1 if re.search(
# #         r'\d+\.\d+\.\d+\.\d+', url) else 0
    
# #     # Has @ symbol
# #     features['has_at']          = 1 if '@' in url else 0
    
# #     # Has double slash
# #     features['has_double_slash'] = 1 if '//' in url[7:] else 0
    
# #     # Domain features
# #     try:
# #         domain = url.split('/')[2]
# #         features['domain_length']   = len(domain)
# #         features['num_subdomains']  = domain.count('.')
# #     except:
# #         features['domain_length']   = 0
# #         features['num_subdomains']  = 0
    
# #     # URL depth
# #     features['url_depth']       = url.count('/')
    
# #     # Has suspicious words
# #     suspicious = ['login', 'verify', 'secure', 'account',
# #                   'update', 'banking', 'confirm', 'signin',
# #                   'password', 'credit', 'paypal', 'ebay']
# #     features['has_suspicious']  = 1 if any(
# #         word in url.lower() for word in suspicious) else 0
    
# #     # URL entropy (randomness)
# #     import math
# #     prob = [float(url.count(c)) / len(url) for c in set(url)]
# #     features['url_entropy']     = -sum(
# #         p * math.log(p + 1e-10) for p in prob)
    
# #     return features

# # # Extract features
# # print("Extracting URL features...")
# # train_features = [extract_features(url) for url in train_df['url']]
# # test_features  = [extract_features(url) for url in test_df['url']]
# # val_features   = [extract_features(url) for url in val_df['url']]

# # train_feat_df  = pd.DataFrame(train_features)
# # test_feat_df   = pd.DataFrame(test_features)
# # val_feat_df    = pd.DataFrame(val_features)

# # # Character N-gram features
# # print("Extracting N-gram features...")
# # vectorizer = CountVectorizer(
# #     analyzer='char',
# #     ngram_range=(2, 3),
# #     max_features=100
# # )

# # # Fit only on training data
# # train_ngram = vectorizer.fit_transform(train_df['url']).toarray()
# # test_ngram  = vectorizer.transform(test_df['url']).toarray()
# # val_ngram   = vectorizer.transform(val_df['url']).toarray()

# # train_ngram_df = pd.DataFrame(
# #     train_ngram, columns=[f'ng_{i}' for i in range(100)])
# # test_ngram_df  = pd.DataFrame(
# #     test_ngram,  columns=[f'ng_{i}' for i in range(100)])
# # val_ngram_df   = pd.DataFrame(
# #     val_ngram,   columns=[f'ng_{i}' for i in range(100)])

# # # Combine all features
# # train_final = pd.concat([train_feat_df, train_ngram_df], axis=1)
# # test_final  = pd.concat([test_feat_df,  test_ngram_df],  axis=1)
# # val_final   = pd.concat([val_feat_df,   val_ngram_df],   axis=1)

# # # Add labels
# # train_final['label'] = train_df['label'].values
# # test_final['label']  = test_df['label'].values
# # val_final['label']   = val_df['label'].values

# # # Save features
# # train_final.to_csv('dataset/train_features.csv', index=False)
# # test_final.to_csv('dataset/test_features.csv',   index=False)
# # val_final.to_csv('dataset/val_features.csv',     index=False)

# # # Save vectorizer for deployment
# # import os
# # os.makedirs('model', exist_ok=True)
# # pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

# # print("Feature extraction done!")
# # print("Train features shape:", train_final.shape)
# # print("Test features shape:",  test_final.shape)
# # print("Val features shape:",   val_final.shape)


# import pandas as pd
# import numpy as np
# import re
# import pickle
# import math
# from sklearn.feature_extraction.text import CountVectorizer

# # Load datasets
# print("Loading data...")
# train_df = pd.read_csv('dataset/train.csv')
# test_df  = pd.read_csv('dataset/test.csv')
# val_df   = pd.read_csv('dataset/val.csv')

# def extract_features(url):
#     features = {}

#     # Basic URL features
#     features['url_length']          = len(url)
#     features['num_dots']            = url.count('.')
#     features['num_hyphens']         = url.count('-')
#     features['num_underscores']     = url.count('_')
#     features['num_slashes']         = url.count('/')
#     features['num_digits']          = sum(c.isdigit() for c in url)
#     features['num_special']         = sum(not c.isalnum() for c in url)
#     features['num_letters']         = sum(c.isalpha() for c in url)
#     features['num_params']          = url.count('?')
#     features['num_fragments']       = url.count('#')
#     features['num_equals']          = url.count('=')
#     features['num_ampersand']       = url.count('&')
#     features['num_percent']         = url.count('%')

#     # Protocol features
#     features['has_https']           = 1 if url.startswith('https') else 0
#     features['has_http']            = 1 if url.startswith('http')  else 0

#     # Suspicious features
#     features['has_ip']              = 1 if re.search(
#         r'\d+\.\d+\.\d+\.\d+', url) else 0
#     features['has_at']              = 1 if '@' in url else 0
#     features['has_double_slash']    = 1 if '//' in url[7:] else 0
#     features['has_port']            = 1 if re.search(
#         r':\d+', url) else 0

#     # Domain features
#     try:
#         domain = url.split('/')[2]
#         features['domain_length']   = len(domain)
#         features['num_subdomains']  = domain.count('.')
#         features['has_www']         = 1 if domain.startswith('www') else 0
#     except:
#         features['domain_length']   = 0
#         features['num_subdomains']  = 0
#         features['has_www']         = 0

#     # Path features
#     features['url_depth']           = url.count('/')
#     features['path_length']         = len(url.split('?')[0])

#     # Ratio features
#     features['digit_ratio']         = features['num_digits'] / (len(url) + 1)
#     features['letter_ratio']        = features['num_letters'] / (len(url) + 1)
#     features['special_ratio']       = features['num_special'] / (len(url) + 1)

#     # Suspicious words
#     suspicious = ['login', 'verify', 'secure', 'account',
#                   'update', 'banking', 'confirm', 'signin',
#                   'password', 'credit', 'paypal', 'ebay',
#                   'amazon', 'apple', 'microsoft', 'google',
#                   'free', 'lucky', 'bonus', 'winner']
#     features['has_suspicious']      = 1 if any(
#         word in url.lower() for word in suspicious) else 0
#     features['suspicious_count']    = sum(
#         1 for word in suspicious if word in url.lower())

#     # Entropy
#     prob = [float(url.count(c)) / len(url) for c in set(url)]
#     features['url_entropy']         = -sum(
#         p * math.log(p + 1e-10) for p in prob)

#     return features

# # Extract features
# print("Extracting URL features...")
# train_features  = [extract_features(url) for url in train_df['url']]
# test_features   = [extract_features(url) for url in test_df['url']]
# val_features    = [extract_features(url) for url in val_df['url']]

# train_feat_df   = pd.DataFrame(train_features)
# test_feat_df    = pd.DataFrame(test_features)
# val_feat_df     = pd.DataFrame(val_features)

# # Character N-gram features (increased to 200)
# print("Extracting N-gram features...")
# vectorizer = CountVectorizer(
#     analyzer='char',
#     ngram_range=(2, 4),      # increased to 4
#     max_features=200         # increased from 100
# )

# train_ngram     = vectorizer.fit_transform(train_df['url']).toarray()
# test_ngram      = vectorizer.transform(test_df['url']).toarray()
# val_ngram       = vectorizer.transform(val_df['url']).toarray()

# train_ngram_df  = pd.DataFrame(
#     train_ngram, columns=[f'ng_{i}' for i in range(200)])
# test_ngram_df   = pd.DataFrame(
#     test_ngram,  columns=[f'ng_{i}' for i in range(200)])
# val_ngram_df    = pd.DataFrame(
#     val_ngram,   columns=[f'ng_{i}' for i in range(200)])

# # Combine all features
# train_final     = pd.concat([train_feat_df, train_ngram_df], axis=1)
# test_final      = pd.concat([test_feat_df,  test_ngram_df],  axis=1)
# val_final       = pd.concat([val_feat_df,   val_ngram_df],   axis=1)

# # Add labels
# train_final['label'] = train_df['label'].values
# test_final['label']  = test_df['label'].values
# val_final['label']   = val_df['label'].values

# # Save features
# train_final.to_csv('dataset/train_features.csv', index=False)
# test_final.to_csv('dataset/test_features.csv',   index=False)
# val_final.to_csv('dataset/val_features.csv',     index=False)

# # Save vectorizer
# import os
# os.makedirs('model', exist_ok=True)
# pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

# print("Feature extraction done!")
# print("Train features shape:", train_final.shape)
# print("Test features shape:",  test_final.shape)
# print("Val features shape:",   val_final.shape)



import pandas as pd
import numpy as np
import re
import pickle
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load datasets
print("Loading data...")
train_df = pd.read_csv('dataset/train.csv')
test_df  = pd.read_csv('dataset/test.csv')
val_df   = pd.read_csv('dataset/val.csv')

def extract_features(url):
    features = {}

    # Basic URL features
    features['url_length']          = len(url)
    features['num_dots']            = url.count('.')
    features['num_hyphens']         = url.count('-')
    features['num_underscores']     = url.count('_')
    features['num_slashes']         = url.count('/')
    features['num_digits']          = sum(c.isdigit() for c in url)
    features['num_special']         = sum(not c.isalnum() for c in url)
    features['num_letters']         = sum(c.isalpha() for c in url)
    features['num_params']          = url.count('?')
    features['num_fragments']       = url.count('#')
    features['num_equals']          = url.count('=')
    features['num_ampersand']       = url.count('&')
    features['num_percent']         = url.count('%')
    features['num_at']              = url.count('@')
    features['num_tilde']           = url.count('~')
    features['num_comma']           = url.count(',')
    features['num_plus']            = url.count('+')
    features['num_asterisk']        = url.count('*')
    features['num_exclamation']     = url.count('!')
    features['num_dollar']          = url.count('$')

    # Protocol features
    features['has_https']           = 1 if url.startswith('https') else 0
    features['has_http']            = 1 if url.startswith('http')  else 0

    # Suspicious features
    features['has_ip']              = 1 if re.search(
        r'\d+\.\d+\.\d+\.\d+', url) else 0
    features['has_at']              = 1 if '@' in url else 0
    features['has_double_slash']    = 1 if '//' in url[7:] else 0
    features['has_port']            = 1 if re.search(
        r':\d+', url) else 0
    features['has_encoded']         = 1 if '%' in url else 0
    features['has_shortener']       = 1 if any(
        s in url for s in ['bit.ly','tinyurl','goo.gl',
                           'ow.ly','t.co','tiny.cc']) else 0

    # Domain features
    try:
        domain = url.split('/')[2]
        features['domain_length']   = len(domain)
        features['num_subdomains']  = domain.count('.')
        features['has_www']         = 1 if domain.startswith('www') else 0
        features['domain_digits']   = sum(c.isdigit() for c in domain)
        features['domain_hyphens']  = domain.count('-')
    except:
        features['domain_length']   = 0
        features['num_subdomains']  = 0
        features['has_www']         = 0
        features['domain_digits']   = 0
        features['domain_hyphens']  = 0

    # Path features
    features['url_depth']           = url.count('/')
    features['path_length']         = len(url.split('?')[0])

    # Ratio features
    features['digit_ratio']         = features['num_digits'] / (len(url) + 1)
    features['letter_ratio']        = features['num_letters'] / (len(url) + 1)
    features['special_ratio']       = features['num_special'] / (len(url) + 1)
    features['dot_ratio']           = features['num_dots'] / (len(url) + 1)
    features['hyphen_ratio']        = features['num_hyphens'] / (len(url) + 1)

    # Suspicious words
    suspicious = ['login', 'verify', 'secure', 'account',
                  'update', 'banking', 'confirm', 'signin',
                  'password', 'credit', 'paypal', 'ebay',
                  'amazon', 'apple', 'microsoft', 'google',
                  'free', 'lucky', 'bonus', 'winner',
                  'click', 'here', 'redirect', 'link',
                  'webscr', 'cmd', 'dispatch', 'session']
    features['has_suspicious']      = 1 if any(
        word in url.lower() for word in suspicious) else 0
    features['suspicious_count']    = sum(
        1 for word in suspicious if word in url.lower())

    # Entropy
    prob = [float(url.count(c)) / len(url) for c in set(url)]
    features['url_entropy']         = -sum(
        p * math.log(p + 1e-10) for p in prob)

    # Longest word length in URL
    words = re.split(r'\W+', url)
    features['longest_word']        = max(
        (len(w) for w in words), default=0)
    features['avg_word_length']     = np.mean(
        [len(w) for w in words if len(w) > 0])

    return features

# Extract features
print("Extracting URL features...")
train_features  = [extract_features(url) for url in train_df['url']]
test_features   = [extract_features(url) for url in test_df['url']]
val_features    = [extract_features(url) for url in val_df['url']]

train_feat_df   = pd.DataFrame(train_features)
test_feat_df    = pd.DataFrame(test_features)
val_feat_df     = pd.DataFrame(val_features)

# N-gram features (increased to 500)
print("Extracting N-gram features...")
ngram_vectorizer = CountVectorizer(
    analyzer='char',
    ngram_range=(2, 4),
    max_features=500
)
train_ngram      = ngram_vectorizer.fit_transform(
    train_df['url']).toarray()
test_ngram       = ngram_vectorizer.transform(
    test_df['url']).toarray()
val_ngram        = ngram_vectorizer.transform(
    val_df['url']).toarray()

train_ngram_df   = pd.DataFrame(
    train_ngram, columns=[f'ng_{i}' for i in range(500)])
test_ngram_df    = pd.DataFrame(
    test_ngram,  columns=[f'ng_{i}' for i in range(500)])
val_ngram_df     = pd.DataFrame(
    val_ngram,   columns=[f'ng_{i}' for i in range(500)])

# TF-IDF features (NEW - this is what will boost accuracy)
print("Extracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 4),
    max_features=500,
    sublinear_tf=True
)
train_tfidf      = tfidf_vectorizer.fit_transform(
    train_df['url']).toarray()
test_tfidf       = tfidf_vectorizer.transform(
    test_df['url']).toarray()
val_tfidf        = tfidf_vectorizer.transform(
    val_df['url']).toarray()

train_tfidf_df   = pd.DataFrame(
    train_tfidf, columns=[f'tf_{i}' for i in range(500)])
test_tfidf_df    = pd.DataFrame(
    test_tfidf,  columns=[f'tf_{i}' for i in range(500)])
val_tfidf_df     = pd.DataFrame(
    val_tfidf,   columns=[f'tf_{i}' for i in range(500)])

# Combine ALL features
print("Combining all features...")
train_final      = pd.concat(
    [train_feat_df, train_ngram_df, train_tfidf_df], axis=1)
test_final       = pd.concat(
    [test_feat_df,  test_ngram_df,  test_tfidf_df],  axis=1)
val_final        = pd.concat(
    [val_feat_df,   val_ngram_df,   val_tfidf_df],   axis=1)

# Add labels
train_final['label'] = train_df['label'].values
test_final['label']  = test_df['label'].values
val_final['label']   = val_df['label'].values

# Save features
train_final.to_csv('dataset/train_features.csv', index=False)
test_final.to_csv('dataset/test_features.csv',   index=False)
val_final.to_csv('dataset/val_features.csv',     index=False)

# Save both vectorizers
import os
os.makedirs('model', exist_ok=True)
pickle.dump(ngram_vectorizer, 
            open('model/ngram_vectorizer.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, 
            open('model/tfidf_vectorizer.pkl', 'wb'))

print("Done!")
print("Total features:", train_final.shape[1])
print("Train shape:",    train_final.shape)
print("Test shape:",     test_final.shape)