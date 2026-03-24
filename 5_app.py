from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np
import re
import math

app   = Flask(__name__)
model = pickle.load(open('model/lightgbm_model.pkl',     'rb'))
ngram_vec = pickle.load(open('model/ngram_vectorizer.pkl', 'rb'))
tfidf_vec = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))

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

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Phishing URL Detector</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: Arial; 
            background: #1a1a2e; 
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            width: 550px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 { color: #1a1a2e; margin-bottom: 5px; }
        h3 { color: #666; margin-bottom: 25px; }
        input {
            width: 100%;
            padding: 12px;
            font-size: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover { background: #0056b3; }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
            display: none;
        }
        .phishing   { background: #ffe0e0; color: red;   }
        .legitimate { background: #e0ffe0; color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Phishing URL Detector</h1>
        <h3>Powered by LightGBM</h3>
        <input type="text" id="url" 
               placeholder="Enter URL to check...">
        <button onclick="checkURL()">Check URL</button>
        <div class="result" id="result"></div>
    </div>
    <script>
        async function checkURL() {
            const url = document.getElementById('url').value;
            if (!url) { alert('Please enter a URL!'); return; }
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url: url})
            });
            const data = await response.json();
            const div  = document.getElementById('result');
            div.style.display = 'block';
            if (data.prediction === 'Phishing') {
                div.className = 'result phishing';
                div.innerHTML  = '⚠️ PHISHING URL DETECTED!';
            } else {
                div.className = 'result legitimate';
                div.innerHTML  = '✅ LEGITIMATE URL - Safe!';
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    url      = request.json['url']
    features = extract_features(url)
    feat_df  = pd.DataFrame([features])

    ngram    = ngram_vec.transform([url]).toarray()
    tfidf    = tfidf_vec.transform([url]).toarray()

    ngram_df = pd.DataFrame(
        ngram, columns=[f'ng_{i}' for i in range(500)])
    tfidf_df = pd.DataFrame(
        tfidf, columns=[f'tf_{i}' for i in range(500)])

    final_df   = pd.concat([feat_df, ngram_df, tfidf_df], axis=1)
    prediction = model.predict(final_df)[0]
    result     = 'Phishing' if prediction == 1 else 'Legitimate'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)