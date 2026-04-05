"""
Phishing URL Detector — Flask App
Run: python 5_app.py
Serves on http://localhost:5000
"""

import re
import math
import pickle
import os

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

# ── Load model + vectorizers ─────────────────────────────────
MODEL_DIR  = 'model'
model      = pickle.load(open(f'{MODEL_DIR}/lightgbm_model.pkl',     'rb'))
ngram_vec  = pickle.load(open(f'{MODEL_DIR}/ngram_vectorizer.pkl',    'rb'))
tfidf_vec  = pickle.load(open(f'{MODEL_DIR}/tfidf_vectorizer.pkl',    'rb'))

# Cache the feature column order expected by the model
TRAINED_COLS = model.feature_name_
NGRAM_COLS   = [f'ng_{i}' for i in range(len(ngram_vec.vocabulary_))]
TFIDF_COLS   = [f'tf_{i}' for i in range(len(tfidf_vec.vocabulary_))]

SUSPICIOUS_WORDS = [
    'login', 'verify', 'secure', 'account', 'update', 'banking',
    'confirm', 'signin', 'password', 'credit', 'paypal', 'ebay',
    'amazon', 'apple', 'microsoft', 'google', 'free', 'lucky',
    'bonus', 'winner', 'click', 'here', 'redirect', 'link',
    'webscr', 'cmd', 'dispatch', 'session',
]
SHORTENERS = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 't.co', 'tiny.cc']


# ─────────────────────────────────────────────────────────────
#  Feature extraction  (must match 2_features.py exactly)
# ─────────────────────────────────────────────────────────────
def extract_features(url: str) -> dict:
    f = {}
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

    f['has_https']        = 1 if url.startswith('https') else 0
    f['has_http']         = 1 if url.startswith('http')  else 0

    f['has_ip']           = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    f['has_at']           = 1 if '@' in url else 0
    f['has_double_slash'] = 1 if '//' in url[7:] else 0
    f['has_port']         = 1 if re.search(r':\d+', url) else 0
    f['has_encoded']      = 1 if '%' in url else 0
    f['has_shortener']    = 1 if any(s in url for s in SHORTENERS) else 0

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

    f['url_depth']   = url.count('/')
    f['path_length'] = len(url.split('?')[0])

    n = len(url) + 1
    f['digit_ratio']   = f['num_digits']  / n
    f['letter_ratio']  = f['num_letters'] / n
    f['special_ratio'] = f['num_special'] / n
    f['dot_ratio']     = f['num_dots']    / n
    f['hyphen_ratio']  = f['num_hyphens'] / n

    url_lower             = url.lower()
    f['has_suspicious']   = 1 if any(w in url_lower for w in SUSPICIOUS_WORDS) else 0
    f['suspicious_count'] = sum(1 for w in SUSPICIOUS_WORDS if w in url_lower)

    prob             = [float(url.count(c)) / len(url) for c in set(url)]
    f['url_entropy'] = -sum(p * math.log(p + 1e-10) for p in prob)

    words = [w for w in re.split(r'\W+', url) if w]
    if words:
        f['longest_word']    = max(len(w) for w in words)
        f['avg_word_length'] = float(np.mean([len(w) for w in words]))
    else:
        f['longest_word']    = 0
        f['avg_word_length'] = 0.0

    return f


def predict_url(url: str):
    """Return (label, confidence_pct, risk_flags)."""
    # Hand-crafted features
    feat_df = pd.DataFrame([extract_features(url)])

    # Vectorized features
    ngram_arr = ngram_vec.transform([url]).toarray()
    tfidf_arr = tfidf_vec.transform([url]).toarray()
    ngram_df  = pd.DataFrame(ngram_arr, columns=NGRAM_COLS)
    tfidf_df  = pd.DataFrame(tfidf_arr, columns=TFIDF_COLS)

    combined = pd.concat([feat_df, ngram_df, tfidf_df], axis=1)

    # ── Feature alignment (CRITICAL — must match training order) ──
    for col in TRAINED_COLS:
        if col not in combined.columns:
            combined[col] = 0
    combined = combined[TRAINED_COLS]   # reorder to match model

    prob       = float(model.predict_proba(combined)[0][1])
    label      = 'Phishing' if prob >= 0.5 else 'Legitimate'
    confidence = round(prob * 100 if label == 'Phishing' else (1 - prob) * 100, 1)

    # Risk flags for UI
    flags = []
    url_lower = url.lower()
    if feat_df['has_ip'].iloc[0]:              flags.append('Contains IP address')
    if feat_df['has_at'].iloc[0]:              flags.append('Contains @ symbol')
    if not feat_df['has_https'].iloc[0]:       flags.append('No HTTPS')
    if feat_df['has_encoded'].iloc[0]:         flags.append('URL encoded characters')
    if feat_df['has_shortener'].iloc[0]:       flags.append('URL shortener detected')
    if feat_df['has_double_slash'].iloc[0]:    flags.append('Double slash in path')
    if feat_df['has_port'].iloc[0]:            flags.append('Non-standard port')
    if feat_df['suspicious_count'].iloc[0] > 0:
        flags.append(f"{int(feat_df['suspicious_count'].iloc[0])} suspicious keyword(s)")
    if feat_df['url_length'].iloc[0] > 100:    flags.append('Unusually long URL')
    if feat_df['num_subdomains'].iloc[0] > 3:  flags.append('Excessive subdomains')

    return label, confidence, prob, flags


# ─────────────────────────────────────────────────────────────
#  HTML Template
# ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phishing URL Detector</title>
<style>
  :root {
    --bg:       #0f0f1a;
    --card:     #1a1a2e;
    --border:   #2a2a45;
    --accent:   #6c63ff;
    --accent2:  #00d2ff;
    --red:      #ff4757;
    --green:    #2ed573;
    --text:     #e8e8f0;
    --muted:    #8888aa;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }

  /* ── Header ── */
  .header { text-align: center; margin-bottom: 32px; }
  .header .logo {
    font-size: 48px;
    margin-bottom: 8px;
    filter: drop-shadow(0 0 20px rgba(108,99,255,.5));
  }
  .header h1 {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
  }
  .header p { color: var(--muted); font-size: 14px; }

  /* ── Card ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 36px;
    width: 100%;
    max-width: 600px;
    box-shadow: 0 20px 60px rgba(0,0,0,.4);
  }

  /* ── Input group ── */
  .input-group {
    display: flex;
    gap: 10px;
    margin-bottom: 24px;
  }
  .input-group input {
    flex: 1;
    padding: 14px 18px;
    background: #0f0f1a;
    border: 2px solid var(--border);
    border-radius: 12px;
    color: var(--text);
    font-size: 15px;
    outline: none;
    transition: border-color .2s;
  }
  .input-group input:focus { border-color: var(--accent); }
  .input-group input::placeholder { color: var(--muted); }

  button {
    padding: 14px 22px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity .2s, transform .1s;
    white-space: nowrap;
  }
  button:hover   { opacity: .9; transform: translateY(-1px); }
  button:active  { transform: translateY(0); }
  button:disabled { opacity: .5; cursor: not-allowed; }

  /* ── Result card ── */
  .result-card {
    display: none;
    border-radius: 16px;
    padding: 24px;
    margin-top: 4px;
    animation: fadeIn .3s ease;
  }
  .result-card.phishing {
    background: rgba(255,71,87,.1);
    border: 1px solid rgba(255,71,87,.3);
  }
  .result-card.legitimate {
    background: rgba(46,213,115,.1);
    border: 1px solid rgba(46,213,115,.3);
  }

  @keyframes fadeIn { from { opacity:0; transform:translateY(8px) } to { opacity:1; transform:none } }

  .result-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 18px;
  }
  .result-icon { font-size: 36px; }
  .result-title { font-size: 20px; font-weight: 700; }
  .result-card.phishing  .result-title { color: var(--red);   }
  .result-card.legitimate .result-title { color: var(--green); }
  .result-subtitle { font-size: 13px; color: var(--muted); margin-top: 2px; }

  /* ── Confidence bar ── */
  .conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .conf-label span:last-child { font-weight: 700; font-size: 15px; color: var(--text); }
  .conf-track {
    height: 8px;
    background: rgba(255,255,255,.08);
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 20px;
  }
  .conf-fill {
    height: 100%;
    border-radius: 99px;
    transition: width .6s cubic-bezier(.4,0,.2,1);
  }
  .phishing  .conf-fill { background: linear-gradient(90deg, #ff4757, #ff6b81); }
  .legitimate .conf-fill { background: linear-gradient(90deg, #2ed573, #7bed9f); }

  /* ── Flags ── */
  .flags-title { font-size: 13px; color: var(--muted); margin-bottom: 10px; }
  .flags { display: flex; flex-wrap: wrap; gap: 8px; }
  .flag {
    background: rgba(255,71,87,.15);
    border: 1px solid rgba(255,71,87,.25);
    color: #ff6b81;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
  }
  .no-flags {
    font-size: 13px;
    color: var(--green);
    opacity: .8;
  }

  /* ── Loader ── */
  .loader {
    display: none;
    text-align: center;
    padding: 20px;
    color: var(--muted);
    font-size: 14px;
  }
  .spinner {
    width: 32px; height: 32px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .7s linear infinite;
    margin: 0 auto 10px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Examples ── */
  .examples {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
  }
  .examples p { font-size: 12px; color: var(--muted); margin-bottom: 10px; }
  .example-chips { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip {
    background: rgba(255,255,255,.05);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 12px;
    cursor: pointer;
    transition: background .15s;
    color: var(--muted);
  }
  .chip:hover { background: rgba(108,99,255,.2); color: var(--text); border-color: var(--accent); }

  /* ── Footer ── */
  .footer { margin-top: 24px; text-align: center; font-size: 12px; color: var(--muted); }
</style>
</head>
<body>

<div class="header">
  <div class="logo">🛡️</div>
  <h1>Phishing URL Detector</h1>
  <p>Powered by LightGBM · Trained on 520K+ URLs</p>
</div>

<div class="card">
  <div class="input-group">
    <input type="text" id="urlInput"
           placeholder="https://example.com/..."
           onkeydown="if(event.key==='Enter') checkURL()">
    <button id="checkBtn" onclick="checkURL()">Analyse</button>
  </div>

  <div class="loader" id="loader">
    <div class="spinner"></div>
    Analysing URL…
  </div>

  <div class="result-card" id="resultCard">
    <div class="result-header">
      <div class="result-icon" id="resultIcon"></div>
      <div>
        <div class="result-title" id="resultTitle"></div>
        <div class="result-subtitle" id="resultSub"></div>
      </div>
    </div>
    <div class="conf-label">
      <span>Confidence</span>
      <span id="confPct"></span>
    </div>
    <div class="conf-track">
      <div class="conf-fill" id="confFill" style="width:0%"></div>
    </div>
    <div class="flags-title">Risk signals detected</div>
    <div class="flags" id="flags"></div>
  </div>

  <div class="examples">
    <p>Try an example URL →</p>
    <div class="example-chips">
      <span class="chip" onclick="setURL('https://www.google.com')">google.com</span>
      <span class="chip" onclick="setURL('http://paypal-login-verify.xyz/secure/account')">paypal-login-verify.xyz</span>
      <span class="chip" onclick="setURL('https://github.com/login')">github.com/login</span>
      <span class="chip" onclick="setURL('http://192.168.1.1/banking/update')">IP-based URL</span>
      <span class="chip" onclick="setURL('https://amazon.com/deals')">amazon.com</span>
      <span class="chip" onclick="setURL('http://bit.ly/3freeprize')">bit.ly shortener</span>
    </div>
  </div>
</div>

<div class="footer">
 Laaibah Tazaeen B23CS089
</div>

<script>
function setURL(url) {
  document.getElementById('urlInput').value = url;
  checkURL();
}

async function checkURL() {
  const url = document.getElementById('urlInput').value.trim();
  if (!url) {
    document.getElementById('urlInput').focus();
    return;
  }

  const btn  = document.getElementById('checkBtn');
  const card = document.getElementById('resultCard');
  const loader = document.getElementById('loader');

  btn.disabled = true;
  card.style.display = 'none';
  loader.style.display = 'block';

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await resp.json();

    loader.style.display = 'none';

    const isPhish = data.prediction === 'Phishing';
    card.className = 'result-card ' + (isPhish ? 'phishing' : 'legitimate');

    document.getElementById('resultIcon').textContent  = isPhish ? '⚠️' : '✅';
    document.getElementById('resultTitle').textContent =
      isPhish ? 'PHISHING URL DETECTED' : 'LEGITIMATE — Appears Safe';
    document.getElementById('resultSub').textContent   =
      isPhish
        ? 'This URL shows signs of being a phishing attempt.'
        : 'No significant phishing indicators found.';

    document.getElementById('confPct').textContent = data.confidence + '%';
    document.getElementById('confFill').style.width = data.confidence + '%';

    const flagsEl = document.getElementById('flags');
    if (data.flags && data.flags.length > 0) {
      flagsEl.innerHTML = data.flags
        .map(f => `<span class="flag">⚡ ${f}</span>`).join('');
    } else {
      flagsEl.innerHTML = '<span class="no-flags">✓ No risk signals found</span>';
    }

    card.style.display = 'block';
  } catch(e) {
    loader.style.display = 'none';
    alert('Error connecting to server. Please try again.');
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
#  Flask routes
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    url = request.json.get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    label, confidence, prob, flags = predict_url(url)
    return jsonify({
        'prediction': label,
        'confidence': confidence,
        'probability': round(prob, 4),
        'flags': flags,
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5008))
    try:
        from waitress import serve
        print(f"🚀  Starting production server at http://localhost:{port}")
        print("    Press Ctrl+C to stop.")
        serve(app, host='0.0.0.0', port=port)
    except ImportError:
        print("⚠️  waitress not installed — using Flask dev server.")
        print("    Install it for production: pip install waitress")
        print(f"🚀  http://localhost:{port}")
        app.run(host='0.0.0.0', port=port, debug=False)