import re, math, pickle, os, time, socket
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

# ── Load artefacts ────────────────────────────────────────────
MODEL_DIR   = 'model'
model       = pickle.load(open(f'{MODEL_DIR}/lightgbm_model.pkl',  'rb'))
ngram_vec   = pickle.load(open(f'{MODEL_DIR}/ngram_vectorizer.pkl', 'rb'))
tfidf_vec   = pickle.load(open(f'{MODEL_DIR}/tfidf_vectorizer.pkl', 'rb'))

TRAINED_COLS = model.feature_name_
NGRAM_COLS   = [f'ng_{i}' for i in range(len(ngram_vec.vocabulary_))]
TFIDF_COLS   = [f'tf_{i}' for i in range(len(tfidf_vec.vocabulary_))]

SUSPICIOUS_WORDS = [
    'login','verify','secure','account','update','banking','confirm',
    'signin','password','credit','paypal','ebay','amazon','apple',
    'microsoft','google','free','lucky','bonus','winner','click',
    'here','redirect','link','webscr','cmd','dispatch','session',
]
SHORTENERS = ['bit.ly','tinyurl','goo.gl','ow.ly','t.co','tiny.cc']

_metrics_cache = {}


# ─────────────────────────────────────────────────────────────
#  Feature extraction
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
        f['domain_length'] = f['num_subdomains'] = f['has_www'] = f['domain_digits'] = f['domain_hyphens'] = 0
    f['url_depth']   = url.count('/')
    f['path_length'] = len(url.split('?')[0])
    n = len(url) + 1
    f['digit_ratio']   = f['num_digits']  / n
    f['letter_ratio']  = f['num_letters'] / n
    f['special_ratio'] = f['num_special'] / n
    f['dot_ratio']     = f['num_dots']    / n
    f['hyphen_ratio']  = f['num_hyphens'] / n
    lo = url.lower()
    f['has_suspicious']   = 1 if any(w in lo for w in SUSPICIOUS_WORDS) else 0
    f['suspicious_count'] = sum(1 for w in SUSPICIOUS_WORDS if w in lo)
    prob             = [float(url.count(c)) / len(url) for c in set(url)]
    f['url_entropy'] = -sum(p * math.log(p + 1e-10) for p in prob)
    words = [w for w in re.split(r'\W+', url) if w]
    f['longest_word']    = max((len(w) for w in words), default=0)
    f['avg_word_length'] = float(np.mean([len(w) for w in words])) if words else 0.0
    return f


def predict_url(url: str):
    feat_df  = pd.DataFrame([extract_features(url)])
    ngram_df = pd.DataFrame(ngram_vec.transform([url]).toarray(), columns=NGRAM_COLS)
    tfidf_df = pd.DataFrame(tfidf_vec.transform([url]).toarray(), columns=TFIDF_COLS)
    combined = pd.concat([feat_df, ngram_df, tfidf_df], axis=1)
    for col in TRAINED_COLS:
        if col not in combined.columns: combined[col] = 0
    combined = combined[TRAINED_COLS]
    prob  = float(model.predict_proba(combined)[0][1])
    label = 'Phishing' if prob >= 0.5 else 'Legitimate'
    conf  = round(prob * 100 if label == 'Phishing' else (1 - prob) * 100, 1)
    flags = []
    if feat_df['has_ip'].iloc[0]:             flags.append('Contains IP address')
    if feat_df['has_at'].iloc[0]:             flags.append('Contains @ symbol')
    if not feat_df['has_https'].iloc[0]:      flags.append('No HTTPS')
    if feat_df['has_encoded'].iloc[0]:        flags.append('URL-encoded characters')
    if feat_df['has_shortener'].iloc[0]:      flags.append('URL shortener detected')
    if feat_df['has_double_slash'].iloc[0]:   flags.append('Double slash in path')
    if feat_df['has_port'].iloc[0]:           flags.append('Non-standard port')
    sc = int(feat_df['suspicious_count'].iloc[0])
    if sc > 0:                                flags.append(f'{sc} suspicious keyword(s)')
    if feat_df['url_length'].iloc[0] > 100:   flags.append('Unusually long URL')
    if feat_df['num_subdomains'].iloc[0] > 3: flags.append('Excessive subdomains')
    return label, conf, round(prob, 4), flags


def compute_metrics():
    global _metrics_cache
    if _metrics_cache:
        return _metrics_cache

    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score,
                                  confusion_matrix, roc_auc_score, roc_curve)
    from sklearn.linear_model import LogisticRegression

    print("Computing dashboard metrics (first time only)…")
    test_df = pd.read_csv('dataset/test_features.csv')
    X_test  = test_df.drop('label', axis=1)
    y_test  = test_df['label'].values

    for col in TRAINED_COLS:
        if col not in X_test.columns: X_test[col] = 0
    X_test = X_test.drop(columns=list(set(X_test.columns)-set(TRAINED_COLS)))[TRAINED_COLS]

    y_prob_lgb = model.predict_proba(X_test)[:, 1]
    y_pred_lgb = (y_prob_lgb >= 0.5).astype(int)

    lgb_acc  = round(accuracy_score(y_test, y_pred_lgb)  * 100, 2)
    lgb_prec = round(precision_score(y_test, y_pred_lgb) * 100, 2)
    lgb_rec  = round(recall_score(y_test, y_pred_lgb)    * 100, 2)
    lgb_f1   = round(f1_score(y_test, y_pred_lgb)        * 100, 2)
    lgb_auc  = round(roc_auc_score(y_test, y_prob_lgb)   * 100, 2)

    hc = [c for c in TRAINED_COLS if not c.startswith(('ng_','tf_'))]
    train_df = pd.read_csv('dataset/train_features.csv')
    lr = LogisticRegression(max_iter=500, n_jobs=-1, random_state=42)
    lr.fit(train_df[hc], train_df['label'].values)
    y_prob_lr = lr.predict_proba(X_test[hc])[:, 1]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)

    lr_acc  = round(accuracy_score(y_test, y_pred_lr)  * 100, 2)
    lr_prec = round(precision_score(y_test, y_pred_lr) * 100, 2)
    lr_rec  = round(recall_score(y_test, y_pred_lr)    * 100, 2)
    lr_f1   = round(f1_score(y_test, y_pred_lr)        * 100, 2)
    lr_auc  = round(roc_auc_score(y_test, y_prob_lr)   * 100, 2)

    fpr_l, tpr_l, _ = roc_curve(y_test, y_prob_lr)
    fpr_g, tpr_g, _ = roc_curve(y_test, y_prob_lgb)
    ix_l = np.linspace(0, len(fpr_l)-1, 120).astype(int)
    ix_g = np.linspace(0, len(fpr_g)-1, 120).astype(int)

    cm = confusion_matrix(y_test, y_pred_lgb).tolist()

    ph_h, edges = np.histogram(y_prob_lgb[y_test==1], bins=50, range=(0,1))
    le_h, _     = np.histogram(y_prob_lgb[y_test==0], bins=50, range=(0,1))
    bins = ((edges[:-1]+edges[1:])/2).round(3).tolist()

    fi = pd.Series(model.feature_importances_, index=model.feature_name_).nlargest(20)
    fi_labels = [('🔬 ' if not n.startswith(('ng_','tf_')) else '🔤 ')+n for n in fi.index[::-1]]
    fi_values = fi.values[::-1].tolist()

    t0 = time.perf_counter()
    model.predict(X_test.iloc[:500]); lgb_ms=round((time.perf_counter()-t0)/500*1000,3)
    t0 = time.perf_counter()
    lr.predict(X_test[hc].iloc[:500]); lr_ms=round((time.perf_counter()-t0)/500*1000,3)

    lgb_mb = round(os.path.getsize('model/lightgbm_model.pkl')/(1024*1024), 2)

    _metrics_cache = {
        'lgb':  {'acc':lgb_acc,'prec':lgb_prec,'rec':lgb_rec,'f1':lgb_f1,'auc':lgb_auc},
        'lr':   {'acc':lr_acc, 'prec':lr_prec, 'rec':lr_rec, 'f1':lr_f1, 'auc':lr_auc},
        'cm':   cm,
        'roc':  {'lr':  {'fpr':fpr_l[ix_l].round(4).tolist(),'tpr':tpr_l[ix_l].round(4).tolist()},
                 'lgb': {'fpr':fpr_g[ix_g].round(4).tolist(),'tpr':tpr_g[ix_g].round(4).tolist()}},
        'conf_dist': {'bins':bins,'phishing':ph_h.tolist(),'legit':le_h.tolist()},
        'fi':   {'labels':fi_labels,'values':fi_values},
        'perf': {'lgb_ms':lgb_ms,'lr_ms':lr_ms,'lgb_mb':lgb_mb,
                 'lgb_features':len(TRAINED_COLS),'lr_features':len(hc)},
        'dataset': {'test_total':int(len(y_test)),'test_phish':int(y_test.sum()),
                    'test_legit':int(len(y_test)-y_test.sum())},
    }
    print("✓ Metrics cached.")
    return _metrics_cache


# ─────────────────────────────────────────────────────────────
#  HTML template
# ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhishGuard — URL Threat Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --bg:#eef4f6;--bg2:#e4eef2;
  --card:rgba(255,255,255,0.72);--border:rgba(100,140,155,0.18);
  --accent:#3d6b78;--accent2:#5e9aaa;
  --green:#3a7d6b;--red:#c0544a;--gold:#b08a3e;--pink:#7a6a8a;
  --text:#1e3340;--muted:#6b8a96;--muted2:#5a7a88;--r:16px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}

.bg-orbs{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden}
.orb{position:absolute;border-radius:50%;filter:blur(100px);opacity:.18;animation:drift 20s ease-in-out infinite alternate}
.orb1{width:700px;height:700px;background:#b0d0da;top:-250px;left:-250px}
.orb2{width:550px;height:550px;background:#c8dfe6;bottom:-200px;right:-200px;animation-delay:-8s}
.orb3{width:400px;height:400px;background:#d6e8ed;top:35%;left:35%;animation-delay:-14s;opacity:.13}
@keyframes drift{from{transform:translate(0,0) scale(1)}to{transform:translate(50px,35px) scale(1.1)}}

.layout{display:flex;min-height:100vh;position:relative;z-index:1}

.sidebar{
  width:230px;flex-shrink:0;
  background:rgba(225,238,244,0.92);border-right:1px solid var(--border);
  backdrop-filter:blur(24px);padding:24px 14px;
  display:flex;flex-direction:column;gap:6px;
  position:sticky;top:0;height:100vh;overflow-y:auto;
}

.hero-icon{
  display:inline-flex;align-items:center;justify-content:center;
  width:80px;height:80px;color:#3d6b78;
  margin-bottom:1rem;margin-left:auto;
}
.hero-svg{width:100%;height:100%}

.logo{display:flex;align-items:center;gap:12px;padding:4px 10px 22px;border-bottom:1px solid var(--border);margin-bottom:6px}
.logo-icon{font-size:28px;filter:drop-shadow(0 0 14px rgba(61,107,120,.45))}
.logo-text{font-size:15px;font-weight:800;
  background:linear-gradient(135deg,#3d6b78,#5e9aaa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.logo-sub{font-size:10px;color:var(--muted);-webkit-text-fill-color:var(--muted);margin-top:1px}

.nav-section{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);padding:14px 10px 6px}
.nav-btn{
  display:flex;align-items:center;gap:11px;padding:11px 14px;
  border-radius:10px;background:transparent;border:none;
  color:var(--muted2);cursor:pointer;font-size:13px;font-weight:500;
  transition:all .2s;text-align:left;width:100%;font-family:inherit;
}
.nav-btn:hover{background:rgba(61,107,120,.09);color:var(--text)}
.nav-btn.active{background:linear-gradient(135deg,rgba(61,107,120,.18),rgba(94,154,170,.12));
  color:var(--text);border:1px solid rgba(61,107,120,.25)}
.nav-icon{font-size:15px;width:20px;text-align:center}

.sidebar-footer{margin-top:auto;padding:16px 10px 0;border-top:1px solid var(--border)}
.sidebar-footer p{font-size:11px;color:var(--muted);line-height:1.7}

.main{flex:1;overflow-y:auto;padding:36px 40px}

.page{display:none;animation:fadeUp .3s ease}
.page.active{display:block}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

.page-header{margin-bottom:30px}
.page-header h1{font-size:24px;font-weight:800;margin-bottom:5px}
.page-header p{color:var(--muted2);font-size:13px}

.glass{background:rgba(255,255,255,0.65);border:1px solid var(--border);border-radius:var(--r);backdrop-filter:blur(12px)}

.checker-wrap{max-width:660px;margin:0 auto;padding-top:10px}
.hero{text-align:center;margin-bottom:36px}
.hero h1{font-size:34px;font-weight:800;line-height:1.2;margin-bottom:12px;
  background:linear-gradient(135deg,#2e5560,#5e9aaa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:var(--muted2);font-size:15px;line-height:1.6}

.input-card{padding:28px 30px}
.input-row{display:flex;gap:10px}
.url-input{
  flex:1;padding:15px 18px;
  background:rgba(255,255,255,.55);border:1.5px solid var(--border);
  border-radius:12px;color:var(--text);font-size:14px;font-family:inherit;
  outline:none;transition:border-color .2s,background .2s;
}
.url-input:focus{border-color:var(--accent);background:rgba(61,107,120,.06)}
.url-input::placeholder{color:var(--muted)}

.btn-check{
  padding:15px 26px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;border:none;border-radius:12px;
  font-size:14px;font-weight:700;font-family:inherit;
  cursor:pointer;transition:all .2s;white-space:nowrap;
  box-shadow:0 4px 20px rgba(61,107,120,.28);
}
.btn-check:hover{opacity:.88;transform:translateY(-2px);box-shadow:0 8px 28px rgba(61,107,120,.38)}
.btn-check:active{transform:translateY(0)}
.btn-check:disabled{opacity:.45;cursor:not-allowed;transform:none;box-shadow:none}

.loader-row{display:none;align-items:center;gap:12px;padding:18px 2px 0;color:var(--muted2);font-size:13px}
.spin{width:20px;height:20px;border:2px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}

.result-card{
  margin-top:16px;border-radius:14px;padding:24px 26px;
  display:none;animation:fadeUp .3s ease;border:1px solid;
}
.result-card.phishing{background:rgba(192,84,74,.07);border-color:rgba(192,84,74,.2)}
.result-card.legit   {background:rgba(58,125,107,.07);border-color:rgba(58,125,107,.2)}

.r-header{display:flex;align-items:flex-start;gap:14px;margin-bottom:20px}
.r-icon{font-size:40px;line-height:1;flex-shrink:0}
.r-title{font-size:18px;font-weight:700;margin-bottom:4px}
.phishing .r-title{color:var(--red)}.legit .r-title{color:var(--green)}
.r-sub{font-size:12px;color:var(--muted2);line-height:1.5}

.conf-row{display:flex;justify-content:space-between;align-items:baseline;
  font-size:12px;color:var(--muted2);margin-bottom:8px}
.conf-row strong{font-size:22px;font-weight:800;color:var(--text)}

.conf-track{height:6px;background:rgba(61,107,120,.12);border-radius:99px;
  overflow:hidden;margin-bottom:20px}
.conf-fill{height:100%;border-radius:99px;transition:width .8s cubic-bezier(.4,0,.2,1)}
.phishing .conf-fill{background:linear-gradient(90deg,#c0544a,#d97060)}
.legit    .conf-fill{background:linear-gradient(90deg,#3a7d6b,#5aab96)}

.flags-label{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--muted);margin-bottom:10px}
.flags{display:flex;flex-wrap:wrap;gap:7px}
.flag{background:rgba(192,84,74,.08);border:1px solid rgba(192,84,74,.18);
  color:#a04040;border-radius:20px;padding:4px 13px;font-size:11px;font-weight:500}
.no-flag{font-size:12px;color:var(--green);opacity:.9}

.examples{margin-top:22px;padding-top:20px;border-top:1px solid var(--border)}
.examples-label{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--muted);margin-bottom:10px}
.chips{display:flex;flex-wrap:wrap;gap:7px}
.chip{background:rgba(255,255,255,.5);border:1px solid var(--border);
  border-radius:20px;padding:5px 13px;font-size:11px;color:var(--muted2);
  cursor:pointer;transition:all .15s;font-family:monospace}
.chip:hover{background:rgba(61,107,120,.1);border-color:rgba(61,107,120,.4);color:var(--text)}

.stats-row{display:flex;gap:12px;margin-top:16px}
.stat-pill{flex:1;padding:14px 10px;border-radius:12px;text-align:center}
.stat-pill .val{font-size:20px;font-weight:800;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.stat-pill .lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-top:3px}

.dash-loading{display:flex;flex-direction:column;align-items:center;
  justify-content:center;min-height:55vh;gap:16px;color:var(--muted2)}
.dash-spinner{width:50px;height:50px;border:3px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
.dash-spinner-sub{font-size:13px;text-align:center;line-height:1.7}

.kpi-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:26px}
.kpi{padding:18px 14px;border-radius:14px;text-align:center;position:relative;overflow:hidden;cursor:default}
.kpi::before{content:'';position:absolute;inset:0;opacity:.1;background:var(--kc)}
.kpi-val{font-size:24px;font-weight:800;color:var(--kc);position:relative}
.kpi-lbl{font-size:10px;color:var(--muted2);text-transform:uppercase;letter-spacing:.07em;margin-top:4px;position:relative}
.kpi-delta{font-size:10px;margin-top:5px;position:relative;font-weight:600}
.kpi-delta.up{color:var(--green)}.kpi-delta.down{color:var(--red)}

.section-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);margin:26px 0 14px;padding-left:2px}

.chart-grid{display:grid;gap:18px}
.g2{grid-template-columns:1fr 1fr}
.g32{grid-template-columns:2fr 1fr}

.chart-card{padding:22px 20px}
.chart-title{font-size:14px;font-weight:700;margin-bottom:3px}
.chart-sub{font-size:11px;color:var(--muted);margin-bottom:16px}

@media(max-width:1000px){.sidebar{width:200px}.main{padding:24px}}
@media(max-width:800px){
  .sidebar{display:none}.main{padding:20px 16px}
  .kpi-grid{grid-template-columns:repeat(3,1fr)}
  .g2,.g32{grid-template-columns:1fr}
}
@media(max-width:480px){
  .kpi-grid{grid-template-columns:repeat(2,1fr)}
  .stats-row{flex-wrap:wrap}
  .stat-pill{min-width:calc(50% - 6px)}
}
</style>
</head>
<body>
<div class="bg-orbs">
  <div class="orb orb1"></div><div class="orb orb2"></div><div class="orb orb3"></div>
</div>

<div class="layout">
  <nav class="sidebar">
    <div class="logo">
      <span class="logo-icon"></span>
      <div><div class="logo-text">PhishGuard</div><div class="logo-sub">LightGBM · v2.0</div></div>
    </div>
    <div class="nav-section">Tools</div>
    <button class="nav-btn active" id="nav-checker" onclick="showPage('checker')">
      <span class="nav-icon"></span>URL Checker
    </button>
    <button class="nav-btn" id="nav-dashboard" onclick="showPage('dashboard')">
      <span class="nav-icon"></span>Analytics Dashboard
    </button>
    <div class="sidebar-footer">
      <p>Trained on 520,285 real URLs<br>LightGBM + 460 features</p>
    </div>
  </nav>

  <main class="main">

    <div class="page active" id="page-checker">
      <div class="checker-wrap">
        <div class="hero">
          <h1>Detect phishing URLs<br>in milliseconds</h1>
          <p>Paste any URL below. Our LightGBM model will analyse<br>460+ features and classify it instantly.</p>
        </div>

        <div class="glass input-card">
          <div class="input-row">
            <input class="url-input" type="text" id="urlInput"
                   placeholder="Paste a URL to check…"
                   onkeydown="if(event.key==='Enter')checkURL()">
            <button class="btn-check" id="checkBtn" onclick="checkURL()">Analyse →</button>
          </div>

          <div class="loader-row" id="loaderRow">
            <div class="spin"></div>Analysing with 460+ features…
          </div>

          <div class="result-card" id="resultCard">
            <div class="r-header">
              <div class="r-icon" id="rIcon"></div>
              <div>
                <div class="r-title" id="rTitle"></div>
                <div class="r-sub"   id="rSub"></div>
              </div>
            </div>
            <div class="conf-row">
              <span>Model confidence</span>
              <strong id="confPct"></strong>
            </div>
            <div class="conf-track"><div class="conf-fill" id="confFill" style="width:0%"></div></div>
            <div class="flags-label">Risk signals detected</div>
            <div class="flags" id="flagsList"></div>
          </div>

          <div class="examples">
            <div class="examples-label">Try an example</div>
            <div class="chips">
              <span class="chip" onclick="setURL('https://www.google.com')">google.com ✓</span>
              <span class="chip" onclick="setURL('http://paypal-login-verify.xyz/secure/account/update')">paypal-login-verify.xyz ⚠</span>
              <span class="chip" onclick="setURL('https://github.com/login')">github.com ✓</span>
              <span class="chip" onclick="setURL('http://192.168.1.1/banking/signin?update=true')">IP address ⚠</span>
              <span class="chip" onclick="setURL('http://bit.ly/3freewinnerprize')">bit.ly shortener ⚠</span>
              <span class="chip" onclick="setURL('https://amazon.com/deals/today')">amazon.com ✓</span>
            </div>
          </div>
        </div>

        <div class="stats-row">
          <div class="glass stat-pill"><div class="val">520K+</div><div class="lbl">Training URLs</div></div>
          <div class="glass stat-pill"><div class="val">95%</div><div class="lbl">Accuracy</div></div>
          <div class="glass stat-pill"><div class="val">460+</div><div class="lbl">Features</div></div>
          <div class="glass stat-pill"><div class="val">&lt;2ms</div><div class="lbl">Per URL</div></div>
          
        </div>
      </div>
       <span style="display:block;text-align:center;color:var(--muted);font-size:12px;margin-top:18px"> Made by Laaibah Tazaeen (B23CS089) </span>
    </div>

    <div class="page" id="page-dashboard">
      <div class="page-header">
        <h1>Analytics Dashboard</h1>
        <p>All charts are rendered live from real predictions on the held-out test set (51,510 URLs).</p>
      </div>

      <div id="dashLoading" class="dash-loading">
        <div class="dash-spinner"></div>
        <div class="dash-spinner-sub">
          Computing predictions on 51K test URLs…<br>Training Logistic Regression baseline…<br>
          <small style="color:var(--muted)">This runs once and is cached for the session.</small>
        </div>
      </div>

      <div id="dashContent" style="display:none">
        <div class="section-label">LightGBM — test set KPIs (vs logistic regression baseline)</div>
        <div class="kpi-grid" id="kpiGrid"></div>

        <div class="section-label">Performance breakdown</div>
        <div class="chart-grid g32">
          <div class="glass chart-card">
            <div class="chart-title">All-Metrics Comparison</div>
            <div class="chart-sub">LightGBM vs Logistic Regression across 5 metrics</div>
            <canvas id="cMetrics" height="200"></canvas>
          </div>
          <div class="glass chart-card">
            <div class="chart-title">Confusion Matrix</div>
            <div class="chart-sub">True vs predicted labels</div>
            <canvas id="cMatrix" height="200"></canvas>
          </div>
        </div>

        <div class="section-label">Prediction quality</div>
        <div class="chart-grid g2">
          <div class="glass chart-card">
            <div class="chart-title">ROC Curve</div>
            <div class="chart-sub">TPR vs FPR — higher AUC = better discrimination</div>
            <canvas id="cRoc" height="240"></canvas>
          </div>
          <div class="glass chart-card">
            <div class="chart-title">Confidence Score Distribution</div>
            <div class="chart-sub">Well-separated peaks = model is confident and calibrated</div>
            <canvas id="cConf" height="240"></canvas>
          </div>
        </div>

        <div class="section-label">Model internals</div>
        <div class="chart-grid g32">
          <div class="glass chart-card">
            <div class="chart-title">Top 20 Feature Importances</div>
            <div class="chart-sub">hand-crafted features (teal) · n-gram / TF-IDF (slate)</div>
            <canvas id="cFI" height="320"></canvas>
          </div>
          <div class="glass chart-card">
            <div class="chart-title">System Performance</div>
            <div class="chart-sub">Speed, size, and feature count vs baseline</div>
            <canvas id="cPerf" height="320"></canvas>
          </div>
        </div>
      </div>
      <span style="display:block;text-align:center;color:var(--muted);font-size:12px;margin-top:18px"> Made by Laaibah Tazaeen (B23CS089) </span>
    </div>

  </main>
</div>

<script>
Chart.defaults.color='#5a7a88'
Chart.defaults.font.family="'Inter',sans-serif"
const G=()=>({color:'rgba(61,107,120,0.1)',drawBorder:false})

function showPage(name){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'))
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'))
  document.getElementById('page-'+name).classList.add('active')
  document.getElementById('nav-'+name).classList.add('active')
  if(name==='dashboard'&&!window._dl)loadDashboard()
}

function setURL(u){document.getElementById('urlInput').value=u;checkURL()}
async function checkURL(){
  const url=document.getElementById('urlInput').value.trim()
  if(!url){document.getElementById('urlInput').focus();return}
  const btn=document.getElementById('checkBtn')
  const loader=document.getElementById('loaderRow')
  const card=document.getElementById('resultCard')
  btn.disabled=true;card.style.display='none';loader.style.display='flex'
  try{
    const r=await fetch('/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({url})})
    const d=await r.json()
    loader.style.display='none'
    const isP=d.prediction==='Phishing'
    card.className='result-card '+(isP?'phishing':'legit')
    document.getElementById('rIcon').textContent=isP?'⚠️':''
    document.getElementById('rTitle').textContent=isP?'PHISHING — High Risk Detected':'LEGITIMATE — Appears Safe'
    document.getElementById('rSub').textContent=isP
      ?'This URL exhibits strong phishing indicators. Do not visit this link.'
      :'No significant phishing signals detected. This URL appears safe.'
    document.getElementById('confPct').textContent=d.confidence+'%'
    document.getElementById('confFill').style.width=d.confidence+'%'
    const fl=document.getElementById('flagsList')
    fl.innerHTML=d.flags&&d.flags.length
      ?d.flags.map(f=>`<span class="flag">⚡ ${f}</span>`).join('')
      :'<span class="no-flag">✓ No risk signals detected</span>'
    card.style.display='block'
  }catch(e){loader.style.display='none';alert('Server error — please retry.')}
  finally{btn.disabled=false}
}

window._dl=false
async function loadDashboard(){
  window._dl=true
  const resp=await fetch('/api/metrics')
  const d=await resp.json()
  document.getElementById('dashLoading').style.display='none'
  document.getElementById('dashContent').style.display='block'
  buildKPIs(d);buildMetrics(d);buildMatrix(d);buildROC(d);buildConf(d);buildFI(d);buildPerf(d)
}

function buildKPIs(d){
  const m=d.lgb,b=d.lr
  const items=[
    {lbl:'Accuracy', val:m.acc,base:b.acc,kc:'#3d6b78'},
    {lbl:'Precision',val:m.prec,base:b.prec,kc:'#5e9aaa'},
    {lbl:'Recall',   val:m.rec,base:b.rec,kc:'#3a7d6b'},
    {lbl:'F1 Score', val:m.f1,base:b.f1,kc:'#b08a3e'},
    {lbl:'ROC-AUC',  val:m.auc,base:b.auc,kc:'#7a6a8a'},
  ]
  document.getElementById('kpiGrid').innerHTML=items.map(i=>{
    const diff=(i.val-i.base).toFixed(2),up=parseFloat(diff)>=0
    return `<div class="glass kpi" style="--kc:${i.kc}">
      <div class="kpi-val">${i.val}%</div>
      <div class="kpi-lbl">${i.lbl}</div>
      <div class="kpi-delta ${up?'up':'down'}">${up?'+':''}${diff}% vs LR</div>
    </div>`
  }).join('')
}

function buildMetrics(d){
  const m=d.lgb,b=d.lr,labels=['Accuracy','Precision','Recall','F1','AUC']
  const mn=Math.floor(Math.min(b.acc,b.prec,b.rec,b.f1,b.auc)-4)
  new Chart(document.getElementById('cMetrics'),{type:'bar',
    data:{labels,datasets:[
      {label:'Logistic Regression',data:[b.acc,b.prec,b.rec,b.f1,b.auc],
       backgroundColor:'rgba(94,154,170,.35)',borderRadius:5,borderSkipped:false},
      {label:'LightGBM (yours)',data:[m.acc,m.prec,m.rec,m.f1,m.auc],
       backgroundColor:'rgba(61,107,120,.75)',borderRadius:5,borderSkipped:false},
    ]},
    options:{responsive:true,plugins:{legend:{position:'bottom',labels:{boxWidth:12,padding:16}}},
      scales:{y:{min:mn,grid:G(),ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}}}
  })
}

function buildMatrix(d){
  const cm=d.cm
  new Chart(document.getElementById('cMatrix'),{type:'bar',
    data:{
      labels:['True Legit\n(TN)','False Phish\n(FP)','False Legit\n(FN)','True Phish\n(TP)'],
      datasets:[{
        data:[cm[0][0],cm[0][1],cm[1][0],cm[1][1]],
        backgroundColor:['rgba(58,125,107,.65)','rgba(192,84,74,.65)',
                         'rgba(192,84,74,.65)','rgba(58,125,107,.65)'],
        borderRadius:6,borderSkipped:false,
      }]
    },
    options:{responsive:true,plugins:{legend:{display:false},
      tooltip:{callbacks:{label:ctx=>'Count: '+ctx.parsed.y.toLocaleString()}}},
      scales:{y:{grid:G(),ticks:{callback:v=>v>=1000?(v/1000).toFixed(1)+'K':v}},
              x:{grid:{display:false},ticks:{font:{size:10}}}}}
  })
}

function buildROC(d){
  const r=d.roc
  new Chart(document.getElementById('cRoc'),{type:'line',
    data:{datasets:[
      {label:'Random',data:[{x:0,y:0},{x:1,y:1}],
       borderColor:'rgba(107,138,150,.3)',borderDash:[5,5],borderWidth:1.2,pointRadius:0,fill:false},
      {label:`Logistic Reg (AUC ${d.lr.auc}%)`,
       data:r.lr.fpr.map((f,i)=>({x:f,y:r.lr.tpr[i]})),
       borderColor:'rgba(94,154,170,.6)',borderWidth:2,pointRadius:0,fill:false},
      {label:`LightGBM (AUC ${d.lgb.auc}%)`,
       data:r.lgb.fpr.map((f,i)=>({x:f,y:r.lgb.tpr[i]})),
       borderColor:'#3d6b78',borderWidth:2.5,pointRadius:0,
       fill:{target:'origin',above:'rgba(61,107,120,.06)'}},
    ]},
    options:{responsive:true,plugins:{legend:{position:'bottom',labels:{boxWidth:12,padding:14}}},
      scales:{
        x:{type:'linear',min:0,max:1,grid:G(),
           title:{display:true,text:'False Positive Rate',color:'#6b8a96',font:{size:11}}},
        y:{min:0,max:1,grid:G(),
           title:{display:true,text:'True Positive Rate',color:'#6b8a96',font:{size:11}}},
      }}
  })
}

function buildConf(d){
  const c=d.conf_dist
  new Chart(document.getElementById('cConf'),{type:'bar',
    data:{labels:c.bins,datasets:[
      {label:'Legitimate URLs',data:c.legit,
       backgroundColor:'rgba(58,125,107,.55)',borderRadius:2},
      {label:'Phishing URLs', data:c.phishing,
       backgroundColor:'rgba(192,84,74,.55)',borderRadius:2},
    ]},
    options:{responsive:true,plugins:{legend:{position:'bottom',labels:{boxWidth:12,padding:14}}},
      scales:{
        x:{grid:{display:false},ticks:{maxTicksLimit:6,callback:v=>parseFloat(v).toFixed(1)},
           title:{display:true,text:'Predicted phishing probability',color:'#6b8a96',font:{size:11}}},
        y:{grid:G(),title:{display:true,text:'URL count',color:'#6b8a96',font:{size:11}}}
      }}
  })
}

function buildFI(d){
  const f=d.fi
  const cols=f.labels.map(l=>l.startsWith('🔬')?'rgba(94,154,170,.8)':'rgba(61,107,120,.75)')
  new Chart(document.getElementById('cFI'),{type:'bar',
    data:{labels:f.labels,datasets:[{data:f.values,backgroundColor:cols,borderRadius:4,borderSkipped:false}]},
    options:{indexAxis:'y',responsive:true,plugins:{legend:{display:false}},
      scales:{
        x:{grid:G(),title:{display:true,text:'Importance score',color:'#6b8a96',font:{size:11}}},
        y:{grid:{display:false},ticks:{font:{size:10}}},
      }}
  })
}

function buildPerf(d){
  const p=d.perf
  new Chart(document.getElementById('cPerf'),{type:'bar',
    data:{
      labels:['Inference\n(ms/sample)','Model Size\n(MB)','Features\nUsed'],
      datasets:[
        {label:'Logistic Regression',data:[p.lr_ms,0.5,p.lr_features],
         backgroundColor:'rgba(94,154,170,.4)',borderRadius:5,borderSkipped:false},
        {label:'LightGBM',data:[p.lgb_ms,p.lgb_mb,p.lgb_features],
         backgroundColor:'rgba(61,107,120,.75)',borderRadius:5,borderSkipped:false},
      ]
    },
    options:{responsive:true,plugins:{legend:{position:'bottom',labels:{boxWidth:12,padding:14}}},
      scales:{y:{grid:G()},x:{grid:{display:false}}}}
  })
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
    label, conf, prob, flags = predict_url(url)
    return jsonify({'prediction':label,'confidence':conf,'probability':prob,'flags':flags})

@app.route('/api/metrics')
def api_metrics():
    try:
        return jsonify(compute_metrics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     def find_free_port(start):
#         for p in range(start, start + 20):
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#                 try:
#                     s.bind(('', p)); return p
#                 except OSError:
#                     continue
#         return start

#     preferred = int(os.environ.get('PORT', 5008))
#     port = find_free_port(preferred)
#     if port != preferred:
#         print(f"⚠️  Port {preferred} busy — using {port}")

#     try:
#         from waitress import serve
#         print(f"\n  PhishGuard → http://localhost:{port}")
#         print(f"    Dashboard → click 'Analytics Dashboard' in the sidebar")
#         print("    Ctrl+C to stop.\n")
#         serve(app, host='0.0.0.0', port=port)
#     except ImportError:
#         print(f"\n  http://localhost:{port}  (install waitress for production)")
#         app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    def find_free_port(start):
        for p in range(start, start + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', p)); return p
                except OSError:
                    continue
        return start

    preferred = int(os.environ.get('PORT', 5008))
    port = find_free_port(preferred)

    print(f"\n PhishGuard → http://localhost:{port}")
    print(" Auto reload enabled\n")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=True,     
        use_reloader=True
    )