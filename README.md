#  Phishing URL Detector

A machine learning web app that detects phishing URLs using LightGBM.  
Trained on 520,000+ real phishing and legitimate URLs.  
Expected accuracy: **97–98%**

---

## 📁 Project Structure

```
phishing_detector/
├── dataset/
│   └── small_dataset/
│       ├── train.txt       ←dataset files 
│       ├── val.txt
│       └── test.txt
├── model/                  ← auto-created during training
├── results/                ← evaluation plots 
├── 1_preprocess.py
├── 2_features.py
├── 3_train.py
├── 4_evaluate.py
├── 5_app.py
├── run.py
└── requirements.txt
```

---

##  Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. dataset files
Put `train.txt`, `val.txt`, `test.txt` inside `dataset/small_dataset/`.  

### 3. Run the full training pipeline
```bash
python run.py
```
This runs preprocessing → feature extraction → training → evaluation
Training takes **5–10 minutes** 

### 4. Start the web app
```bash
python 5_app.py
```
Open **http://localhost:5000** in browser

---

##  Deployment

### Option A — Local (default)
`python 5_app.py` uses **Waitress** (production WSGI server).  
No extra configuration needed.

### Option B — Deploy to Render.com (free)
1. Push your code to GitHub
2. Create a new **Web Service** on render.com
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python 5_app.py`

### Option C — Deploy to Heroku
Create a `Procfile`:
```
web: python 5_app.py
```
Then `heroku create` → `git push heroku main`


---

##  Why Small Dataset?

| | Small (520K) | Big (5.2M) |
|---|---|---|
| Training time | ~5–10 min | ~1–2 hours |
| Accuracy | 97–98% | ~98–99% |
| RAM needed | ~4 GB | ~16+ GB |

The small dataset gives excellent accuracy and trains in minutes. Only use the big dataset if you have a powerful machine and need that last ~1%

---

##  Common Issues

**`FileNotFoundError: dataset/small_dataset/train.txt`**  
→ Make sure `.txt` files are inside `dataset/small_dataset/`

**`MemoryError` during feature extraction**  
→  need to close other apps. The script uses ~3–4 GB RAM during feature extraction

**`ModuleNotFoundError`**  
→ Run `pip install -r requirements.txt` first

**App starts but predictions seem wrong**  
→ Make sure u ran the full pipeline (`python run.py`) before starting the app