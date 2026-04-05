"""
run.py — Install dependencies and run the training pipeline.
After this finishes, start the web app separately with:

    python 5_app.py
"""

import sys
import subprocess
import importlib

# ── Auto-install missing packages ────────────────────────────
REQUIRED = {
    'lightgbm':   'lightgbm',
    'pandas':     'pandas',
    'numpy':      'numpy',
    'sklearn':    'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn':    'seaborn',
    'flask':      'flask',
    'waitress':   'waitress',
}

for import_name, pip_name in REQUIRED.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               pip_name, '--quiet'])

# ── Pipeline scripts (NOT including 5_app.py — that's a server) ──
PIPELINE = [
    '1_preprocess.py',
    '2_features.py',
    '3_train.py',
    '4_evaluate.py',
]

for script in PIPELINE:
    print(f"\n{'='*50}")
    print(f"  Running {script}")
    print('='*50)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n✗ Error in {script} — stopping.")
        sys.exit(1)

print("\n" + "="*50)
print("  ✓ Pipeline complete!")
print("="*50)
print("\nTo start the web app run:")
print("  python 5_app.py")
print("\nThen open http://localhost:5000 in your browser.")