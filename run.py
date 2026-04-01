import subprocess
import sys
import importlib

required = [
    "lightgbm",
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "seaborn"
]

for pkg in required:
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

scripts = [
    "1_preprocess.py",
    "2_features.py",
    "3_train.py",
    "4_evaluate.py",
    "5_app.py"
]

for script in scripts:
    print(f"\nRunning {script}...")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"Error while running {script}")
        sys.exit(1)

print("\nAll scripts executed successfully!")