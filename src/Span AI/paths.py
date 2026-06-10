"""
paths.py
========
Location-independent project paths. Walks up from this file to find the project
root (the folder that holds results/ and data/, or is named 'span_benchmark'),
so the scripts run no matter where the source folder is moved to.
"""
import os

def _find_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if (os.path.basename(d) == "span_benchmark"
                or os.path.isdir(os.path.join(d, "results"))
                or os.path.isfile(os.path.join(d, "README.md"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            # fallback: two levels up from src/<here>
            return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d = parent

ROOT = _find_root()
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
