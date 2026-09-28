"""
download_data.py
----------------
Downloads the Heart Disease dataset from a public source and saves it to data/heart.csv.
Run this once before training if you don't have the dataset.

Usage:
    python download_data.py
"""

import urllib.request
from pathlib import Path

# Mirror of the UCI Heart Disease dataset (Cleveland) hosted on GitHub
DATA_URL = (
    "https://raw.githubusercontent.com/dsrscientist/"
    "dataset1/master/heart_disease.csv"
)

# Fallback: a well-known copy used in many ML tutorials
FALLBACK_URL = (
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/"
    "master/dataset.csv"
)

SAVE_PATH = Path("data/heart.csv")


def download():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    urls = [DATA_URL, FALLBACK_URL]
    for url in urls:
        try:
            print(f"[INFO] Downloading dataset from:\n  {url}")
            urllib.request.urlretrieve(url, SAVE_PATH)
            print(f"[INFO] Dataset saved to '{SAVE_PATH}'")
            return
        except Exception as e:
            print(f"[WARN] Failed: {e}. Trying next URL...")

    print("[ERROR] All download attempts failed.")
    print("Please manually download heart.csv from:")
    print("  https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
    print(f"and place it at '{SAVE_PATH}'")


if __name__ == "__main__":
    download()
