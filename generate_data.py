"""
generate_data.py
----------------
Generates a synthetic but realistic Heart Disease dataset for demonstration purposes.
This mirrors the structure and distributions of the UCI Cleveland Heart Disease dataset.

Run this if you cannot download the real dataset:
    python generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 303  # Same size as the original Cleveland dataset


def generate_heart_disease_data(n: int = 303) -> pd.DataFrame:
    """
    Generate a synthetic heart disease dataset with realistic distributions
    based on the statistical properties of the UCI Cleveland dataset.
    """
    # Ages skewed toward middle-aged and elderly
    age = np.random.normal(54.4, 9.0, n).clip(29, 77).astype(int)

    # Sex: ~68% male in the original dataset
    sex = np.random.binomial(1, 0.68, n)

    # Chest pain type: ordinal 0–3
    cp = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])

    # Resting blood pressure
    trestbps = np.random.normal(131.7, 17.6, n).clip(94, 200).astype(int)

    # Serum cholesterol
    chol = np.random.normal(246.7, 51.8, n).clip(126, 564).astype(int)

    # Fasting blood sugar > 120: ~15%
    fbs = np.random.binomial(1, 0.15, n)

    # Resting ECG: mostly normal (0) or ST-T abnormality (1)
    restecg = np.random.choice([0, 1, 2], n, p=[0.49, 0.48, 0.03])

    # Max heart rate achieved
    thalach = np.random.normal(149.6, 22.9, n).clip(71, 202).astype(int)

    # Exercise-induced angina: ~33%
    exang = np.random.binomial(1, 0.33, n)

    # ST depression
    oldpeak = np.abs(np.random.normal(1.04, 1.16, n)).clip(0, 6.2).round(1)

    # Slope of peak ST segment
    slope = np.random.choice([0, 1, 2], n, p=[0.21, 0.47, 0.32])

    # Number of major vessels
    ca = np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])

    # Thalassemia
    thal = np.random.choice([0, 1, 2, 3], n, p=[0.02, 0.05, 0.55, 0.38])

    # Target: create a logistically-derived target correlated with features
    # Risk score (simplified logistic model) — intercept tuned for ~50% positive rate
    risk_score = (
        0.04 * (age - 54)
        - 0.3 * sex                           # adjusted: being male slightly protective in this synthetic set
        + 0.5 * cp
        - 0.03 * (thalach - 150)
        + 0.5 * exang
        + 0.5 * oldpeak
        + 0.4 * ca
        + 0.4 * (thal == 3).astype(int)
        - 0.3 * (slope == 2).astype(int)
        - 0.5                                 # intercept to shift baseline probability down to ~50%
        + np.random.normal(0, 1.0, n)
    )
    prob = 1 / (1 + np.exp(-risk_score))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
        "thal": thal, "target": target,
    })

    print(f"[INFO] Generated {n} samples")
    print(f"[INFO] Class distribution:\n{df['target'].value_counts()}")
    return df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = generate_heart_disease_data(N)
    df.to_csv("data/heart.csv", index=False)
    print("[INFO] Dataset saved to 'data/heart.csv'")
