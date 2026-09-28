"""
predict.py
----------
Utility for making single-sample predictions using the saved model and scaler.

This module is used by the Streamlit app to run inference on user-provided inputs
without re-training or re-loading the full pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import ALL_FEATURES, FEATURES_TO_SCALE


def predict_single(
    sample: dict,
    model: LogisticRegression,
    scaler: StandardScaler,
) -> tuple[int, float]:
    """
    Predict heart disease for a single patient.

    Pipeline steps:
        1. Convert dict → DataFrame (ensures correct column order)
        2. Apply the same StandardScaler fit on training data
        3. Run model.predict() and model.predict_proba()

    Args:
        sample: Dictionary mapping feature names to their values.
                Example: {"age": 54, "sex": 1, "cp": 2, ...}
        model:  Loaded LogisticRegression model.
        scaler: Loaded StandardScaler (fit on training data).

    Returns:
        (prediction, probability)
            prediction:  0 (No Heart Disease) or 1 (Heart Disease)
            probability: Float in [0, 1] representing confidence for class 1
    """
    # Build a single-row DataFrame with features in the correct order
    df = pd.DataFrame([sample])[ALL_FEATURES]

    # Scale only the continuous features — same transformation as training
    df[FEATURES_TO_SCALE] = scaler.transform(df[FEATURES_TO_SCALE])

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return prediction, probability


def get_risk_label(probability: float) -> tuple[str, str]:
    """
    Map a predicted probability to a human-readable risk level.

    Thresholds:
        < 30%  → Low Risk
        30-60% → Moderate Risk
        > 60%  → High Risk

    Args:
        probability: Model's predicted probability of heart disease.

    Returns:
        (risk_level, emoji) — e.g., ("High Risk", "🔴")
    """
    if probability < 0.30:
        return "Low Risk", "🟢"
    elif probability < 0.60:
        return "Moderate Risk", "🟡"
    else:
        return "High Risk", "🔴"
