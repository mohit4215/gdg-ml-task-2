"""
feature_engineering.py
-----------------------
Handles feature scaling and selection for the Heart Disease dataset.

Why this module?
    Feature engineering is separated so it can be applied consistently to
    both training and inference data — the scaler must be fit ONLY on
    training data, then applied to test/production data to prevent
    data leakage.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# Features to scale (all continuous/ordinal numeric features)
FEATURES_TO_SCALE = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

# All expected feature columns (in order)
ALL_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: str = "models/scaler.pkl",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit a StandardScaler on training data and transform both train and test sets.

    Why StandardScaler?
        Logistic Regression uses gradient descent internally. Features on
        different scales (e.g., age: 30–80 vs cholesterol: 100–600) make
        optimization slow and biased. StandardScaler normalises each feature
        to mean=0, std=1, ensuring equal contribution.

    Why fit ONLY on training data?
        Fitting on test data would leak information about the test set into
        the model — a form of data leakage. The test set must simulate
        truly unseen data.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        scaler_path: Path to save the fitted scaler.

    Returns:
        Scaled X_train and X_test DataFrames.
    """
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Scale only continuous features; binary/categorical stay as-is
    cols_to_scale = [c for c in FEATURES_TO_SCALE if c in X_train.columns]

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # Persist scaler for use at inference time
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Scaler saved to '{scaler_path}'")

    return X_train_scaled, X_test_scaled


def load_scaler(scaler_path: str = "models/scaler.pkl") -> StandardScaler:
    """Load a previously saved StandardScaler from disk."""
    scaler = joblib.load(scaler_path)
    print(f"[INFO] Scaler loaded from '{scaler_path}'")
    return scaler


def scale_single_sample(
    sample: dict,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Scale a single input sample for inference.

    Args:
        sample: Dictionary of feature_name → value.
        scaler: Fitted StandardScaler instance.

    Returns:
        2D numpy array ready for model.predict().
    """
    df = pd.DataFrame([sample])[ALL_FEATURES]
    df[FEATURES_TO_SCALE] = scaler.transform(df[FEATURES_TO_SCALE])
    return df.values
