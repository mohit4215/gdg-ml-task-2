"""
model_training.py
-----------------
Trains a Logistic Regression model with hyperparameter tuning via GridSearchCV.

Why Logistic Regression?
    - Heart disease prediction is a binary classification problem (disease / no disease).
    - Logistic Regression outputs a probability (0–1) via the sigmoid function,
      which is clinically meaningful: "Patient has 78% probability of heart disease."
    - It is highly interpretable: positive coefficients indicate risk factors,
      negative coefficients indicate protective factors.
    - With feature scaling and regularisation (C parameter), it generalises well
      even on small medical datasets (~300 rows).
    - It serves as the ideal baseline before trying complex models like Random Forest or XGBoost.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_path: str = "models/logistic_model.pkl",
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a Logistic Regression model with GridSearchCV hyperparameter tuning.

    Hyperparameters tuned:
        C (inverse of regularisation strength):
            - Small C  → stronger regularisation → simpler model → avoids overfitting
            - Large C  → weaker regularisation  → more complex model → may overfit
        solver:
            - 'liblinear': Efficient for small datasets; supports L1 & L2 regularisation.
            - 'lbfgs': Uses second-order gradient info; good for multi-class & L2.
        penalty:
            - 'l1': Produces sparse coefficients (feature selection effect).
            - 'l2': Distributes penalty across all features; default & more stable.

    Why GridSearchCV with StratifiedKFold?
        Cross-validation evaluates each hyperparameter combo on multiple folds
        to prevent overfitting to a single validation split. Stratified ensures
        class balance is preserved in each fold (critical for medical data).

    Args:
        X_train: Scaled training features.
        y_train: Training labels.
        model_path: Path to save the trained model.
        random_state: Seed for reproducibility.

    Returns:
        Best trained LogisticRegression estimator.
    """
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"],
        "penalty": ["l1", "l2"],
    }

    # Remove incompatible combinations (lbfgs doesn't support L1)
    # GridSearchCV handles this via error_score='raise' — we filter manually
    # to avoid warnings.
    base_lr = LogisticRegression(random_state=random_state, max_iter=1000)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_lr,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",   # ROC-AUC is the best metric for imbalanced medical classification
        n_jobs=-1,            # Use all CPU cores
        verbose=1,
        error_score=0,        # Silently assign 0 for invalid combos (e.g., lbfgs + l1)
    )

    print("[INFO] Starting GridSearchCV for Logistic Regression...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"[INFO] Best parameters: {grid_search.best_params_}")
    print(f"[INFO] Best cross-val ROC-AUC: {grid_search.best_score_:.4f}")

    # Save the best model to disk
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"[INFO] Model saved to '{model_path}'")

    return best_model


def load_model(model_path: str = "models/logistic_model.pkl") -> LogisticRegression:
    """Load a previously saved Logistic Regression model from disk."""
    model = joblib.load(model_path)
    print(f"[INFO] Model loaded from '{model_path}'")
    return model
