"""
model_evaluation.py
-------------------
Evaluates the trained model and generates visualisations.

Why multiple metrics?
    Accuracy alone is misleading for medical problems:
    - A model that predicts "no disease" for everyone achieves ~54% accuracy
      on a balanced dataset — but misses ALL sick patients (catastrophic!).
    - Recall (sensitivity) measures: "Of all actual heart disease patients,
      how many did we correctly identify?" → most important in medical screening.
    - Precision measures: "Of all patients we flagged as sick, how many actually are?"
    - F1-Score balances precision and recall.
    - ROC-AUC measures the model's ability to discriminate between classes
      across all probability thresholds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression


def evaluate_model(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "outputs",
) -> dict:
    """
    Compute all evaluation metrics and save visualisation plots.

    Args:
        model: Trained LogisticRegression model.
        X_test: Scaled test features.
        y_test: True test labels.
        output_dir: Directory to save plots.

    Returns:
        Dictionary of metric names → values.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (heart disease)

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1-Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
    }

    print("\n" + "=" * 45)
    print("          MODEL EVALUATION RESULTS")
    print("=" * 45)
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")
    print("=" * 45)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))

    # 1. Confusion Matrix
    _plot_confusion_matrix(y_test, y_pred, output_dir)

    # 2. ROC Curve
    _plot_roc_curve(y_test, y_prob, metrics["ROC-AUC"], output_dir)

    # 3. Feature Importance (Logistic Regression coefficients)
    _plot_feature_importance(model, X_test.columns.tolist(), output_dir)

    return metrics


def _plot_confusion_matrix(y_test, y_pred, output_dir: str) -> None:
    """
    Plot and save the confusion matrix.

    A confusion matrix shows:
        - True Positives (TP): Sick patients correctly identified
        - True Negatives (TN): Healthy patients correctly identified
        - False Positives (FP): Healthy patients wrongly flagged as sick
        - False Negatives (FN): Sick patients missed — most dangerous error!
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Heart Disease"],
        yticklabels=["No Disease", "Heart Disease"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to '{path}'")


def _plot_roc_curve(y_test, y_prob, auc_score: float, output_dir: str) -> None:
    """
    Plot and save the ROC (Receiver Operating Characteristic) curve.

    The ROC curve plots True Positive Rate (Recall) vs False Positive Rate
    at every probability threshold. The Area Under the Curve (AUC) summarises
    this: AUC=1.0 is perfect, AUC=0.5 is random guessing.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="royalblue", lw=2, label=f"Logistic Regression (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="royalblue")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve — Heart Disease Prediction", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] ROC curve saved to '{path}'")


def _plot_feature_importance(model: LogisticRegression, feature_names: list, output_dir: str) -> None:
    """
    Plot and save feature importances from Logistic Regression coefficients.

    In Logistic Regression, each coefficient tells us:
      - Positive coefficient → feature increases probability of heart disease
      - Negative coefficient → feature decreases probability of heart disease
      - Magnitude → strength of the effect

    Note: Because features are standardised, coefficients are comparable in magnitude.
    """
    coefficients = model.coef_[0]
    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
    }).sort_values("Coefficient", ascending=True)

    colors = ["#d73027" if c > 0 else "#4575b4" for c in feature_df["Coefficient"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(feature_df["Feature"], feature_df["Coefficient"], color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Logistic Regression Coefficient", fontsize=12)
    ax.set_title("Feature Importance (LR Coefficients)\nRed = Risk Factor | Blue = Protective", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Feature importance plot saved to '{path}'")
