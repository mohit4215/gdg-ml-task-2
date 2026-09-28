"""
train.py
--------
Main entry point for the complete ML training pipeline.

Run this script to:
  1. Load and clean the dataset
  2. Split into train/test sets
  3. Scale features
  4. Train the Logistic Regression model with hyperparameter tuning
  5. Evaluate and save all metrics + visualisations

Usage:
    python train.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' can be imported
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import load_data, inspect_data, clean_data, split_data
from src.feature_engineering import scale_features
from src.model_training import train_model
from src.model_evaluation import evaluate_model


DATA_PATH  = "data/heart.csv"
MODEL_PATH = "models/logistic_model.pkl"
SCALER_PATH = "models/scaler.pkl"
OUTPUT_DIR = "outputs"


def run_pipeline():
    print("\n" + "=" * 50)
    print("  ❤️  HEART DISEASE PREDICTION — ML PIPELINE")
    print("=" * 50)

    # ── Step 1: Load Data ────────────────────────────────
    print("\n[STEP 1] Loading dataset...")
    df = load_data(DATA_PATH)
    inspect_data(df)

    # ── Step 2: Clean Data ───────────────────────────────
    print("\n[STEP 2] Cleaning data...")
    df_clean = clean_data(df)

    # ── Step 3: Train/Test Split ─────────────────────────
    print("\n[STEP 3] Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df_clean)

    # ── Step 4: Feature Engineering ──────────────────────
    print("\n[STEP 4] Scaling features...")
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, SCALER_PATH)

    # ── Step 5: Train Model ───────────────────────────────
    print("\n[STEP 5] Training Logistic Regression model...")
    model = train_model(X_train_scaled, y_train, MODEL_PATH)

    # ── Step 6: Evaluate Model ────────────────────────────
    print("\n[STEP 6] Evaluating model on test set...")
    metrics = evaluate_model(model, X_test_scaled, y_test, OUTPUT_DIR)

    print("\n✅ Pipeline complete!")
    print(f"   Model saved  → {MODEL_PATH}")
    print(f"   Scaler saved → {SCALER_PATH}")
    print(f"   Plots saved  → {OUTPUT_DIR}/")
    print("\nRun the Streamlit app with:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    run_pipeline()
