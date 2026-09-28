"""
data_preprocessing.py
---------------------
Handles loading, cleaning, and splitting of the Heart Disease dataset.

Why this module?
    Keeping data loading and cleaning separate from model training ensures
    that preprocessing logic is reusable, testable, and easy to swap datasets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the heart disease CSV dataset into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Raw DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a summary of the dataset for quick inspection."""
    print("\n--- Dataset Info ---")
    print(df.info())
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    print("\n--- Class distribution (target) ---")
    print(df["target"].value_counts())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
      - Drop duplicate rows
      - Handle missing values (median imputation for numerics)
      - Ensure correct data types

    Why median imputation?
        Medical data often has outliers (e.g., very high cholesterol).
        Median is robust to outliers unlike mean.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    original_shape = df.shape

    # Drop duplicate rows
    df = df.drop_duplicates()
    print(f"[INFO] Duplicates removed: {original_shape[0] - df.shape[0]} rows")

    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[INFO] Filled missing values in '{col}' with median={median_val:.2f}")

    # Ensure target column is integer
    df["target"] = df["target"].astype(int)

    print(f"[INFO] Data after cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the cleaned DataFrame into train and test sets.

    Why 80/20 split?
        Gives enough training data for a ~300-row dataset while keeping
        a meaningful held-out test set for unbiased evaluation.

    Why stratify?
        Preserves the class ratio in both train and test sets — critical
        for imbalanced medical datasets.

    Args:
        df: Cleaned DataFrame.
        target_col: Name of the target column.
        test_size: Fraction of data for testing.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test
