import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Features that are truly continuous/numerical — need StandardScaler
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Features that are categorical or ordinal — we leave as-is (already encoded as numbers)
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"


class FeatureEngineer:
    """
    Handles feature scaling and train/test splitting.

    Design notes:
    - Numerical features are standardized (zero mean, unit variance).
      This is required for Logistic Regression to converge properly,
      and helps Decision Tree / Random Forest as well.
    - Categorical features are left as integer-encoded values.
      They are already in a usable numeric form from preprocessing.
    - The scaler is fit ONLY on training data to prevent data leakage.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False

    def split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Split the dataset into train and test sets.

        Args:
            df: Cleaned DataFrame from preprocessing pipeline.
            test_size: Fraction of data to use for testing (default 20%).
            random_state: Seed for reproducibility.

        Returns:
            X_train, X_test, y_train, y_test as DataFrames/Series.
        """
        X = df[ALL_FEATURES]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,   # preserve class balance in both splits
        )

        print(f"Train size: {len(X_train)} samples")
        print(f"Test size:  {len(X_test)} samples")
        print(f"Train class distribution:\n{y_train.value_counts().to_string()}")
        print(f"Test class distribution:\n{y_test.value_counts().to_string()}")

        return X_train, X_test, y_train, y_test

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the scaler on training data and transform it.
        Call this ONLY on training data.

        Returns a new DataFrame with scaled numerical features.
        """
        X_scaled = X_train.copy()
        X_scaled[NUMERICAL_FEATURES] = self.scaler.fit_transform(X_train[NUMERICAL_FEATURES])
        self._is_fitted = True
        print(f"Scaler fitted on {len(X_train)} training samples.")
        return X_scaled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the already-fitted scaler.
        Call this on test data (and any new data at prediction time).

        Returns a new DataFrame with scaled numerical features.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit_transform() first.")
        X_scaled = X.copy()
        X_scaled[NUMERICAL_FEATURES] = self.scaler.transform(X[NUMERICAL_FEATURES])
        return X_scaled

    def save_scaler(self, filename: str = "scaler.joblib"):
        """Save the fitted scaler to disk for later use."""
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        path = os.path.join(PROCESSED_DATA_DIR, filename)
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to: {path}")

    def load_scaler(self, filename: str = "scaler.joblib"):
        """Load a previously saved scaler from disk."""
        path = os.path.join(PROCESSED_DATA_DIR, filename)
        self.scaler = joblib.load(path)
        self._is_fitted = True
        print(f"Scaler loaded from: {path}")


def run_feature_engineering(df: pd.DataFrame = None):
    """
    Full feature engineering pipeline:
    load cleaned data -> split -> scale -> return ready-to-use arrays.

    Args:
        df: Optional pre-loaded DataFrame. If None, loads from disk.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, feature_engineer
    """
    if df is None:
        path = os.path.join(PROCESSED_DATA_DIR, "heart_disease_clean.csv")
        df = pd.read_csv(path)
        print(f"Loaded cleaned data: {df.shape}")

    fe = FeatureEngineer()

    print("\n" + "=" * 50)
    print("Step 1: Train/Test Split (80/20, stratified)")
    print("=" * 50)
    X_train, X_test, y_train, y_test = fe.split(df)

    print("\n" + "=" * 50)
    print("Step 2: Feature Scaling (StandardScaler on numerical features)")
    print("=" * 50)
    X_train_scaled = fe.fit_transform(X_train)
    X_test_scaled = fe.transform(X_test)

    fe.save_scaler()

    print(f"\nFeature engineering complete.")
    print(f"  Numerical features scaled: {NUMERICAL_FEATURES}")
    print(f"  Categorical features unchanged: {CATEGORICAL_FEATURES}")

    return X_train_scaled, X_test_scaled, y_train, y_test, fe


if __name__ == "__main__":
    run_feature_engineering()
