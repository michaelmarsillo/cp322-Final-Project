"""
Verifies the sensitivity analysis numbers reported in Section 4.4 of the report.
Run from project root: python verify_sensitivity.py
"""

import os
import sys
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from feature_engineering.build_features import run_feature_engineering, NUMERICAL_FEATURES

import pandas as pd

MODELS_DIR = os.path.join("src", "models")
PROCESSED_DATA_DIR = os.path.join("src", "data", "processed")


def main():
    print("=" * 55)
    print("SENSITIVITY ANALYSIS VERIFICATION")
    print("=" * 55)

    # Load data
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "heart_disease_clean.csv"))
    X_train, X_test, y_train, y_test, fe = run_feature_engineering()

    # Load best model (tuned Random Forest)
    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))

    # Part 1: 5-fold CV on full dataset ────────────────────────────────────
    print("\nPart 1: 5-Fold Cross-Validation on full dataset")
    print("-" * 55)

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    ALL_FEATURES = X_train.columns.tolist()
    TARGET = "target"

    X_full = df[ALL_FEATURES].copy()
    y_full = df[TARGET]

    # Scale numerical features for the full dataset
    scaler = StandardScaler()
    X_full[NUMERICAL_FEATURES] = scaler.fit_transform(X_full[NUMERICAL_FEATURES])

    cv_scores = cross_val_score(best_model, X_full, y_full, cv=5, scoring="accuracy")
    print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Range:  {cv_scores.min()*100:.1f}% - {cv_scores.max()*100:.1f}%")
    print(f"  Mean:   {cv_scores.mean()*100:.1f}%")
    print(f"  Std:    {cv_scores.std()*100:.1f}%")

    # Part 2: Feature ablation ──────────────────────────────────────────────
    print("\nPart 2: Feature Ablation (drop one top feature at a time)")
    print("-" * 55)

    from sklearn.metrics import accuracy_score

    baseline_acc = accuracy_score(y_test, best_model.predict(X_test))
    print(f"  Baseline accuracy (all features): {baseline_acc*100:.2f}%")

    for feature in ["cp", "thalach", "ca"]:
        X_test_dropped = X_test.drop(columns=[feature])
        X_train_dropped = X_train.drop(columns=[feature])

        # Retrain a fresh RF with same params on reduced features
        rf_reduced = RandomForestClassifier(
            max_depth=15, min_samples_leaf=1,
            min_samples_split=5, n_estimators=200, random_state=42
        )
        rf_reduced.fit(X_train_dropped, y_train)
        acc_dropped = accuracy_score(y_test, rf_reduced.predict(X_test_dropped))
        drop = (baseline_acc - acc_dropped) * 100
        print(f"  Drop '{feature}': accuracy = {acc_dropped*100:.2f}% (change: -{drop:.1f}%)")

    print("\nDone.")
    print("=" * 55)


if __name__ == "__main__":
    main()
