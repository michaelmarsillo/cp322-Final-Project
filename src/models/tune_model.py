"""
Hyperparameter tuning for heart disease prediction models.
Uses GridSearchCV with 5-fold cross-validation to find the best
parameters for Logistic Regression, Decision Tree, and Random Forest.
"""

import os
import sys
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering.build_features import run_feature_engineering

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def tune_logistic_regression(X_train, y_train):
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def tune_decision_tree(X_train, y_train):
    param_grid = {
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return acc


def tune_models():
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test, fe = run_feature_engineering()

    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING")
    print("(5-fold cross-validation)")
    print("=" * 50)

    print("\n1. Tuning Logistic Regression...")
    best_lr = tune_logistic_regression(X_train, y_train)
    acc_lr = evaluate(best_lr, X_test, y_test, "Logistic Regression (Tuned)")

    print("\n2. Tuning Decision Tree...")
    best_dt = tune_decision_tree(X_train, y_train)
    acc_dt = evaluate(best_dt, X_test, y_test, "Decision Tree (Tuned)")

    print("\n3. Tuning Random Forest...")
    best_rf = tune_random_forest(X_train, y_train)
    acc_rf = evaluate(best_rf, X_test, y_test, "Random Forest (Tuned)")

    # Save tuned models
    joblib.dump(best_lr, os.path.join(MODELS_DIR, "logistic_regression_tuned.joblib"))
    joblib.dump(best_dt, os.path.join(MODELS_DIR, "decision_tree_tuned.joblib"))
    joblib.dump(best_rf, os.path.join(MODELS_DIR, "random_forest_tuned.joblib"))
    print("\nAll tuned models saved.")

    print("\n" + "=" * 50)
    print("TUNING SUMMARY")
    print("=" * 50)
    print(f"Logistic Regression (Tuned): {acc_lr:.4f}")
    print(f"Decision Tree      (Tuned): {acc_dt:.4f}")
    print(f"Random Forest      (Tuned): {acc_rf:.4f}")
    print("=" * 50)

    # Return best overall model for use in predict_model.py
    best_acc = max(acc_lr, acc_dt, acc_rf)
    best_model = {acc_lr: best_lr, acc_dt: best_dt, acc_rf: best_rf}[best_acc]
    best_name = {acc_lr: "Logistic Regression", acc_dt: "Decision Tree", acc_rf: "Random Forest"}[best_acc]
    print(f"\nBest model: {best_name} ({best_acc:.4f})")
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    print("Best model saved as best_model.joblib")

    return best_lr, best_dt, best_rf


if __name__ == "__main__":
    tune_models()
