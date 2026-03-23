"""
Train machine learning models for heart disease prediction
Trains Logistic Regression, Decision Tree, and Random Forest models
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys

# Add parent directory to path to import feature engineering module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.build_features import run_feature_engineering


def train_models():
    """
    Train 3 different models on the preprocessed data
    """
    
    print("Loading preprocessed data...")
    X_train_scaled, X_test_scaled, y_train, y_test, fe = run_feature_engineering()
    
    # Create models directory if it doesn't exist
    models_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    # 1. LOGISTIC REGRESSION
    print("\n1. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr))
    
    # Save model
    lr_path = os.path.join(models_dir, 'logistic_regression.joblib')
    joblib.dump(lr, lr_path)
    print(f"Saved to {lr_path}")
    
    # 2. DECISION TREE
    print("\n2. Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_scaled, y_train)
    
    y_pred_dt = dt.predict(X_test_scaled)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    
    print(f"Decision Tree Accuracy: {acc_dt:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_dt))
    
    # Save model
    dt_path = os.path.join(models_dir, 'decision_tree.joblib')
    joblib.dump(dt, dt_path)
    print(f"Saved to {dt_path}")
    
    # 3. RANDOM FOREST
    print("\n3. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))
    
    # Save model
    rf_path = os.path.join(models_dir, 'random_forest.joblib')
    joblib.dump(rf, rf_path)
    print(f"Saved to {rf_path}")
    
    # Summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Logistic Regression: {acc_lr:.4f}")
    print(f"Decision Tree:       {acc_dt:.4f}")
    print(f"Random Forest:       {acc_rf:.4f}")
    print("="*50)


if __name__ == "__main__":
    train_models()
