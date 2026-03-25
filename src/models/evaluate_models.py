"""
Evaluate and compare all trained models
Loads all 6 models (3 baseline + 3 tuned) and compares their performance
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.build_features import run_feature_engineering


def load_all_models():
    """
    Load all 6 trained models from disk
    """
    models_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_files = {
        "Logistic Regression (Baseline)": "logistic_regression.joblib",
        "Decision Tree (Baseline)": "decision_tree.joblib",
        "Random Forest (Baseline)": "random_forest.joblib",
        "Logistic Regression (Tuned)": "logistic_regression_tuned.joblib",
        "Decision Tree (Tuned)": "decision_tree_tuned.joblib",
        "Random Forest (Tuned)": "random_forest_tuned.joblib",
    }
    
    models = {}
    for model_name, filename in model_files.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            models[model_name] = joblib.load(filepath)
            print(f"Loaded: {model_name}")
        else:
            print(f"Warning: {filename} not found at {filepath}")
    
    return models


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model and return metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


def main():
    """
    Load all models, evaluate them, and create comparison table
    """
    
    print("="*70)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*70)
    
    # Load feature engineering data
    print("\nLoading data...")
    _, X_test, _, y_test, _ = run_feature_engineering()
    
    # Load all models
    print("\nLoading models...")
    models = load_all_models()
    
    if not models:
        print("No models found to evaluate.")
        return
    
    # Evaluate each model
    print("\n" + "="*70)
    print("EVALUATING MODELS")
    print("="*70)
    
    results = []
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
        
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  ROC-AUC:  {metrics['ROC-AUC']:.4f}")
        print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    
    # Create comparison dataframe
    df_results = pd.DataFrame(results)
    
    # Sort by accuracy (descending)
    df_results = df_results.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    
    # Print detailed comparison table
    print("\n" + "="*70)
    print("DETAILED COMPARISON TABLE")
    print("="*70)
    
    comparison_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    print("\n")
    print(df_results[comparison_cols].to_string(index=False))
    
    # Print confusion matrix summary
    print("\n" + "="*70)
    print("CONFUSION MATRIX SUMMARY (ranked by accuracy)")
    print("="*70)
    
    for idx, row in df_results.iterrows():
        print(f"\n{idx+1}. {row['Model']}")
        print(f"   True Positives:  {int(row['TP'])}")
        print(f"   True Negatives:  {int(row['TN'])}")
        print(f"   False Positives: {int(row['FP'])}")
        print(f"   False Negatives: {int(row['FN'])}")
    
    # Print winner
    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)
    best = df_results.iloc[0]
    print(f"\n🏆 {best['Model']}")
    print(f"   Accuracy: {best['Accuracy']:.4f} ({best['Accuracy']*100:.2f}%)")
    print(f"   ROC-AUC:  {best['ROC-AUC']:.4f}")
    print(f"   F1-Score: {best['F1-Score']:.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
