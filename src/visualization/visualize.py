"""
Visualization script for heart disease prediction models.
Generates 5 plots saved to reports/figures/ for use in the final report.
"""

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering.build_features import run_feature_engineering

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports", "figures")

# Consistent color palette across all plots
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "light": "#C73E1D",
    "neutral": "#5C6BC0",
    "green": "#43A047",
}


def setup():
    """Load data and models needed for all plots."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, fe = run_feature_engineering()

    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))

    model_files = {
        "LR (Baseline)":  "logistic_regression.joblib",
        "DT (Baseline)":  "decision_tree.joblib",
        "RF (Baseline)":  "random_forest.joblib",
        "LR (Tuned)":     "logistic_regression_tuned.joblib",
        "DT (Tuned)":     "decision_tree_tuned.joblib",
        "RF (Tuned)":     "random_forest_tuned.joblib",
    }
    all_models = {}
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            all_models[name] = joblib.load(path)

    return X_train, X_test, y_train, y_test, best_model, all_models


def plot_confusion_matrix(best_model, X_test, y_test):
    """Heatmap of the confusion matrix for the best model (Random Forest Tuned)."""
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Random Forest (Tuned)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: confusion_matrix.png")


def plot_roc_curves(all_models, X_test, y_test):
    """ROC curve for each model on the same axes so they can be directly compared."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Colour cycle for the 6 models
    palette = [
        "#2E86AB", "#A23B72", "#F18F01",
        "#C73E1D", "#5C6BC0", "#43A047",
    ]

    for (name, model), color in zip(all_models.items(), palette):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Guess")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: roc_curves.png")


def plot_feature_importance(best_model, X_test):
    """
    Horizontal bar chart of feature importances from the best model (Random Forest).
    Shows which patient attributes matter most for the prediction.
    """
    importances = best_model.feature_importances_
    feature_names = X_test.columns.tolist()

    # Sort by importance
    indices = np.argsort(importances)
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(sorted_names, sorted_vals, color=COLORS["primary"], edgecolor="white")

    # Highlight top 3 features
    top3_threshold = sorted(importances, reverse=True)[2]
    for bar, val in zip(bars, sorted_vals):
        if val >= top3_threshold:
            bar.set_color(COLORS["accent"])

    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Feature Importance — Random Forest (Tuned)", fontsize=13, fontweight="bold")

    highlight = mpatches.Patch(color=COLORS["accent"], label="Top 3 features")
    rest = mpatches.Patch(color=COLORS["primary"], label="Other features")
    ax.legend(handles=[highlight, rest], fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: feature_importance.png")


def plot_model_comparison(all_models, X_test, y_test):
    """
    Grouped bar chart comparing Accuracy and ROC-AUC for all 6 models.
    Makes it easy to see the improvement tuning brought.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    names, accuracies, aucs = [], [], []
    for name, model in all_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        names.append(name)
        accuracies.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_proba))

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color=COLORS["primary"], edgecolor="white")
    bars2 = ax.bar(x + width / 2, aucs, width, label="ROC-AUC", color=COLORS["accent"], edgecolor="white")

    # Value labels on top of each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim([0.5, 1.05])
    ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1, label="80% target")
    ax.set_title("Model Comparison — Accuracy & ROC-AUC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: model_comparison.png")


def plot_precision_recall_f1(all_models, X_test, y_test):
    """
    Grouped bar chart showing Precision, Recall, and F1-score for each model.
    Useful for showing that accuracy alone doesn't tell the full story.
    """
    from sklearn.metrics import classification_report

    names, precisions, recalls, f1s = [], [], [], []
    for name, model in all_models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        names.append(name)
        precisions.append(report["weighted avg"]["precision"])
        recalls.append(report["weighted avg"]["recall"])
        f1s.append(report["weighted avg"]["f1-score"])

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, precisions, width, label="Precision", color=COLORS["primary"], edgecolor="white")
    ax.bar(x,          recalls,   width, label="Recall",    color=COLORS["secondary"], edgecolor="white")
    ax.bar(x + width,  f1s,       width, label="F1-Score",  color=COLORS["green"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim([0.5, 1.05])
    ax.set_title("Precision, Recall & F1-Score — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "precision_recall_f1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: precision_recall_f1.png")


def main():
    print("=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)

    print("\nLoading data and models...")
    X_train, X_test, y_train, y_test, best_model, all_models = setup()

    print("\nPlot 1: Confusion Matrix")
    plot_confusion_matrix(best_model, X_test, y_test)

    print("Plot 2: ROC Curves")
    plot_roc_curves(all_models, X_test, y_test)

    print("Plot 3: Feature Importance")
    plot_feature_importance(best_model, X_test)

    print("Plot 4: Model Comparison (Accuracy & ROC-AUC)")
    plot_model_comparison(all_models, X_test, y_test)

    print("Plot 5: Precision, Recall & F1-Score")
    plot_precision_recall_f1(all_models, X_test, y_test)

    print(f"\nAll plots saved to: {os.path.abspath(FIGURES_DIR)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
