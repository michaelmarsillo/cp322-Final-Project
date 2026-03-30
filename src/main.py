"""
Main orchestration script for the heart disease prediction project
Runs the complete ML pipeline from feature engineering through visualization
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """
    Run the complete pipeline: feature engineering → training → tuning → evaluation → visualization
    """
    
    print("\n" + "="*70)
    print("HEART DISEASE PREDICTION - COMPLETE ML PIPELINE")
    print("="*70)
    
    completed_phases = []
    failed_phases = []
    
    # Phase 3: Feature Engineering
    print("\n" + "-"*70)
    print("Phase 3: Feature Engineering (Train/Test Split & Scaling)")
    print("-"*70)
    try:
        from feature_engineering.build_features import run_feature_engineering
        print("Running feature engineering pipeline...")
        X_train_scaled, X_test_scaled, y_train, y_test, fe = run_feature_engineering()
        print("[OK] Phase 3 Complete: Data split (80/20) and scaled")
        completed_phases.append("Phase 3: Feature Engineering")
    except Exception as e:
        print(f"[FAILED] Phase 3 Failed: {str(e)}")
        failed_phases.append(("Phase 3", str(e)))
        return
    
    # Phase 4: Train Models
    print("\n" + "-"*70)
    print("Phase 4: Model Training (Baseline Models)")
    print("-"*70)
    try:
        from models.train_model import train_models
        print("Training baseline models...")
        train_models()
        print("[OK] Phase 4 Complete: 3 baseline models trained and saved")
        completed_phases.append("Phase 4: Model Training")
    except Exception as e:
        print(f"[FAILED] Phase 4 Failed: {str(e)}")
        failed_phases.append(("Phase 4", str(e)))
        return
    
    # Phase 5: Tune Models
    print("\n" + "-"*70)
    print("Phase 5: Hyperparameter Tuning")
    print("-"*70)
    try:
        from models.tune_model import tune_models
        print("Tuning model hyperparameters with GridSearchCV...")
        tune_models()
        print("[OK] Phase 5 Complete: 3 tuned models created, best model saved")
        completed_phases.append("Phase 5: Hyperparameter Tuning")
    except Exception as e:
        print(f"[FAILED] Phase 5 Failed: {str(e)}")
        failed_phases.append(("Phase 5", str(e)))
        return
    
    # Phase 6a: Evaluate Models
    print("\n" + "-"*70)
    print("Phase 6a: Model Evaluation & Comparison")
    print("-"*70)
    try:
        from models.evaluate_models import main as evaluate_main
        print("Evaluating all models...")
        evaluate_main()
        print("[OK] Phase 6a Complete: All models compared and ranked")
        completed_phases.append("Phase 6a: Model Evaluation")
    except Exception as e:
        print(f"[FAILED] Phase 6a Failed: {str(e)}")
        failed_phases.append(("Phase 6a", str(e)))
        return
    
    # Phase 6b: Visualize Results
    print("\n" + "-"*70)
    print("Phase 6b: Visualization & Reporting")
    print("-"*70)
    try:
        from visualization.visualize import main as visualize_main
        print("Generating visualizations...")
        visualize_main()
        print("[OK] Phase 6b Complete: 5 plots saved to reports/figures/")
        completed_phases.append("Phase 6b: Visualization")
    except Exception as e:
        print(f"[FAILED] Phase 6b Failed: {str(e)}")
        failed_phases.append(("Phase 6b", str(e)))
        return
    
    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    
    print(f"\nCompleted Phases ({len(completed_phases)}):")
    for i, phase in enumerate(completed_phases, 1):
        print(f"  {i}. {phase}")

    if failed_phases:
        print(f"\nFailed Phases ({len(failed_phases)}):")
        for phase, error in failed_phases:
            print(f"  - {phase}: {error}")
    else:
        print("\nAll phases completed successfully!")
    
    print("\n" + "="*70)
    print("OUTPUT LOCATIONS")
    print("="*70)
    print("  Models:      src/models/")
    print("  Scaler:      src/data/processed/scaler.joblib")
    print("  Plots:       reports/figures/")
    print("  Best Model:  src/models/best_model.joblib (83.70% accuracy)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
