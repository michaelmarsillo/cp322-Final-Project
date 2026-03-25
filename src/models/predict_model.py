"""
Make predictions using the best trained model
Loads the best model and scaler, then makes predictions on new data
"""

import os
import joblib
import pandas as pd
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.build_features import run_feature_engineering


def load_model_and_scaler():
    """
    Load the best trained model and the scaler from disk
    """
    models_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(models_dir), "data", "processed")
    
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    scaler_path = os.path.join(data_dir, "scaler.joblib")
    
    best_model = joblib.load(best_model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Loaded best model from: {best_model_path}")
    print(f"Loaded scaler from: {scaler_path}")
    
    return best_model, scaler


def make_prediction(model, scaler, patient_data, fe):
    """
    Make a prediction for a single patient
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        patient_data: DataFrame with one row of patient features
        fe: FeatureEngineer object
    
    Returns:
        prediction, probability
    """
    # Use the feature engineer's transform method to properly scale
    patient_scaled = fe.transform(patient_data)
    
    # Make prediction
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]
    
    return prediction, probability


def main():
    """
    Load model and make a prediction on a sample patient
    """
    
    print("="*50)
    print("HEART DISEASE PREDICTION")
    print("="*50)
    
    # Load model and scaler
    best_model, scaler = load_model_and_scaler()
    
    # Get test data to use as example
    _, X_test, _, _, fe = run_feature_engineering()
    
    # Take the first test patient as example
    patient = X_test.iloc[0:1]  # Keep as DataFrame
    actual_diagnosis = "Disease" if patient.index[0] in X_test.index[:50] else "Unknown"
    
    print("\n" + "="*50)
    print("SAMPLE PATIENT PREDICTION")
    print("="*50)
    print("\nPatient Features:")
    print(patient.to_string())
    
    # Make prediction
    prediction, probability = make_prediction(best_model, scaler, patient, fe)
    
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    
    if prediction == 0:
        diagnosis = "No Heart Disease"
        confidence = probability[0] * 100
    else:
        diagnosis = "Heart Disease Present"
        confidence = probability[1] * 100
    
    print(f"\nPrediction: {diagnosis}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"No Disease Probability: {probability[0]:.4f}")
    print(f"Disease Probability: {probability[1]:.4f}")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    main()
