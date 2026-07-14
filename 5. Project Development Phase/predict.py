import os
import argparse
import joblib
import pandas as pd
import numpy as np

# Define columns in order of training
FEATURE_COLUMNS = [
    "Annual Rainfall", "Cloud Visibility", "January Rainfall", "February Rainfall",
    "March Rainfall", "April Rainfall", "May Rainfall", "June Rainfall",
    "July Rainfall", "August Rainfall", "September Rainfall", "October Rainfall",
    "November Rainfall", "December Rainfall"
]

def load_prediction_resources(model_path="model/flood_model.pkl", scaler_path="model/scaler.pkl"):
    """
    Loads and returns the trained model and scaler.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please run train_model.py first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at '{scaler_path}'. Please run train_model.py first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_flood(feature_values, model_path="model/flood_model.pkl", scaler_path="model/scaler.pkl"):
    """
    Takes a dictionary or list of feature values, scales them, and makes a prediction.
    
    Parameters:
    - feature_values: A dictionary with keys as feature names, OR a list/numpy array of 14 values
                      ordered matching FEATURE_COLUMNS.
                      
    Returns:
    - result: dict containing 'prediction' (0 or 1), 'probability' (float), and 'status' (string)
    """
    model, scaler = load_prediction_resources(model_path, scaler_path)
    
    # 1. Standardize inputs to a DataFrame with correct column names and order
    if isinstance(feature_values, dict):
        # Fill in missing columns with 0 if any
        data_dict = {col: [feature_values.get(col, 0.0)] for col in FEATURE_COLUMNS}
        input_df = pd.DataFrame(data_dict)
    elif isinstance(feature_values, (list, np.ndarray)):
        if len(feature_values) != len(FEATURE_COLUMNS):
            raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {len(feature_values)}")
        input_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
    elif isinstance(input_df, pd.DataFrame):
        input_df = feature_values[FEATURE_COLUMNS] # ensure column order
    else:
        raise TypeError("Input must be a dictionary, list, or numpy array.")
        
    # 2. Scale features
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=FEATURE_COLUMNS)
    
    # 3. Predict class and probability
    pred = int(model.predict(input_scaled_df)[0])
    
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_scaled_df)[0][1])
    else:
        prob = 1.0 if pred == 1 else 0.0
        
    status = "Flood Detected" if pred == 1 else "No Flood"
    
    return {
        "prediction": pred,
        "probability": round(prob, 4),
        "status": status
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Powered Flood Prediction CLI")
    parser.add_argument("--jan", type=float, default=20.0, help="January Rainfall in mm")
    parser.add_argument("--feb", type=float, default=25.0, help="February Rainfall in mm")
    parser.add_argument("--mar", type=float, default=40.0, help="March Rainfall in mm")
    parser.add_argument("--apr", type=float, default=80.0, help="April Rainfall in mm")
    parser.add_argument("--may", type=float, default=180.0, help="May Rainfall in mm")
    parser.add_argument("--jun", type=float, default=450.0, help="June Rainfall in mm")
    parser.add_argument("--jul", type=float, default=620.0, help="July Rainfall in mm")
    parser.add_argument("--aug", type=float, default=580.0, help="August Rainfall in mm")
    parser.add_argument("--sep", type=float, default=320.0, help="September Rainfall in mm")
    parser.add_argument("--oct", type=float, default=110.0, help="October Rainfall in mm")
    parser.add_argument("--nov", type=float, default=40.0, help="November Rainfall in mm")
    parser.add_argument("--dec", type=float, default=20.0, help="December Rainfall in mm")
    parser.add_argument("--visibility", type=float, default=35.0, help="Cloud Visibility in percentage")
    
    args = parser.parse_args()
    
    # Calculate annual rainfall as the sum of monthly ones
    monthly_vals = [
        args.jan, args.feb, args.mar, args.apr, args.may, args.jun,
        args.jul, args.aug, args.sep, args.oct, args.nov, args.dec
    ]
    annual = sum(monthly_vals)
    
    # Pack input dictionary
    input_data = {
        "Annual Rainfall": annual,
        "Cloud Visibility": args.visibility,
        "January Rainfall": args.jan,
        "February Rainfall": args.feb,
        "March Rainfall": args.mar,
        "April Rainfall": args.apr,
        "May Rainfall": args.may,
        "June Rainfall": args.jun,
        "July Rainfall": args.jul,
        "August Rainfall": args.aug,
        "September Rainfall": args.sep,
        "October Rainfall": args.oct,
        "November Rainfall": args.nov,
        "December Rainfall": args.dec
    }
    
    print("\n" + "="*40)
    print("RUNNING FLOOD PREDICTION TEST INPUT")
    print("="*40)
    for k, v in input_data.items():
        print(f"{k}: {v}")
    print("-"*40)
    
    try:
        res = predict_flood(input_data)
        print(f"Prediction Result: {res['status']}")
        print(f"Flood Probability: {res['probability']*100:.2f}%")
        print(f"Class Label: {res['prediction']}")
        print("="*40 + "\n")
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Make sure you have run 'train_model.py' to generate 'flood_model.pkl' and 'scaler.pkl'.")
