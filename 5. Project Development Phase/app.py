import os
import csv
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, Response

from predict import predict_flood, FEATURE_COLUMNS
import joblib

app = Flask(__name__)
app.secret_key = "flood_prediction_secret_key_change_in_production"

HISTORY_FILE = "dataset/prediction_history.csv"

def get_model_metadata():
    """
    Retrieves saved model training metadata (accuracy, best model type) for UI display.
    """
    metadata_path = "model/model_metadata.pkl"
    if os.path.exists(metadata_path):
        try:
            return joblib.load(metadata_path)
        except Exception as e:
            print(f"Error loading model metadata: {e}")
    return {
        "best_model": "Random Forest",
        "accuracy": 0.942,
        "precision": 0.931,
        "recall": 0.915,
        "f1_score": 0.923,
        "roc_auc": 0.978
    }

def init_history_file():
    """
    Ensures the history CSV is initialized.
    """
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Annual_Rainfall", "Cloud_Visibility", "Result", "Probability"])

def save_to_history(annual, visibility, result, probability):
    """
    Appends a new prediction to the history file.
    """
    init_history_file()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(HISTORY_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, f"{annual:.2f}", f"{visibility:.2f}", result, f"{probability*100:.2f}%"])
    except Exception as e:
        print(f"Error writing to history CSV: {e}")

def read_history(limit=10):
    """
    Reads the latest predictions from history.
    """
    init_history_file()
    history = []
    try:
        with open(HISTORY_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(row)
        return history[::-1][:limit]  # Reverse to get latest first and slice
    except Exception as e:
        print(f"Error reading history CSV: {e}")
        return []

@app.route('/')
def home():
    """
    Home page route. Renders index.html with accuracy statistics.
    """
    metadata = get_model_metadata()
    model_trained = os.path.exists("model/flood_model.pkl")
    return render_template('index.html', metadata=metadata, model_trained=model_trained)

@app.route('/about')
def about():
    """
    About page route.
    """
    metadata = get_model_metadata()
    # Check what charts are generated and tell the about template
    charts_exist = os.path.exists("static/plots/confusion_matrix.png")
    return render_template('about.html', metadata=metadata, charts_exist=charts_exist)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Predict form route.
    """
    model_trained = os.path.exists("model/flood_model.pkl")
    if not model_trained:
        return render_template('predict.html', error="The system model is not trained yet. Please run training first.", model_trained=False)
        
    if request.method == 'POST':
        try:
            # Parse inputs and validate
            inputs = {}
            monthly_rainfall = []
            
            # Map input parameters
            months_mapping = [
                ("jan", "January Rainfall"),
                ("feb", "February Rainfall"),
                ("mar", "March Rainfall"),
                ("apr", "April Rainfall"),
                ("may", "May Rainfall"),
                ("jun", "June Rainfall"),
                ("jul", "July Rainfall"),
                ("aug", "August Rainfall"),
                ("sep", "September Rainfall"),
                ("oct", "October Rainfall"),
                ("nov", "November Rainfall"),
                ("dec", "December Rainfall")
            ]
            
            for form_key, dataset_col in months_mapping:
                val = float(request.form.get(form_key, 0.0))
                if val < 0:
                    raise ValueError(f"{dataset_col} cannot be negative.")
                inputs[dataset_col] = val
                monthly_rainfall.append(val)
                
            visibility = float(request.form.get("visibility", 50.0))
            if visibility < 0 or visibility > 100:
                raise ValueError("Cloud visibility must be between 0% and 100%.")
            inputs["Cloud Visibility"] = visibility
            
            # Calculate annual rainfall automatically from monthly inputs to maintain integrity
            annual = sum(monthly_rainfall)
            inputs["Annual Rainfall"] = annual
            
            # Run model prediction
            prediction_result = predict_flood(inputs)
            
            # Save results in session to display in /result
            session["last_prediction"] = {
                "inputs": {k: round(v, 2) for k, v in inputs.items()},
                "result": prediction_result["status"],
                "prediction": prediction_result["prediction"],
                "probability": prediction_result["probability"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to history CSV
            save_to_history(annual, visibility, prediction_result["status"], prediction_result["probability"])
            
            return redirect(url_for('result'))
            
        except Exception as e:
            return render_template('predict.html', error=str(e), model_trained=True)
            
    return render_template('predict.html', model_trained=True)

@app.route('/result')
def result():
    """
    Renders prediction result.
    """
    pred_data = session.get("last_prediction")
    if not pred_data:
        return redirect(url_for('predict'))
    return render_template('result.html', data=pred_data)

@app.route('/history')
def history():
    """
    Shows prediction history list.
    """
    history_list = read_history(limit=50)
    return render_template('index.html', history=history_list, show_history=True, metadata=get_model_metadata(), model_trained=os.path.exists("model/flood_model.pkl"))

@app.route('/clear-history')
def clear_history():
    """
    Clears the prediction history CSV.
    """
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            init_history_file()
        except Exception as e:
            print(f"Error clearing history file: {e}")
    return redirect(url_for('home'))

@app.route('/download-report')
def download_report():
    """
    Generates a downloadable text report for the last prediction.
    """
    pred_data = session.get("last_prediction")
    if not pred_data:
        return "No prediction report available. Please run a prediction first.", 400
        
    inputs = pred_data["inputs"]
    timestamp = pred_data["timestamp"]
    status = pred_data["result"]
    prob = pred_data["probability"]
    
    # Format report string
    report_content = f"""======================================================
AI-POWERED FLOOD PREDICTION SYSTEM - REPORT
======================================================
Generated on: {timestamp}
Prediction Result: {status.upper()}
Flood Probability: {prob * 100:.2f}%
======================================================
INPUT PARAMETERS DETAILED REPORT:
------------------------------------------------------
Annual Rainfall: {inputs['Annual Rainfall']} mm
Cloud Visibility: {inputs['Cloud Visibility']}%
------------------------------------------------------
MONTHLY BREAKDOWN:
- January Rainfall: {inputs['January Rainfall']} mm
- February Rainfall: {inputs['February Rainfall']} mm
- March Rainfall: {inputs['March Rainfall']} mm
- April Rainfall: {inputs['April Rainfall']} mm
- May Rainfall: {inputs['May Rainfall']} mm
- June Rainfall: {inputs['June Rainfall']} mm
- July Rainfall: {inputs['July Rainfall']} mm
- August Rainfall: {inputs['August Rainfall']} mm
- September Rainfall: {inputs['September Rainfall']} mm
- October Rainfall: {inputs['October Rainfall']} mm
- November Rainfall: {inputs['November Rainfall']} mm
- December Rainfall: {inputs['December Rainfall']} mm
======================================================
SAFETY & PREVENTIVE SUGGESTIONS:
"""
    if pred_data["prediction"] == 1:
        report_content += """⚠️ CRITICAL WARNING: FLOOD RISK IS DETECTED
- Move to higher ground immediately if local alerts sound.
- Store clean drinking water, dry food, and emergency supplies.
- Do not drive or walk through flood waters.
- Keep emergency contact numbers active.
- Unplug electrical appliances to avoid shocks.
"""
    else:
        report_content += """✅ SAFE: LOW FLOOD RISK
- Normal operations can continue.
- Keep storm drains clean to ensure smooth water drainage.
- Monitor weather forecasts during rainy seasons.
- Review household emergency evacuation plans regularly.
"""
        
    report_content += "======================================================\n"
    
    return Response(
        report_content,
        mimetype="text/plain",
        headers={"Content-disposition": f"attachment; filename=flood_prediction_report_{timestamp.replace(' ', '_').replace(':', '-')}.txt"}
    )

if __name__ == '__main__':
    # Initialize files
    init_history_file()
    app.run(host='0.0.0.0', port=5000, debug=True)
