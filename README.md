# Project Documentation: AquaGuard AI

## 1. Project Title
**AquaGuard AI: AI-Powered Flood Prediction and Early Warning System**

---

## 2. Abstract
Climate volatility and monsoonal shifts cause severe flooding events that impact infrastructure, livelihoods, and public safety. *AquaGuard AI* is an automated Machine Learning classification system designed to predict localized flooding hazards. By evaluating a profile of 14 key inputs (12 monthly rainfall variables, annual rainfall sum, and cloud visibility index), the system calculates flooding risk in real-time.

Four classification models (Decision Tree, Random Forest, KNN, and XGBoost) are automatically trained, benchmarked, and selected based on F1-Score. In our implementation, a Random Forest Classifier yielded a test accuracy of **95.25%** and an F1 Score of **0.9757**. The production system features a responsive, glassmorphic Flask web interface complete with dark mode toggle, prediction logs, validation graphs, and custom safety recommendations.

---

## 3. Problem Statement
Manual disaster alerts are frequently delayed due to complex data processing pipelines, leaving emergency teams and citizens with minimal response time. 

**AquaGuard AI** addresses this by analyzing monthly precipitation spikes, annual totals, and visibility metrics to calculate flooding risk in less than a second. This early warning system helps communities take proactive safety measures.

---

## 4. Features
- **Exploratory Data Analysis (EDA)**: Dynamic generation of 7 plots illustrating rainfall histogram curves, seasonal monsoonal trends, class ratios, correlation heatmaps, confusion matrices, ROC comparisons, and feature importances.
- **Ensemble Classifier Benchmarking**: Evaluates and compares Decision Tree, Random Forest, KNN, and XGBoost structures.
- **Automated Deployments**: Automatically identifies the best classifier based on F1-Score and saves it as `flood_model.pkl`.
- **Responsive Web Dashboard**: A modern landing interface containing early-warning hero metrics, feature details, and accuracy cards.
- **Parametric Form Panel**: Synchronized range slider and numeric inputs for cloud visibility, with automatic real-time annual rainfall calculation.
- **Action Guidelines Sheet**: Renders custom warnings, action plans, and download hooks based on model outputs.
- **History Logger**: Captures user predictions and timestamps, saving them to `dataset/prediction_history.csv` for display in the log sheet.
- **Dark Mode Support**: Fluid styling transitions saved to LocalStorage.

---

## 5. Tech Stack
- **Languages**: Python 3.10+, HTML5, CSS3, JavaScript (ES6)
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib, NumPy, Pandas
- **Visualizations**: Matplotlib, Seaborn
- **Backend Serving**: Flask, Gunicorn
- **Frontend Design**: Vanilla CSS (no dependencies), Google Fonts (Outfit, Plus Jakarta Sans)

---

## 6. Installation Steps

### Clone & Initialize Environment
```bash
# Clone the repository
git clone https://github.com/Shruthi-2244/flood_predictor.git
cd flood_predictor

# Create a virtual environment
py -m venv venv

# Activate virtual environment
# Windows PowerShell:
venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate
```

### Install Required Dependencies
```bash
pip install -r "5. Project Development Phase/requirements.txt"
```

---

## 7. Usage Instructions

### Running Model Training
To run preprocessing, train the classifiers, and generate visual charts:
```bash
cd "5. Project Development Phase"
py train_model.py
```

### Running Automated Test Suite
To run code verification checks:
```bash
cd "6.Project Testing"
py test_all.py
```

### Running Web Application
To launch the Flask localhost server:
```bash
cd "5. Project Development Phase"
py app.py
```
Open https://flood-predictor-1-v5te.onrender.com in your browser.

LINK https://flood-predictor-1-v5te.onrender.com

---

## 8. Screenshots
*(Plots are output under `static/plots/` during training)*
- **Rainfall Histogram**: `static/plots/histogram.png`
- **Correlation Heatmap**: `static/plots/correlation_heatmap.png`
- **Class Balance**: `static/plots/class_distribution.png`
- **Confusion Matrix**: `static/plots/confusion_matrix.png`
- **ROC Curves**: `static/plots/roc_curve.png`

---

## 9. Future Scope
- **Live APIs Integration**: Connect OpenWeather API to load local forecast inputs automatically.
- **Geographic Information Systems (GIS)**: Map regional hazard zones visually using Leaflet.js.
- **Notification Services**: Integrate Twilio SMS notifications to broadcast alerts to affected areas.

---

## 10. Team Member Details
- **Shruthi** ([@Shruthi-2244](https://github.com/Shruthi-2244)) - Team Member

---

## 11. Demo Video
- **Google Drive** Link - https://drive.google.com/file/d/1VQ-hBObvPfYqW8OlMMzZT59JKsvbGoho/view?usp=drive_link


