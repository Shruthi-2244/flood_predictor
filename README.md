# AI-Powered Flood Prediction and Early Warning System

An end-to-end Machine Learning web application designed to predict regional flood risks. The system aggregates monthly precipitation metrics, yearly totals, and atmospheric cloud visibility indexes, using high-performance classification algorithms to deliver early warning statuses.

---

## 🌟 Features
- **Multi-Model ML Comparison:** Automatically trains and evaluates **Decision Tree**, **Random Forest**, **K-Nearest Neighbors (KNN)**, and **XGBoost** classifiers.
- **Automated Model Selection:** Evaluates metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC) and deploys the best performing block automatically (based on F1-Score).
- **Responsive Dashboard UI:** Crafted in clean vanilla HTML5/CSS3 and Javascript, featuring a fully-responsive layout, glassmorphic styles, and fluid animations.
- **Dark Mode Support:** A localized theme-controller toggles styling properties and persists selections across page loads.
- **Prediction History Logs:** Stores all evaluated outputs in a CSV file and presents recent predictions in a historical log table.
- **Report Downloader:** Enables downloading full meteorology reports and custom safety protocols in text format.
- **Form Validation & Calculation:** Client-side scripts calculate annual rainfall aggregates in real-time as users type monthly metrics, checking value ranges before analysis.

---

## 📂 Project Structure
```
flood_project/
├── 1. Brainstorming & Ideation/   # Project ideas and abstract info
│   └── .gitkeep
├── 2. Requirement Analysis/       # Specs and library requisites
│   └── .gitkeep
├── 3. Project Design Phase/       # DFD diagrams and layout plan
│   └── .gitkeep
├── 4. Project Planning Phase/     # Milestone dates and timelines
│   └── .gitkeep
├── 5. Project Development Phase/  # Active Flask & ML Codebase
│   ├── app.py                     # Flask web app router
│   ├── train_model.py             # Fitting pipeline & visualizer
│   ├── predict.py                 # Predict engine CLI & API
│   ├── requirements.txt           # Python library requirements
│   ├── Procfile                   # Process command for servers
│   ├── runtime.txt                # Target Python environment
│   ├── utils/                     # Preprocessor utilities
│   │   └── preprocessing.py
│   ├── dataset/                   # Rain datasets & outputs
│   │   └── flood.csv
│   ├── model/                     # Serialized PKL files & comparison CSVs
│   │   ├── flood_model.pkl
│   │   └── scaler.pkl
│   ├── static/                    # Script, CSS, and output plots
│   └── templates/                 # HTML UI layouts
├── 6.Project Testing/             # Testing suite scripts
│   ├── .gitkeep
│   └── test_all.py
├── 7.Project Documentation/       # Explanatory system markdown
│   ├── .gitkeep
│   └── Documentation.md
├── 8.Project Demonstration/       # Recording links and assets
│   └── .gitkeep
└── README.md                      # Main readme file
```

---

## 🛠️ Installation & Setup

### 1. Clone the Project Workspace
```bash
git clone <repository-url>
cd flood_project
```

### 2. Set Up a Virtual Environment
We recommend using Python 3.10+ to maintain compatibility with modern Scikit-Learn and XGBoost wheels.

**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r "5. Project Development Phase/requirements.txt"
```

---

## 🚀 Usage Guide

### Step 1: Train Models & Generate Visuals
Run the model training pipeline:
```bash
cd "5. Project Development Phase"
python train_model.py
```

### Step 2: Test via Command Line Interface (Optional)
Evaluate a single set of precipitation parameters directly in the CLI:
```bash
python predict.py --jan 30 --feb 25 --mar 50 --jun 400 --jul 600 --aug 500 --visibility 45
```

### Step 3: Run the Web Server
Launch the Flask application locally:
```bash
python app.py
```
Open a browser and navigate to `http://localhost:5000` to interact with the early warning system.

### Step 4: Run Automated Tests
Execute verification unit tests:
```bash
cd "../6.Project Testing"
python test_all.py
```

---

## 🌍 IBM Cloud Deployment

The application is deployment-ready for **IBM Cloud Code Engine** or **IBM Cloud Foundry** utilizing the provided `Procfile` and `runtime.txt`.

### Deployment using IBM Cloud Code Engine (Recommended)

1. **Install IBM Cloud CLI:**
   Download and install the CLI from [IBM Cloud](https://cloud.ibm.com/docs/cli).
2. **Log In to IBM Cloud:**
   ```bash
   ibmcloud login -a cloud.ibm.com -g Default
   ```
3. **Target Code Engine:**
   ```bash
   ibmcloud cr login
   ibmcloud target -cf
   ```
4. **Deploy Application from Source:**
   Run the following command in the project root. IBM Code Engine will automatically detect the Python environment via `requirements.txt` and serve the web interface via the `Procfile`:
   ```bash
   ibmcloud ce app create --name aquaguard-flood-system --build-source . --port 5000
   ```
5. **Access URL:**
   Code Engine will print the live HTTPS endpoint URL upon successful build.

---

## 🔮 Future Scope
- **Live Meteorological Feed:** Integrating public API feeds (e.g., OpenWeatherMap) to fetch real-time precipitation estimates.
- **Spatial Map Projections:** Plotting hazard ratings across geographic spatial layers using Mapbox or Leaflet.
- **SMS Broadcasting:** Utilizing Twilio hooks to broadcast SMS warning alerts to localized regions under severe threat levels.
