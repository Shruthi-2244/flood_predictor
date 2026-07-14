import os
import joblib
import numpy as np
import pandas as pd

# Set matplotlib backend to Agg to run headlessly without window interface issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from utils.preprocessing import load_and_preprocess_data, prepare_train_test_split

def generate_eda_plots(df, X_train, model_rf, static_plots_dir):
    """
    Generates and saves the EDA plots as required.
    """
    os.makedirs(static_plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Flood', data=df, hue='Flood', legend=False, palette='coolwarm')
    plt.title('Class Distribution: Flood (1) vs No Flood (0)')
    plt.xlabel('Flood Occurred')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No Flood (0)', 'Flood (1)'])
    plt.tight_layout()
    plt.savefig(os.path.join(static_plots_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    
    # 2. Histogram of Annual Rainfall
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Annual Rainfall'], kde=True, color='dodgerblue', bins=30)
    plt.title('Distribution of Annual Rainfall')
    plt.xlabel('Annual Rainfall (mm)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(static_plots_dir, 'histogram.png'), dpi=150)
    plt.close()
    
    # 3. Rainfall Distribution by Month
    months = [
        "January Rainfall", "February Rainfall", "March Rainfall", "April Rainfall",
        "May Rainfall", "June Rainfall", "July Rainfall", "August Rainfall",
        "September Rainfall", "October Rainfall", "November Rainfall", "December Rainfall"
    ]
    month_names = [m.split()[0] for m in months]
    monthly_means = df[months].mean()
    monthly_stds = df[months].std()
    
    plt.figure(figsize=(10, 5))
    plt.errorbar(month_names, monthly_means, yerr=monthly_stds, fmt='-o', color='teal', 
                 ecolor='lightblue', elinewidth=2, capsize=4, label='Mean Monthly Rainfall')
    plt.title('Monthly Rainfall Profile (Mean & Std Dev)')
    plt.xlabel('Month')
    plt.ylabel('Rainfall (mm)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(static_plots_dir, 'rainfall_distribution.png'), dpi=150)
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, 
                cbar_kws={"shrink": .8}, annot_kws={"size": 8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(static_plots_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    
    # 5. Feature Importance (using Random Forest)
    importances = model_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_train.columns
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=features[indices], palette='viridis', hue=features[indices], legend=False)
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(static_plots_dir, 'feature_importance.png'), dpi=150)
    plt.close()
    
    print("EDA plots saved successfully under static/plots/.")

def generate_evaluation_plots(best_model_name, y_test, y_pred, y_prob, models_probs, static_plots_dir):
    """
    Generates and saves model-specific evaluation plots.
    """
    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Flood', 'Flood'], yticklabels=['No Flood', 'Flood'],
                annot_kws={"size": 14})
    plt.title(f'Confusion Matrix ({best_model_name})', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(static_plots_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # 7. ROC Curve comparison
    plt.figure(figsize=(8, 6))
    for model_name, probs in models_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_score = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(static_plots_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    
    print("Model evaluation plots saved successfully under static/plots/.")

def main():
    print("Starting Model Training and Evaluation...")
    
    # 1. Load & Preprocess Data
    csv_path = "dataset/flood.csv"
    X, y = load_and_preprocess_data(csv_path)
    
    # Reload dataframe directly for full EDA calculations (preserving non-scaled states)
    df_raw = pd.read_csv(csv_path)
    # Handle missing values in df_raw for plotting consistency
    for col in df_raw.columns:
        if col != "Flood":
            df_raw[col] = df_raw[col].fillna(df_raw[col].median())
            
    # 2. Train-Test Split and Scaling
    X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(X, y)
    
    # 3. Initialize Models
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    # 4. Train and Evaluate Each Model
    results = {}
    models_probs = {}
    fitted_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred # fallback if prob not available
            
        models_probs[name] = y_prob
        
        # Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": auc,
            "y_pred": y_pred,
            "y_prob": y_prob
        }
        
    # 5. Display Comparison Table
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            "Model": name,
            "Accuracy": f"{metrics['Accuracy']:.4f}",
            "Precision": f"{metrics['Precision']:.4f}",
            "Recall": f"{metrics['Recall']:.4f}",
            "F1 Score": f"{metrics['F1 Score']:.4f}",
            "ROC AUC": f"{metrics['ROC AUC']:.4f}"
        })
        
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + "="*50)
    print("MODEL COMPARISON TABLE")
    print("="*50)
    print(comparison_df.to_string(index=False))
    print("="*50 + "\n")
    
    # Save comparison as a CSV for backend presentation if needed
    os.makedirs("model", exist_ok=True)
    comparison_df.to_csv("model/model_comparison.csv", index=False)
    
    # 6. Automatically select the best model (based on F1 Score)
    best_model_name = max(results, key=lambda k: results[k]["F1 Score"])
    best_model = fitted_models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"--> Automatically selected the best model: {best_model_name} (F1 Score: {best_metrics['F1 Score']:.4f})")
    
    # Save the best model
    joblib.dump(best_model, "model/flood_model.pkl")
    print(f"Saved best model to 'model/flood_model.pkl'")
    
    # 7. Generate Plots
    static_plots_dir = "static/plots"
    # Ensure RF is trained specifically for feature importance if best isn't tree/rf
    # We will use RF from our fitted dictionary
    generate_eda_plots(df_raw, X_train, fitted_models["Random Forest"], static_plots_dir)
    generate_evaluation_plots(
        best_model_name, y_test, 
        best_metrics["y_pred"], 
        best_metrics["y_prob"], 
        models_probs, static_plots_dir
    )
    
    # Save metadata about the run
    summary_data = {
        "best_model": best_model_name,
        "accuracy": float(best_metrics["Accuracy"]),
        "precision": float(best_metrics["Precision"]),
        "recall": float(best_metrics["Recall"]),
        "f1_score": float(best_metrics["F1 Score"]),
        "roc_auc": float(best_metrics["ROC AUC"])
    }
    joblib.dump(summary_data, "model/model_metadata.pkl")
    print("Training process finished successfully!")

if __name__ == "__main__":
    main()
