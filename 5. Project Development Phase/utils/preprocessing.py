import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def generate_synthetic_data(filepath="dataset/flood.csv", num_samples=2000, random_state=42):
    """
    Generates a realistic synthetic dataset for flood prediction.
    Models monsoonal rainfall distribution and correlation with flooding.
    """
    np.random.seed(random_state)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data = []
    for _ in range(num_samples):
        # Wet season months (Jun, Jul, Aug, Sep) have higher rainfall
        # Dry season months (Jan, Feb, Mar, Nov, Dec) have lower rainfall
        # Transition months (Apr, May, Oct) have moderate rainfall
        jan = np.random.gamma(2, 15)  # mean ~30mm
        feb = np.random.gamma(2, 15)  # mean ~30mm
        mar = np.random.gamma(3, 15)  # mean ~45mm
        apr = np.random.gamma(5, 20)  # mean ~100mm
        may = np.random.gamma(8, 25)  # mean ~200mm
        jun = np.random.gamma(12, 35) # mean ~420mm (Monsoon starts)
        jul = np.random.gamma(15, 40) # mean ~600mm (Peak monsoon)
        aug = np.random.gamma(14, 38) # mean ~532mm
        sep = np.random.gamma(10, 30) # mean ~300mm
        oct_ = np.random.gamma(6, 20) # mean ~120mm
        nov = np.random.gamma(3, 15)  # mean ~45mm
        dec = np.random.gamma(2, 15)  # mean ~30mm
        
        # Calculate true annual rainfall
        monthly_rainfall = [jan, feb, mar, apr, may, jun, jul, aug, sep, oct_, nov, dec]
        annual = sum(monthly_rainfall)
        
        # Cloud visibility is lower when rainfall is high (fog/heavy clouds)
        # Higher rainfall -> Lower visibility (range 10 to 95)
        base_visibility = 95 - (annual / 40.0)
        visibility = np.clip(base_visibility + np.random.normal(0, 5), 10, 95)
        
        # Non-linear flood probability logic:
        # High monsoon peaks, heavy annual rainfall, and low visibility increase flood risk
        monsoon_total = jun + jul + aug + sep
        max_monthly = max(monthly_rainfall)
        
        # Score calculation for flood probability
        score = (monsoon_total / 2000.0) * 0.4 + (max_monthly / 750.0) * 0.3 + ((100 - visibility) / 90.0) * 0.3
        
        # Add random noise to probability
        prob = 1 / (1 + np.exp(-12 * (score - 0.55))) # Sigmoid function centered around 0.55
        
        flood = 1 if np.random.rand() < prob else 0
        
        data.append([
            annual, visibility, jan, feb, mar, apr, may, jun, jul, aug, sep, oct_, nov, dec, flood
        ])
        
    columns = [
        "Annual Rainfall", "Cloud Visibility", "January Rainfall", "February Rainfall",
        "March Rainfall", "April Rainfall", "May Rainfall", "June Rainfall",
        "July Rainfall", "August Rainfall", "September Rainfall", "October Rainfall",
        "November Rainfall", "December Rainfall", "Flood"
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Introduce some artificial missing values (e.g., 1%) to demonstrate handling of missing data
    for col in columns[:-1]:
        mask = np.random.rand(*df[col].shape) < 0.01
        df.loc[mask, col] = np.nan
        
    df.to_csv(filepath, index=False)
    print(f"Synthetic dataset created successfully at '{filepath}' with {num_samples} records.")
    return df

def load_and_preprocess_data(filepath="dataset/flood.csv"):
    """
    Loads dataset, handles missing values, removes duplicates, handles outliers,
    and returns features and targets.
    """
    if not os.path.exists(filepath):
        print(f"Dataset not found at {filepath}. Generating synthetic dataset...")
        df = generate_synthetic_data(filepath)
    else:
        df = pd.read_csv(filepath)
        
    # 1. Remove Duplicates
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_shape[0]:
        print(f"Removed {initial_shape[0] - df.shape[0]} duplicate records.")
        
    # 2. Handle Missing Values
    # Fill missing values with the median for numeric features
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Filling with column medians...")
        for col in df.columns:
            if col != "Flood":
                df[col] = df[col].fillna(df[col].median())
                
    # 3. Detect and Handle Outliers (IQR Method)
    # Only bound outliers to 1.5 * IQR to prevent model skewing, but keep valid extreme rainfall events
    feature_cols = [c for c in df.columns if c != "Flood"]
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Cap outliers instead of dropping to retain monsoonal extremes reasonably
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        
    # Split into features and target
    X = df[feature_cols]
    y = df["Flood"]
    
    return X, y

def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits features and target into train and test sets, and scales features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature Scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Save the scaler
    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/scaler.pkl")
    print("Scaler saved to 'model/scaler.pkl'")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
