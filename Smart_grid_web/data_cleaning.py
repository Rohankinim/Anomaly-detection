 # 📌 data_cleaning.py
import pandas as pd
import numpy as np
import logging

# 🚀 Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def load_and_prepare_data(filepath):
    """
    Loads the dataset and performs:
    - Infinite value handling ✅
    - Missing value handling ✅
    - Extreme value capping ✅
    - Keeps dataset in memory, NO modifications to the original CSV ✅
    """
    logging.info(f"🔄 Loading data from: {filepath}")
    
    # Load dataset
    df = pd.read_csv(filepath, low_memory=False)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # ✅ 1. CAP EXTREME VALUES (Prevent unrealistic spikes)
    cap_value = 1e10
    df[numeric_cols] = df[numeric_cols].clip(upper=cap_value)

    # ✅ 2. HANDLE INFINITE VALUES
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # ✅ 3. HANDLE MISSING VALUES (Interpolate, then backfill as fallback)
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').bfill()

    # ✅ 4. Ensure no NaNs remain
    df[numeric_cols] = df[numeric_cols].fillna(0)

    logging.info("✅ Data cleaning complete. Dataset remains **unchanged on disk**.")
    
    return df



def analyze_dataset(df):
    """
    Analyzes the dataset for:
    - Missing values ✅
    - Infinite values ✅
    - Duplicate rows ✅
    """
    logging.info(f"🔍 Running dataset analysis...")

    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # ✅ Check for missing values
    missing_values = numeric_df.isnull().sum()

    # ✅ Check for infinite values
    infinite_values = np.isinf(numeric_df).sum()

    # ✅ Check for duplicate rows
    duplicate_rows = df.duplicated().sum()

    # 🔥 Print Results
    print("\n🔎 **Dataset Analysis Report**")
    print("📌 Missing Values per Column:\n", missing_values)
    print("\n⚠️ Infinite Values per Column (After Fix):\n", infinite_values)
    print("\n🔄 Number of Duplicate Rows:", duplicate_rows)



import pandas as pd

# Load the dataset with predictions
file_path = r'C:\Users\rohan\OneDrive\Documents\ANOMALY_DETECTION\data\ana.csv'
df = pd.read_csv(file_path)

# Check the anomaly predictions in the test set
total_anomalies = df['anomaly_score_lv'].sum()  # Assuming 'anomaly_score_lv' represents anomalies
total_records = len(df)

# Calculate anomaly percentage
anomaly_percentage = (total_anomalies / total_records) * 100

# Display results
print(f"🔍 **Total Records in Dataset:** {total_records}")
print(f"⚠ **Total Anomalies Detected:** {int(total_anomalies)}")
print(f"📊 **Percentage of Anomalies:** {anomaly_percentage:.2f}%") 