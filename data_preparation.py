# 📌 data_preparation.py
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.utils import resample
from data_cleaning import load_and_prepare_data, analyze_dataset

# 🚀 Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# 🔥 Filepath
filepath = r'C:\Users\rohan\OneDrive\Documents\ANOMALY_DETECTION\data\ana.csv'

# ✅ Load and clean the dataset
df_cleaned = load_and_prepare_data(filepath)

# ✅ Timestamp handling
df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'])
df_cleaned = df_cleaned.sort_values(by='timestamp')

# ✅ Location clustering
if 'latitude_lv' in df_cleaned.columns and 'longitude_lv' in df_cleaned.columns:
    logging.info("📊 Clustering Latitude & Longitude into Regions...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_cleaned['location_cluster'] = kmeans.fit_predict(df_cleaned[['latitude_lv', 'longitude_lv']])
    logging.info("✅ Clustered locations into 5 regions.")
else:
    logging.warning("⚠️ Latitude and Longitude features not found! Skipping clustering.")

# ✅ Drop weak/irrelevant features
drop_features = [
    'day_of_week_lv', 'day_of_week_substation',
    'secondary_substation_id', 'lv_feeder_id',
    'latitude_lv', 'longitude_lv',
    'latitude_substation', 'longitude_substation',
    'month_lv', 'month_substation'
]


important_features = [
    'reactive_power_ratio', 'power_factor_avg',
    'total_consumption_reactive_import_lv', 'reactive_power_per_device',
    'total_consumption_active_import_lv', 'active_power_per_device',
    'aggregated_device_count_active_lv'
]

df_cleaned.drop(columns=[col for col in drop_features if col in df_cleaned.columns], inplace=True, errors='ignore')
logging.info(f"✅ Dropped low-importance features: {drop_features}")

# ✅ Feature Engineering
logging.info("📊 Applying Feature Engineering...")
df_cleaned['active_power_per_device'] = df_cleaned['total_consumption_active_import_lv'] / (df_cleaned['aggregated_device_count_active_lv'] + 1e-6)
df_cleaned['reactive_power_per_device'] = df_cleaned['total_consumption_reactive_import_lv'] / (df_cleaned['aggregated_device_count_reactive_lv'] + 1e-6)
df_cleaned['reactive_power_ratio'] = df_cleaned['total_consumption_reactive_import_lv'] / (df_cleaned['total_consumption_active_import_lv'] + 1e-6)
df_cleaned['power_factor_avg'] = (df_cleaned['power_factor_lv'] + df_cleaned['power_factor_substation']) / 2
df_cleaned.drop(columns=['power_factor_lv', 'power_factor_substation'], inplace=True, errors='ignore')
logging.info("✅ Derived feature computation complete.")

# ✅ Standardization
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])
logging.info("✅ Standardization complete.")

# ✅ Time-based train-test split
logging.info("📊 Splitting dataset into Training and Testing sets (Time-Series Aware)...")
train_size = int(len(df_cleaned) * 0.8)
df_train = df_cleaned.iloc[:train_size].copy()
df_test = df_cleaned.iloc[train_size:].copy()
logging.info(f"✅ Training Set: {df_train.shape[0]} samples")
logging.info(f"✅ Testing Set: {df_test.shape[0]} samples")

# ✅ Class balancing (Undersample normal, keep all anomalies)
logging.info("🔁 Performing class balancing on training set...")
normal = df_train[df_train['anomaly_score_lv'] <= 0.5]
anomaly = df_train[df_train['anomaly_score_lv'] > 0.5]

normal_downsampled = resample(
    normal,
    replace=False,
    n_samples=len(anomaly) * 3,  # adjust multiplier as needed
    random_state=42
)

df_train_balanced = pd.concat([normal_downsampled, anomaly]).sample(frac=1, random_state=42).reset_index(drop=True)
logging.info(f"✅ Balanced Training Set: {df_train_balanced.shape[0]} samples (Anomaly ratio: {df_train_balanced['anomaly_score_lv'].gt(0.5).mean():.2%})")

# ✅ Final dataset check
analyze_dataset(df_cleaned)

# ✅ Exported objects
print("\n✅ Data is cleaned, balanced, engineered, and ready for model training!")
