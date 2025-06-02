import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate complete synthetic dataset with all required columns
def generate_sample_data(num_records=1000):
    np.random.seed(42)
    timestamps = [datetime(2024, 6, 1) + timedelta(hours=i) for i in range(num_records)]
    
    data = {
        'timestamp': timestamps,
        'dno_alias_lv': ['UKPN_EPN'] * num_records,
        'secondary_substation_id': [f"EPN-S{str(i).zfill(7)}U{np.random.randint(1000,9999)}" for i in range(num_records)],
        'lv_feeder_id': np.random.randint(80, 90, num_records),
        'aggregated_device_count_active_lv': np.random.uniform(0.05, 0.4, num_records),
        'total_consumption_active_import_lv': np.sin(np.arange(num_records)/10 + np.random.normal(0.5, 0.1, num_records)),
        'total_consumption_reactive_import_lv': np.random.uniform(0.1, 0.5, num_records),
        'power_factor_lv': np.random.uniform(0.85, 1.0, num_records),
        'power_factor_substation': np.random.uniform(0.88, 1.02, num_records),
        'latitude_lv': np.random.uniform(51.8, 52.2, num_records),
        'longitude_lv': np.random.uniform(-0.6, -0.3, num_records),
        'anomaly_score_lv': np.zeros(num_records)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Inject anomalies
    anomaly_indices = np.random.choice(num_records, size=int(num_records * 0.05), replace=False)
    df.loc[anomaly_indices, 'total_consumption_active_import_lv'] *= 3
    df.loc[anomaly_indices, 'power_factor_lv'] *= 0.7
    df.loc[anomaly_indices, 'anomaly_score_lv'] = 1

    # Add derived features
    df['reactive_per_device_lv'] = df['total_consumption_reactive_import_lv'] * 0.2
    df['reactive_per_device_substation'] = df['total_consumption_reactive_import_lv'] * 0.18

    return df

# Save to Excel
df = generate_sample_data()
df.to_excel("sample_complete.xlsx", index=False)
print("Generated sample_complete.xlsx with all required columns")
