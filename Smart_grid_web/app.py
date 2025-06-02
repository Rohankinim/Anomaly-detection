import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = load_model("trial_47.h5")
scaler = StandardScaler()

# UI Configuration (must be first Streamlit command)
st.set_page_config(page_title="Smart Grid Anomaly Detection", layout="wide")

# Verify model input shape (moved after set_page_config)
st.write("Model input shape:", model.input_shape)  # Should show (None, 7)

st.title("🔌 Smart Grid Anomaly Detection")
st.markdown("""
    **Model**: Optuna-tuned Autoencoder  
    **Default Threshold**: 0.13092 (adjustable)
""")

# File Upload Section
with st.expander("📁 Upload Data", expanded=True):
    uploaded_file = st.file_uploader(
        "Upload energy data (CSV or Excel)",
        type=["csv", "xlsx"],
        help="File must contain required power measurement columns"
    )

# Column Requirements
REQUIRED_COLUMNS = {
    'total_consumption_active_import_lv': 'Active Power (LV)',
    'total_consumption_reactive_import_lv': 'Reactive Power (LV)',
    'power_factor_lv': 'Power Factor (LV)',
    'power_factor_substation': 'Power Factor (Substation)',
    'aggregated_device_count_active_lv': 'Active Device Count'
}

# Data Loading
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return pd.read_excel("sample_complete.xlsx")
    elif uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

try:
    df = load_data(uploaded_file)
    
    # Validate columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.info("Please use the sample data or ensure your file contains these columns")
        st.stop()
        
except Exception as e:
    st.error(f"Data loading failed: {str(e)}")
    st.stop()

# Feature Engineering
try:
    with st.spinner("Preprocessing data..."):
        # Calculate required features
        df['reactive_power_ratio'] = np.where(
            df['total_consumption_active_import_lv'] == 0, 0,
            df['total_consumption_reactive_import_lv'] / df['total_consumption_active_import_lv']
        )
        df['power_factor_avg'] = (df['power_factor_lv'] + df['power_factor_substation']) / 2
        
        # Prepare ALL features the model expects
        features = [
            'reactive_power_ratio',
            'power_factor_avg',
            'total_consumption_reactive_import_lv',
            'aggregated_device_count_active_lv',
            'power_factor_lv',
            'power_factor_substation',
            'total_consumption_active_import_lv'
        ]
        
        # Verify we have all columns
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Missing required features: {missing}")
            st.stop()
            
        X = df[features].fillna(0)
        X_scaled = scaler.fit_transform(X)
        
        # Debugging output
        st.write(f"Input shape to model: {X_scaled.shape}")  # Should be (n_samples, 7)
        
        if X_scaled.shape[1] != model.input_shape[1]:
            st.error(f"Feature dimension mismatch. Model expects {model.input_shape[1]} features, got {X_scaled.shape[1]}")
            st.stop()
            
        reconstructions = model.predict(X_scaled, verbose=0)
        errors = np.mean(np.abs(X_scaled - reconstructions), axis=1)
        
except Exception as e:
    st.error(f"Processing error: {str(e)}")
    st.stop()

# User Controls
st.sidebar.header("Detection Settings")
threshold = st.sidebar.slider(
    "Anomaly Threshold",
    min_value=0.01,
    max_value=0.5,
    value=0.13092,
    step=0.01,
    help="Higher values reduce false positives"
)
df['anomaly'] = (errors > threshold).astype(int)
anomalies = df[df['anomaly'] == 1]

# Visualization
tab1, tab2 = st.tabs(["📈 Power Flow", "🗃️ Anomaly Details"])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['timestamp'], df['total_consumption_active_import_lv'], 
            label="Active Power", linewidth=1, alpha=0.7)
    ax.scatter(anomalies['timestamp'], anomalies['total_consumption_active_import_lv'],
               color='red', label=f"Anomalies (n={len(anomalies)})", s=40)
    ax.set_title("Power Consumption with Anomaly Detection")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Active Power (kW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    st.metric("Anomaly Detection Rate", f"{len(anomalies)/len(df):.2%}")
    st.dataframe(anomalies.sort_values('timestamp').head(100))

# Export
st.sidebar.download_button(
    "💾 Download Results",
    data=df.to_csv(index=False),
    file_name="anomaly_report.csv",
    mime="text/csv"
)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
    **Sample data available**:  
    [Download sample_complete.xlsx](https://example.com/sample_complete.xlsx)
""")