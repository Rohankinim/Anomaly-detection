import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model # type: ignore
from data_preparation import df_test
import matplotlib.pyplot as plt

# === FIXED CONFIGURATION === 
OPTIMAL_THRESHOLD = 0.145 # Your best-performing threshold
FEATURES = [
    'reactive_power_ratio',
    'power_factor_avg',
    'total_consumption_reactive_import_lv',
    'reactive_power_per_device',
    'total_consumption_active_import_lv',
    'active_power_per_device',
    'aggregated_device_count_active_lv'
]

# === DATA LOADING ===
X_test = df_test[FEATURES].copy()
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.fillna(0, inplace=True)
y_true = (df_test['anomaly_score_lv'] > 0.5).astype(int).values

# === MODEL SETUP ===
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
model = load_model("optuna_models_deep/trial_47.h5")  # Your exact model

# === RECONSTRUCTION ERRORS ===
reconstructions = model.predict(X_test_scaled, batch_size=512, verbose=1)
reconstruction_errors = np.mean(np.abs(X_test_scaled - reconstructions), axis=1)

# === EVALUATION AT 0.15 THRESHOLD ===
y_pred = (reconstruction_errors > OPTIMAL_THRESHOLD).astype(int)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# === METRICS CALCULATION ===
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
fp_rate = fp / (fp + tn)

# === RESULTS DISPLAY ===
print("\n" + "="*50)
print(f"🔍 EVALUATION AT FIXED THRESHOLD: {OPTIMAL_THRESHOLD:.2f}")
print("="*50)
print(f"True Positives (TP): {tp} ({recall:.1%} of anomalies)")
print(f"False Positives (FP): {fp} ({fp_rate:.1%} of normals)")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")
print("\n📊 Performance Metrics:")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"TP/FP Ratio: {tp/fp:.2f}:1")

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

# === CONFUSION MATRIX PLOT ===
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Threshold={OPTIMAL_THRESHOLD:.2f})")
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ["Normal", "Anomaly"])
plt.yticks(tick_marks, ["Normal", "Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i,j]:,}", 
                 horizontalalignment="center",
                 color="white" if cm[i,j] > cm.max()/2 else "black")

plt.tight_layout()
plt.show()

# === ERROR DISTRIBUTION ANALYSIS ===
plt.figure(figsize=(10,6))
plt.hist(reconstruction_errors[y_true==0], bins=50, alpha=0.5, label='Normal', color='blue')
plt.hist(reconstruction_errors[y_true==1], bins=50, alpha=0.5, label='Anomaly', color='red')
plt.axvline(OPTIMAL_THRESHOLD, color='black', linestyle='--', 
            label=f'Threshold = {OPTIMAL_THRESHOLD:.2f}')
plt.yscale('log')
plt.title("Reconstruction Error Distribution")
plt.xlabel("Error Magnitude")
plt.ylabel("Count (log scale)")
plt.legend()
plt.show()

# === PRECISION-RECALL CURVE ===
precision_curve, recall_curve, _ = precision_recall_curve(y_true, reconstruction_errors)
plt.figure(figsize=(8,6))
plt.plot(recall_curve, precision_curve, lw=2)
plt.scatter(recall, precision, c='red', s=100, 
            label=f'Threshold {OPTIMAL_THRESHOLD:.2f}\n(P={precision:.2f}, R={recall:.2f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.legend()
plt.show()

