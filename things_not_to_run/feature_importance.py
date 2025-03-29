# 📌 feature_importance.py
import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import shap
from data_preparation import df_train, df_test  # Import training and test data

# 🚀 Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ✅ Select Features & Target
features = df_train.drop(columns=['anomaly_score_lv', 'anomaly_score_substation', 'timestamp'], errors='ignore')
target = (df_train['anomaly_score_lv'] > 0.5).astype(int)  # Convert to binary anomaly labels

# ✅ Identify Categorical Features
categorical_cols = features.select_dtypes(include=['object']).columns

# ✅ Apply Label Encoding to Categorical Variables
if categorical_cols.any():
    logging.info("📊 Applying Label Encoding to categorical variables...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
        label_encoders[col] = le  # Store encoders for later use if needed
    logging.info("✅ Categorical Encoding Complete.")

# ✅ 1. Mutual Information
mi_scores = mutual_info_classif(features, target)
mi_scores_df = pd.DataFrame({'Feature': features.columns, 'Mutual_Information': mi_scores})
mi_scores_df = mi_scores_df.sort_values(by='Mutual_Information', ascending=False)

# ✅ 2. Feature Importance using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(features, target)
rf_importances = pd.DataFrame({'Feature': features.columns, 'Importance': rf.feature_importances_})
rf_importances = rf_importances.sort_values(by='Importance', ascending=False)

# ✅ Select Top Features (Threshold-based Selection)
top_features = rf_importances[rf_importances['Importance'] > 0.001]['Feature'].tolist()

# ✅ 3. SHAP Analysis
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(features)

# ✅ Plot Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(data=rf_importances, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importance (Random Forest)")
plt.savefig(r'C:\Users\rohan\OneDrive\Documents\ANOMALY_DETECTION\plots\random_forest_importance.png')
logging.info("✅ Feature Importance Plot Saved!")

shap.summary_plot(shap_values, features)

# ✅ Print Results
print("\n📌 Mutual Information Scores:\n", mi_scores_df)
print("\n📌 Random Forest Feature Importance:\n", rf_importances)

# ✅ Export Important Features for Import in model_selection.py
important_features = top_features  # ✅ Store globally for import

logging.info(f"✅ Selected {len(important_features)} Important Features: {important_features}")
logging.info("✅ Feature Importance Analysis Complete!")
