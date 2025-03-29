import numpy as np
import os
import joblib
import optuna
import concurrent.futures
from tensorflow.keras.models import Model # type: ignore # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from data_preparation import df_train_balanced, df_test

# ✅ Features
important_features = [
    'reactive_power_ratio', 'power_factor_avg',
    'total_consumption_reactive_import_lv', 'reactive_power_per_device',
    'total_consumption_active_import_lv', 'active_power_per_device',
    'aggregated_device_count_active_lv'
]

# ✅ Data Preparation
X_train = df_train_balanced[important_features].copy()
X_test = df_test[important_features].copy()
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_true = (df_test['anomaly_score_lv'] > 0.5).astype(int).values

# 🎯 Optuna Objective
def objective(trial):
    encoding_dim = trial.suggest_int("encoding_dim", 4, 16)
    hidden_1 = trial.suggest_int("hidden_1", 16, 64)
    hidden_2 = trial.suggest_int("hidden_2", 8, 32)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.4)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu"])
    batch_size = trial.suggest_categorical("batch", [32, 64])
    loss_fn = trial.suggest_categorical("loss", ["mae", "mse", "huber"])

    input_dim = X_train_scaled.shape[1]
    inp = Input(shape=(input_dim,))
    x = Dense(hidden_1)(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x) if activation == "leaky_relu" else Dense(hidden_1, activation=activation)(x)
    x = Dense(hidden_2)(x)
    x = Dropout(dropout_rate)(x)
    encoded = Dense(encoding_dim)(x)
    x = Dense(hidden_2)(encoded)
    x = Dropout(dropout_rate)(x)
    x = Dense(hidden_1)(x)
    out = Dense(input_dim, activation='linear')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_fn)

    os.makedirs("optuna_models_deep", exist_ok=True)
    model_path = f"optuna_models_deep/trial_{trial.number}.h5"
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True)
    ]

    model.fit(
        X_train_scaled, X_train_scaled,
        validation_data=(X_test_scaled, X_test_scaled),
        epochs=100,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    preds = model.predict(X_test_scaled, batch_size=512, verbose=0)
    reconstruction_error = np.mean(np.abs(preds - X_test_scaled), axis=1)

    fpr, tpr, thresholds = roc_curve(y_true, reconstruction_error)
    optimal_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred = (reconstruction_error > optimal_thresh).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, reconstruction_error)

    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    trial.set_user_attr("f1", f1)
    trial.set_user_attr("auc", auc)
    trial.set_user_attr("threshold", optimal_thresh)
    trial.set_user_attr("model_path", model_path)

    return (precision + recall + auc) / 3  # Balanced optimization

# ✅ Safe multi-threaded execution (no freezing on Windows)
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="deep_autoencoder_auc_precision_recall")
    study.optimize(objective, n_trials=50, n_jobs=1)


    # 💾 Save study
    joblib.dump(study, "optuna_study_deep_autoencoder.pkl")

    # 🏆 Best Trial Summary
    best = study.best_trial
    print(f"\n✅ Best Trial: {best.number}")
    print(f"Precision: {best.user_attrs['precision']:.4f}")
    print(f"Recall:    {best.user_attrs['recall']:.4f}")
    print(f"F1 Score:  {best.user_attrs['f1']:.4f}")
    print(f"AUC-ROC:   {best.user_attrs['auc']:.4f}")
    print(f"Threshold: {best.user_attrs['threshold']:.5f}")
    print(f"Model Path: {best.user_attrs['model_path']}")
    print(f"Hyperparameters: {best.params}")


# output: trail 47 with best results 