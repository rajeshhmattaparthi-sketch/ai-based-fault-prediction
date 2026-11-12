"""
AI-Based Fault Prediction (simple base project)
-----------------------------------------------
Predicts whether a machine is about to FAIL from sensor readings
(temperature, vibration, pressure, rpm, current, operating hours).

Pipeline:
  1. Generate synthetic sensor data (replace with your CSV if you have one)
  2. Preprocess (train/test split + scaling)
  3. Train a Random Forest classifier
  4. Evaluate (accuracy, confusion matrix, report, feature importance)
  5. Save the model and predict on new readings

Install:  pip install numpy pandas scikit-learn matplotlib joblib
Run:      python fault_prediction.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42
FEATURES = ["temperature", "vibration", "pressure", "rpm", "current", "operating_hours"]


# ---------------------------------------------------------------- 1. DATA
def generate_data(n_samples: int = 5000) -> pd.DataFrame:
    """Create synthetic sensor data. Faulty machines run hotter, vibrate more, etc."""
    rng = np.random.default_rng(RANDOM_STATE)

    df = pd.DataFrame({
        "temperature":     rng.normal(70, 10, n_samples),      # deg C
        "vibration":       rng.normal(3.0, 1.0, n_samples),    # mm/s
        "pressure":        rng.normal(100, 15, n_samples),     # psi
        "rpm":             rng.normal(1500, 200, n_samples),
        "current":         rng.normal(12, 2.5, n_samples),     # amps
        "operating_hours": rng.uniform(0, 10000, n_samples),
    })

    # Hidden rule that decides the fault risk (plus noise)
    risk = (
        0.04 * (df["temperature"] - 70)
        + 0.60 * (df["vibration"] - 3.0)
        + 0.01 * np.abs(df["pressure"] - 100)
        + 0.15 * (df["current"] - 12)
        + 0.0003 * df["operating_hours"]
        + rng.normal(0, 0.5, n_samples)
    )
    df["fault"] = (risk > np.percentile(risk, 85)).astype(int)  # ~15% faults
    return df


# ------------------------------------------------------------ 2-3. TRAINING
def train_model(df: pd.DataFrame):
    X, y = df[FEATURES], df["fault"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",   # faults are rare, so balance the classes
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)
    return model, scaler, X_test_s, y_test


# ------------------------------------------------------------ 4. EVALUATION
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred), "\n")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fault"]))

    # Which sensors matter most?
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    importance.plot(kind="barh", title="Feature Importance", color="steelblue")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print("Saved feature_importance.png")


# ------------------------------------------------------------ 5. PREDICTION
def predict_fault(reading: dict, model_path: str = "fault_model.joblib") -> None:
    bundle = joblib.load(model_path)
    X = pd.DataFrame([reading])[FEATURES]
    prob = bundle["model"].predict_proba(bundle["scaler"].transform(X))[0, 1]
    status = "FAULT LIKELY" if prob >= 0.5 else "Normal"
    print(f"Fault probability: {prob:.1%} -> {status}")


if __name__ == "__main__":
    data = generate_data()
    data.to_csv("sensor_data.csv", index=False)
    print(f"Dataset: {data.shape}, fault rate = {data['fault'].mean():.1%}\n")

    model, scaler, X_test, y_test = train_model(data)
    evaluate(model, X_test, y_test)

    joblib.dump({"model": model, "scaler": scaler}, "fault_model.joblib")
    print("Saved fault_model.joblib\n")

    # Demo predictions
    print("Healthy machine:")
    predict_fault({"temperature": 68, "vibration": 2.5, "pressure": 98,
                   "rpm": 1500, "current": 11.5, "operating_hours": 1200})

    print("Degraded machine:")
    predict_fault({"temperature": 95, "vibration": 6.5, "pressure": 130,
                   "rpm": 1450, "current": 17, "operating_hours": 9000})
