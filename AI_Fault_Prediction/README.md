# AI-Based Fault Prediction

Predicts whether a machine is likely to fail from sensor readings
(temperature, vibration, pressure, rpm, current, operating hours)
using a Random Forest classifier.

## Setup
    pip install -r requirements.txt

## Run
    python fault_prediction.py

## Output files
- sensor_data.csv         - generated dataset
- fault_model.joblib      - trained model + scaler
- feature_importance.png  - chart of most important sensors

## Using your own data
Replace generate_data() with pd.read_csv("your_file.csv"). The CSV needs a
"fault" column (0/1); update the FEATURES list to match your columns.
