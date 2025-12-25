#### For feeding data manually into the model

import joblib
import pandas as pd
import  numpy as np

model = joblib.load("models/random_forest_v1.pkl")

features = ['V17', 'V12', 'V14', 'V16', 'V10', 'V11', 'V18', 'V9', 'V7', 'V4']  #same features as training
fraud_input = {
    'V17': -5.2,  # Very low
    'V12': -4.1,
    'V14': -4.5,
    'V16': -1.0,
    'V10': -2.5,
    'V11': 3.2,   # High
    'V18': -1.2,
    'V9': -0.5,
    'V7': -1.1,
    'V4': 2.5
}

df = pd.DataFrame([fraud_input])
prediction = model.predict(df)[0]
probability = model.predict_proba(df)[0][1]

if prediction == 1:
    print(f"Fraud Detected: (Confidence: {probability:.2%})") 
else:
    print(f"Transaction Safe. (Risk Level: {probability:.2%})")   #Risk Level: How likely that it is fraud
print("-----------------------------")


