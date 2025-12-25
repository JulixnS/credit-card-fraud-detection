import streamlit as st
import joblib
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():    # Prevents the model from reloading everytime a slider value is adjusted
    return joblib.load("models/random_forest_v1.pkl")

model = load_model()

st.title("Credit Card Fraud Detection")
st.write("Adjust the values to see if the model flags them as fraud")

st.sidebar.header("Features")

input_data = {}
features = ['V17', 'V12', 'V14', 'V16', 'V10', 'V11', 'V18', 'V9', 'V7', 'V4']

for col in features:
    input_data[col] = st.sidebar.slider(f"{col}", -10.0, 10.0, 0.0)

input_df = pd.DataFrame([input_data])

if st.button("Analyze Transaction"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if probability > 0.5:
        st.error(f"🚨 Fraud Detected")
        st.write(f"**Confidence:** {probability:.2%}")
        st.write("This transaction is suspicious.")
    else:
        st.success(f"✅ Transaction Safe")
        st.write(f"**Risk Score:** {probability:.2%}")
        st.write("This transaction seems normal.")

    # Optional: Show the data being analyzed
    st.write("----")
    st.write("Input Data used for prediction:")
    st.dataframe(input_df)