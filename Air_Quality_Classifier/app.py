import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessing artifacts
artifacts = joblib.load("aqc_artifacts.joblib")

preprocessor = artifacts["preprocessor"]
model = artifacts["model"]
classes = artifacts["classes"]

st.set_page_config(
    page_title="Air Quality Classifier",
    page_icon="🌫️",
    layout="centered"
)

st.title("Air Quality Classifier")
st.write(
    "Enter pollutant measurements to estimate the air quality category."
)

st.info(
    "This is an educational model trained on historical data. "
    "Its predictions are estimates, not official air-quality readings."
)

st.subheader("Pollutant measurements")

city = st.selectbox(
    "City",
    ["Ahmedabad", "Delhi", "Mumbai", "Bengaluru", "Chennai"]
)

# Default values are placeholders for testing the prediction pipeline.
# Replace them with actual measurements before interpreting a prediction.
pollutants = {
    "PM2.5": 30.0,
    "PM10": 50.0,
    "NO": 10.0,
    "NO2": 20.0,
    "NOx": 25.0,
    "NH3": 10.0,
    "CO": 1.0,
    "SO2": 10.0,
    "O3": 30.0,
    "Benzene": 1.0,
    "Toluene": 5.0,
    "Xylene": 1.0,
}

values = {}

for pollutant, default in pollutants.items():
    values[pollutant] = st.number_input(
        pollutant,
        min_value=0.0,
        value=default,
        step=0.1
    )

if st.button("Predict air quality"):
    input_df = pd.DataFrame([{"City": city, **values}])

    processed_input = preprocessor.transform(input_df)
    probabilities = model.predict_proba(processed_input)[0]

    predicted_index = probabilities.argmax()
    predicted_class = classes[predicted_index]

    st.subheader("Prediction")
    st.write(f"Estimated AQI category: **{predicted_class}**")

    st.subheader("Class probabilities")
    probability_df = pd.DataFrame({
        "AQI category": classes,
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)

    st.bar_chart(
        probability_df.set_index("AQI category")
    )