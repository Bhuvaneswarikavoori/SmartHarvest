import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
classifier = load('naive_bayes_model.joblib')
scaler = load('scaler.joblib')

# Define custom CSS for the app
custom_css = """
<style>
    .container {
        background-color: lightblue;
    }

    h1 {
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }

    p {
        font-family: 'Arial', sans-serif;
        font-size: 18px;
    }
    
    img.rounded-image {
        border-radius: 50%;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

with st.container():
    # Add an image above the text with custom width and round shape
    st.markdown(
    f'<img src="https://www.alamy.com/portrait-of-happy-indian-farmer-with-arms-outstretched-in-rapeseed-agricultural-field-image230603167.html/150" width="150" class="rounded-image"/>', unsafe_allow_html=True
)


    # Define the app title and layout
    st.title("**Welcome to SmartHarvest!**")
    st.write("*Provide the soil conditions to get a crop recommendation*:")

    # Collect user input
    nitrogen = st.number_input("Nitrogen (%)", value=25.0, step=0.1)
    phosphorous = st.number_input("Phosphorous (%)", value=50.0, step=0.1)
    potassium = st.number_input("Potassium (%)", value=30.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", value=20.0, step=0.1)
    humidity = st.number_input("Humidity (%)", value=60.0, step=0.1)
    ph = st.number_input("pH", value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", value=200.0, step=1.0)

    # Make a prediction
    if st.button("Predict"):
        input_values = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])
        scaled_input_values = scaler.transform(input_values)
        predicted_crop = classifier.predict(scaled_input_values)
        st.write(f"The recommended crop for the given soil condition is: *{predicted_crop[0]}*")
