import streamlit as st
import pickle
import numpy as np
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'ml_model/spaceship_pipeline.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Mappings for input conversion
home_planet_mapping = {'Earth': 1, 'Europa': 2, 'Mars': 3}
destination_mapping = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}

# Streamlit UI
st.title("Spaceship Titanic Prediction")

# User input
home_planet = st.selectbox("Select Home Planet", list(home_planet_mapping.keys()))
cryo_sleep = st.selectbox("Cryo Sleep?", ["Yes", "No"])
destination = st.selectbox("Select Destination", list(destination_mapping.keys()))
age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
vip = st.selectbox("VIP?", ["Yes", "No"])
total_spend = st.number_input("Total Spend", min_value=0.0, value=0.0, step=1.0)

# Button to make prediction
if st.button("Predict"):
    # Convert user input to expected format
    cryo_sleep = 1 if cryo_sleep == "Yes" else 0
    vip = 1 if vip == "Yes" else 0

    # Create input array for the model
    input_data = np.array([[home_planet_mapping[home_planet], cryo_sleep, destination_mapping[destination], age, vip, total_spend]])

    # Call the prediction function
    result = model.predict(input_data)

    # Display the result
    if result[0] == 1:
        st.success("Passenger predicted to have been transported!")
    else:
        st.error("Passenger not predicted to have been transported.")
