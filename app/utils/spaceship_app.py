import streamlit as st
import pandas as pd

class SpaceshipApp:
    def __init__(self, db, model):
        self.db = db
        self.model = model

    def prediction_page(self):
        st.title("Spaceship Titanic Prediction")
        
        home_planet = st.selectbox("Select Home Planet", ["Earth", "Europa", "Mars"])
        cryo_sleep = st.selectbox("Cryo Sleep?", ["Yes", "No"])
        destination = st.selectbox("Select Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        vip = st.selectbox("VIP?", ["Yes", "No"])
        total_spend = st.number_input("Total Spend", min_value=0.0, value=0.0, step=1.0)

        input_data = self.model.map_inputs(home_planet, cryo_sleep, destination, age, vip, total_spend)

        if st.button("Predict"):
            prediction_result = self.model.predict(input_data)
            st.success(f"Prediction result: {prediction_result}")
            self.db.insert_prediction(home_planet, cryo_sleep, destination, age, vip, total_spend, prediction_result)

    def display_history(self):
        st.subheader("Check the status of each entry.")
        predictions = self.db.fetch_predictions()

        if predictions:
            df = pd.DataFrame(predictions, columns=['ID', 'Home Planet', 'Cryo Sleep', 'Destination', 'Age', 'VIP', 'Total Spend', 'Transported'])
            styled_df = df.style.applymap(self.color_transport, subset=['Transported'])
            st.dataframe(styled_df)
        else:
            st.write("No predictions made yet.")

    def color_transport(self, val):
        return 'color: orange' if val == "Transported" else 'color: green'

