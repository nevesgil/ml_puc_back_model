import streamlit as st
import os
from utils.database import PredictionDB
from utils.spaceship_model import SpaceshipModel
from utils.spaceship_app import SpaceshipApp

# Initialize the database and the model
db = PredictionDB()
model_path = os.path.join(
    os.path.dirname(__file__), "../ml_model/spaceship_pipeline.pkl"
)
model = SpaceshipModel(model_path)

# Create the Streamlit app
app = SpaceshipApp(db, model)

# Display the prediction page and the history of predictions
app.prediction_page()
st.markdown("---")
st.title("History of Predictions")
app.display_history()
