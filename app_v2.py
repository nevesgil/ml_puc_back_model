import streamlit as st
import numpy as np
import sqlite3
import pandas as pd

# Function to create a connection to the SQLite database
def create_connection():
    conn = sqlite3.connect('predictions.db')  # This will create the DB file if it doesn't exist
    return conn

# Function to create the predictions table (if it doesn't exist)
def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            home_planet TEXT,
            cryo_sleep TEXT,
            destination TEXT,
            age INTEGER,
            vip TEXT,
            total_spend REAL,
            transported TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert a prediction into the database
def insert_prediction(home_planet, cryo_sleep, destination, age, vip, total_spend, transported):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (home_planet, cryo_sleep, destination, age, vip, total_spend, transported)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (home_planet, cryo_sleep, destination, age, vip, total_spend, transported))
    conn.commit()
    conn.close()

# Function to fetch all predictions from the database
def fetch_predictions():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions')
    predictions = cursor.fetchall()
    conn.close()
    return predictions

# Function to display the prediction page
def prediction_page():
    st.title("Spaceship Titanic Prediction")
    
    # User input
    home_planet = st.selectbox("Select Home Planet", ["Earth", "Europa", "Mars"])
    cryo_sleep = st.selectbox("Cryo Sleep?", ["Yes", "No"])
    destination = st.selectbox("Select Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    vip = st.selectbox("VIP?", ["Yes", "No"])
    total_spend = st.number_input("Total Spend", min_value=0.0, value=0.0, step=1.0)

    # Button to make prediction (just for demonstration, no actual prediction)
    if st.button("Predict"):
        # Simulated prediction result
        prediction_result = np.random.choice(["Transported", "Not Transported"])
        st.success(f"Prediction result: {prediction_result}")
        
        # Insert prediction into the database
        insert_prediction(home_planet, cryo_sleep, destination, age, vip, total_spend, prediction_result)

# Function to style the dataframe with color formatting
def color_transport(val):
    color = 'green' if val == "Transported" else 'orange'
    return f'color: {color}'

# Function to display the history of predictions
def display_history():
    st.subheader("History of Predictions")
    predictions = fetch_predictions()

    if predictions:
        # Convert fetched data into a pandas DataFrame
        df = pd.DataFrame(predictions, columns=['ID', 'Home Planet', 'Cryo Sleep', 'Destination', 'Age', 'VIP', 'Total Spend', 'Transported'])
        df = df.drop(columns=['ID'])  # Drop the ID column as it's not necessary

        # Apply color formatting to the 'Transported' column
        styled_df = df.style.applymap(color_transport, subset=['Transported'])

        # Display the dataframe as a table with Streamlit
        st.dataframe(styled_df)
    else:
        st.write("No predictions made yet.")

# Main app layout
create_table()  # Ensure the database and table are created

prediction_page()
st.markdown("---")  # Divider line
st.title("History of Predictions")
display_history()
