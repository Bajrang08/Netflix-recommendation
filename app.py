import streamlit as st
import pandas as pd
import joblib

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('netflix_duration_model.pkl')

# Load unique ratings for the dropdown
@st.cache_data
def get_ratings():
    with open('ratings.txt', 'r') as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
ratings_list = get_ratings()

# UI Layout
st.title("🎬 Netflix Movie Duration Predictor")
st.markdown("Enter the movie details below to predict how long it will be (in minutes).")

with st.form("prediction_form"):
    release_year = st.number_input("Release Year", min_value=1940, max_value=2025, value=2021)
    rating = st.selectbox("Maturity Rating", options=ratings_list)
    
    submit = st.form_submit_button("Predict Duration")

if submit:
    # Create a DataFrame for prediction matching the training format
    input_data = pd.DataFrame([[release_year, rating]], columns=['release_year', 'rating'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f"🎥 Predicted Movie Duration: **{prediction:.2f} minutes**")
    
    # Add a fun visual
    st.progress(min(int(prediction), 180) / 180)