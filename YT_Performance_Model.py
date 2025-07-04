import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration for better aesthetics
st.set_page_config(
    page_title="YouTube Revenue Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Load the trained model and features ---
try:
    model = joblib.load('youtube_revenue_model.joblib')
    model_features = joblib.load('model_features.joblib')
    feature_means = joblib.load('feature_means.joblib')
    st.success("Model and features loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'youtube_revenue_model.joblib', 'model_features.joblib', and 'feature_means.joblib' are in the same directory as this app.")
    st.stop() # Stop the app if model files are missing

# --- Streamlit UI ---
st.title("💰 YouTube Channel Revenue Predictor")
st.markdown("""
Welcome to the YouTube Channel Revenue Predictor!
Input your channel's key performance metrics below to get an estimated revenue.
""")

st.write("---")

# Define key features for user input (a subset of all features)
input_features_display_names = {
    'Views': 'Total Views',
    'Watch Time (hours)': 'Total Watch Time (hours)',
    'Subscribers': 'Total Subscribers',
    'Video Duration': 'Average Video Duration (seconds)',
    'Likes': 'Total Likes',
    'New Comments': 'Total New Comments',
    'Shares': 'Total Shares',
    'Impressions': 'Total Impressions',
    'Video Thumbnail CTR (%)': 'Video Thumbnail Click-Through Rate (%)',
    'Average View Percentage (%)': 'Average View Percentage (%)',
    'Average View Duration': 'Average View Duration (seconds)'
}

# Create a dictionary to hold user inputs
user_inputs = {}

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

# Input fields for selected key features
with col1:
    user_inputs['Views'] = st.number_input(
        input_features_display_names['Views'],
        min_value=0.0,
        value=float(feature_means.get('Views', 0)), # Use mean as default
        step=1000.0,
        format="%.2f"
    )
    user_inputs['Watch Time (hours)'] = st.number_input(
        input_features_display_names['Watch Time (hours)'],
        min_value=0.0,
        value=float(feature_means.get('Watch Time (hours)', 0)),
        step=100.0,
        format="%.2f"
    )
    user_inputs['Subscribers'] = st.number_input(
        input_features_display_names['Subscribers'],
        min_value=0.0,
        value=float(feature_means.get('Subscribers', 0)),
        step=10.0,
        format="%.2f"
    )
    user_inputs['Video Duration'] = st.number_input(
        input_features_display_names['Video Duration'],
        min_value=0.0,
        value=float(feature_means.get('Video Duration', 0)),
        step=10.0,
        format="%.2f"
    )
    user_inputs['Likes'] = st.number_input(
        input_features_display_names['Likes'],
        min_value=0.0,
        value=float(feature_means.get('Likes', 0)),
        step=10.0,
        format="%.2f"
    )
    user_inputs['New Comments'] = st.number_input(
        input_features_display_names['New Comments'],
        min_value=0.0,
        value=float(feature_means.get('New Comments', 0)),
        step=1.0,
        format="%.2f"
    )

with col2:
    user_inputs['Shares'] = st.number_input(
        input_features_display_names['Shares'],
        min_value=0.0,
        value=float(feature_means.get('Shares', 0)),
        step=1.0,
        format="%.2f"
    )
    user_inputs['Impressions'] = st.number_input(
        input_features_display_names['Impressions'],
        min_value=0.0,
        value=float(feature_means.get('Impressions', 0)),
        step=1000.0,
        format="%.2f"
    )
    user_inputs['Video Thumbnail CTR (%)'] = st.number_input(
        input_features_display_names['Video Thumbnail CTR (%)'],
        min_value=0.0,
        max_value=100.0,
        value=float(feature_means.get('Video Thumbnail CTR (%)', 0)),
        step=0.1,
        format="%.2f"
    )
    user_inputs['Average View Percentage (%)'] = st.number_input(
        input_features_display_names['Average View Percentage (%)'],
        min_value=0.0,
        max_value=100.0,
        value=float(feature_means.get('Average View Percentage (%)', 0)),
        step=0.1,
        format="%.2f"
    )
    user_inputs['Average View Duration'] = st.number_input(
        input_features_display_names['Average View Duration'],
        min_value=0.0,
        value=float(feature_means.get('Average View Duration', 0)),
        step=10.0,
        format="%.2f"
    )

# Prepare the input DataFrame for prediction
input_df_dict = {feature: feature_means[feature] for feature in model_features} # Start with all means

# Update with user inputs for the selected features
for feature, value in user_inputs.items():
    if feature in input_df_dict: # Ensure the feature is part of the model's expected features
        input_df_dict[feature] = value

# Convert the dictionary to a DataFrame, ensuring correct column order
input_df = pd.DataFrame([input_df_dict], columns=model_features)

# --- Prediction button ---
st.write("---")
if st.button("Predict Estimated Revenue", help="Click to get the revenue prediction based on your inputs"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"**Predicted Estimated Revenue (USD): ${prediction:.2f}**")
        st.balloons() # Fun animation!
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.write("---")
st.markdown("### How to Use This App")
st.info("""
1.  **Run Part 1 code (Model Training and Saving)** in your Jupyter Notebook to generate `youtube_revenue_model.joblib`, `model_features.joblib`, and `feature_means.joblib`. Make sure these files are saved in the same directory where you'll save this Streamlit app script.
2.  **Save this code** as `streamlit_app.py` in the same directory.
3.  **Open your terminal or command prompt**, navigate to that directory, and run:
    `streamlit run streamlit_app.py`
4.  Your browser will automatically open to the Streamlit app.
""")

st.markdown("Developed with ❤️ using Python & Streamlit")