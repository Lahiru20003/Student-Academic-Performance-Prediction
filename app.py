# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack

# Load the preprocessor pipeline and the best-performing model
try:
    preprocessor = joblib.load('models/updated_preprocessor_pipeline.pkl')
    model = joblib.load('models/updated_random_forest_model.pkl') 
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure the 'models' directory exists and contains valid model files.")
    st.stop()

# Define the input features based on the original dataset columns
gender_options = ['female', 'male']
race_options = ['group A', 'group B', 'group C', 'group D', 'group E']
parental_edu_options = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
lunch_options = ['standard', 'free/reduced']
test_prep_options = ['none', 'completed']

# Create the Streamlit app layout
st.title('Student Academic Performance Predictor')
st.write('Predict student average score based on background and preparation.')

# Input form for the user
with st.form("prediction_form"):
    st.header("Student Information")
    
    # Use selectbox for categorical features
    gender = st.selectbox('Gender', options=gender_options)
    race_ethnicity = st.selectbox('Race/Ethnicity', options=race_options)
    parental_edu = st.selectbox('Parental Level of Education', options=parental_edu_options)
    lunch = st.selectbox('Lunch Type', options=lunch_options)
    test_prep = st.selectbox('Test Preparation Course', options=test_prep_options)
    
    # Submit button
    submitted = st.form_submit_button("Predict Score")

if submitted:
    # Create a DataFrame from user inputs with corrected column names
    user_input = pd.DataFrame([[gender, race_ethnicity, parental_edu, lunch, test_prep]],
                              columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])
    
    try:
        # Use the loaded preprocessor pipeline to transform the user input
        processed_input = preprocessor.transform(user_input)
        
        # Ensure the input data has the correct number of features for the model
        num_missing_features = model.n_features_in_ - processed_input.shape[1]
        if num_missing_features > 0:
            # Add missing features as columns of zeros
            missing_features = np.zeros((processed_input.shape[0], num_missing_features))
            processed_input = hstack([processed_input, missing_features])
        
        # Make the prediction
        predicted_score = model.predict(processed_input)[0]
        
        st.subheader("Prediction Result")
        st.success(f'The predicted average score is: **{predicted_score:.2f}**')
        st.info("Note: This prediction is a point estimate based on the model's training data.")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")