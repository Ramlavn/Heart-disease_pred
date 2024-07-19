import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = pickle.load(open('Heart_disease_pred_two.pkl', 'rb'))

# Mapping for Chest Pain Type
chest_pain_type_map = {
    1: 'Typical Angina',
    2: 'Atypical Angina',
    3: 'Non-Anginal Type',
    4: 'Asymptomatic'
}

Sex = {
    0: 'Female',
    1: 'Male'
}

# Streamlit app
st.title('Heart Disease Prediction')

# Input controls
resting_bp = st.slider('Resting Blood Pressure', min_value=90, max_value=200, step=1, value=120)
chest_pain_type = st.selectbox('Chest Pain Type', list(chest_pain_type_map.values()))
num_vessels = st.selectbox('Number of Blood Vessels Blocked', [0, 1, 2, 3])
serum_cholesterol = st.slider('Serum Cholesterol (mg/dl)', min_value=100, max_value=400, step=1, value=200)
oldpeak_depression = st.slider('Oldpeak ST Depression (mm)', min_value=1.0, max_value=4.0, step=0.1, value=0.0)
sex = st.selectbox('Sex', list(Sex.values()))
age = st.slider('Age', min_value=25, max_value=80, step=1, value=45)
max_hr = st.slider('Max Heart Rate Achieved', min_value=80, max_value=200, step=1, value=150)
thallium_normal = st.radio('Thallium Stress Test Result', ['True', 'False'])

thallium_normal = True if thallium_normal == 'True' else False

# Prediction button
if st.button('Predict'):
    # Map selected chest pain type back to numeric value
    for key, value in chest_pain_type_map.items():
        if value == chest_pain_type:
            chest_pain_type = key
            break
    
    input_data = np.array([[resting_bp, chest_pain_type, num_vessels, serum_cholesterol, oldpeak_depression, age, max_hr, thallium_normal]])
    prediction = model.predict(input_data)
    
    # Display the result with built-in Streamlit functions
    if prediction == 1:
        st.error('Patient is predicted to have Heart Disease')
    else:
        st.success('Patient is predicted not to have Heart Disease')
