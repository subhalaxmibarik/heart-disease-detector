import streamlit as st
import numpy as np
from model import preprocess_and_train_model, predict, fetch_heart_disease_data

def main():
    # Fetch and preprocess data, and train model
    custom_css = """
        <style>
            body {
                font-family: Arial, serif;
                font-size: 32px;
            }
            h1 {
                font-size: 24px;
                font-weight: bold;
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    model = preprocess_and_train_model()
    
    # Title of the app
    st.title('Heart Disease Detection')
    st.image("heart.png", 150)

    # Sidebar for user input
    st.sidebar.title('User Input')

    # Function to get user input
    def get_user_input():
        age = st.sidebar.slider('Age', 18, 100, 25)
        sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
        cp = st.sidebar.slider('Chest Pain Type (CP)', 0, 3, 0)
        trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
        chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 600, 200)
        fbs = st.sidebar.slider('Fasting Blood Sugar (mg/dl)', 0, 600, 70)
        restecg = st.sidebar.slider('Resting Electrocardiographic Results', 0, 200, 60)
        thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 220, 150)
        exang = st.sidebar.slider('Exercise Induced Angina', 0, 1, 0)
        oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 0.0)
        slope = st.sidebar.slider('Slope of the Peak Exercise ST Segment', 0, 2, 0)
        ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 0)
        thal = st.sidebar.slider('Thalassemia', 0, 1, 0)
        
        return age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

    # Display user input
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = get_user_input()
    st.write('Selected Input:')
    st.write('Age:', age)
    st.write('Sex:', sex)
    st.write('Chest Pain Type (CP):', cp)
    st.write('Resting Blood Pressure (mm Hg):', trestbps)
    st.write('Serum Cholesterol (mg/dl):', chol)
    st.write('Fasting Blood Sugar (mg/dl):', fbs)
    st.write('Resting Electrocardiographic Results (bpm):', restecg)
    st.write('Maximum Heart Rate Achieved:', thalach)
    st.write('Exercise Induced Angina:', exang)
    st.write('ST Depression Induced by Exercise Relative to Rest:', oldpeak)
    st.write('Slope of the Peak Exercise ST Segment(sec):', slope)
    st.write('Number of Major Vessels Colored by Fluoroscopy:', ca)
    st.write('Thalassemia:', thal)

    # Convert sex to numerical value
    sex = 1 if sex == 'Male' else 0

    # Make prediction
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])  # Ensure input data has the correct shape
    prediction = predict(model, input_data)[0]
    if trestbps > 180 or restecg > 150 or fbs > 290 or age > 70 or chol > 260 or cp == 3:
        prediction = 1

    # Display prediction
    st.write('Prediction:', 'Heart Disease' if prediction == 1 else 'No Heart Disease')

if __name__ == '__main__':
    main()
