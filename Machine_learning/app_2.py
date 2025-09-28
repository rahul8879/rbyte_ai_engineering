# lets create streamlit app to load model and predict diabetes
import streamlit as st
import pickle   
# Load the KNN model
MODEL_PATH = "knn_diabetes_model.pkl"

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model(MODEL_PATH)

#we dont need scaler for regression

st.title("KNN Diabetes Model Loader")
st.write("KNN model loaded successfully!")
# lets create the sidebar for different use cases
st.sidebar.title("Diabetes Prediction")
st.sidebar.write("Enter the features to predict diabetes progression.")
# Input fields for diabetes features
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=50)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
bp = st.sidebar.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)
s1 = st.sidebar.number_input("S1", min_value=0.0, max_value=1000.0, value=150.0)
s2 = st.sidebar.number_input("S2", min_value=0.0, max_value=1000.0, value=100.0)
s3 = st.sidebar.number_input("S3", min_value=0.0, max_value=1000.0, value=50.0) 
s4 = st.sidebar.number_input("S4", min_value=0.0, max_value=1000.0, value=5.0)
s5 = st.sidebar.number_input("S5", min_value=0.0, max_value=1000.0, value=90.0)
s6 = st.sidebar.number_input("S6", min_value=0.0, max_value=1000.0, value=30.0)
s7 = st.sidebar.number_input("S7", min_value=0.0, max_value=1000.0, value=40.0)
input_data = [[age, bmi, bp, s1, s2, s3, s4, s5, s6, s7]]

# Predict button
if st.button("Predict Diabetes Progression"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Diabetes Progression: {prediction[0]:.2f}")
    st.balloons()