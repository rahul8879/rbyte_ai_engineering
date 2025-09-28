import streamlit as st
import pickle

# Load the KNN model
MODEL_PATH = "knn_iris_model.pkl"

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model(MODEL_PATH)

# lets load scaler too
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_scaler(path):
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler

scaler = load_scaler(SCALER_PATH)

st.title("KNN Iris Model Loader")

st.write("KNN model loaded successfully!")

# Input fields for iris features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
input_data = scaler.transform(input_data)

if st.button("Predict Iris Species"):
    prediction = model.predict(input_data)
    species = ["Setosa", "Versicolor", "Virginica"]
    st.write(f"Predicted Iris Species: {species[prediction[0]]}")
    st.balloons()

# To run the app, use the command: streamlit run app.py
