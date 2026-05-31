import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="AdaBoost Classifier")

st.title("Insurance Purchase Prediction")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "adaboost_model.pkl"
)

# Load Model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Inputs
age = st.number_input("Age", 18, 100, 25)

bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

children = st.number_input("Children", 0, 10, 0)

gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["northeast",
     "northwest",
     "southeast",
     "southwest"]
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "age":[age],
        "bmi":[bmi],
        "children":[children],
        "gender_male":[1 if gender=="male" else 0],
        "smoker_yes":[1 if smoker=="yes" else 0],
        "region_northwest":[1 if region=="northwest" else 0],
        "region_southeast":[1 if region=="southeast" else 0],
        "region_southwest":[1 if region=="southwest" else 0]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Customer is likely to purchase insurance.")
    else:
        st.error("Customer is unlikely to purchase insurance.")
