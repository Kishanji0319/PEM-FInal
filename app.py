import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load Pickle Files

with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("categorical_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.title("🎓 Student Performance Prediction Web App")

st.write("This app predicts a student's **Average Score** and whether they **Pass or Fail** based on the average score.")


# Input Features

# Collect categorical inputs
gender = st.selectbox("Gender", encoders["gender"].classes_)
race = st.selectbox("Race/Ethnicity", encoders["race/ethnicity"].classes_)
parent_edu = st.selectbox("Parental Level of Education", encoders["parental level of education"].classes_)
lunch = st.selectbox("Lunch Type", encoders["lunch"].classes_)
prep = st.selectbox("Test Preparation Course", encoders["test preparation course"].classes_)

# Collect numerical inputs
math_score = st.number_input("Math Score", min_value=0, max_value=100, step=1)
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)


# Encoding categorical inputs

def encode_input(col, value):
    return encoders[col].transform([value])[0]

gender_encoded = encode_input("gender", gender)
race_encoded = encode_input("race/ethnicity", race)
parent_edu_encoded = encode_input("parental level of education", parent_edu)
lunch_encoded = encode_input("lunch", lunch)
prep_encoded = encode_input("test preparation course", prep)


# Prepare input for model

# Adjust column order to match training data
input_data = np.array([[gender_encoded, race_encoded, parent_edu_encoded,
                        lunch_encoded, prep_encoded, math_score, 
                        reading_score, writing_score]])

# Prediction

if st.button("Predict Performance"):
    avg_score = model.predict(input_data)[0]
    result = "✅ Pass" if avg_score >= 40 else "❌ Fail"

    st.subheader(f"📊 Predicted Average Score: {avg_score:.2f}")
    st.subheader(f"🎯 Result: {result}")
