import streamlit as st
import pickle
import numpy as np


# Load Pickle Files

with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("categorical_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🏡",  # Unicode emoji or path to a custom icon
)

import base64


def get_base64_of_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img_path = "Student Performance Predictor.png"
img_base64 = get_base64_of_image(img_path)


st.markdown("""
    <style>
            
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.st-emotion-cache-1itdyc2.eczjsme18 > div.st-emotion-cache-6qob1r.eczjsme11 > div.st-emotion-cache-1gwvy71.eczjsme12 > div > div > div > div > div:nth-child(2) > div > label > div > p{
            font-size : 25px;
            }

   #root > div:nth-child(1) > div.withScreencast > div > div > div > section.st-emotion-cache-1itdyc2.eczjsme18 > div.st-emotion-cache-6qob1r.eczjsme11 > div.st-emotion-cache-1gwvy71.eczjsme12 > div > div > div > div > div:nth-child(2) > div > div > label:nth-child(1) > div.st-bc.st-br.st-bs.st-bt.st-bu.st-bv.st-bw.st-bx.st-by.st-bz.st-c0 > div > p{ 
        font-size : 23px;
        font-weight: bold;
    }
            

    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.st-emotion-cache-1itdyc2.eczjsme18 > div.st-emotion-cache-6qob1r.eczjsme11 > div.st-emotion-cache-1gwvy71.eczjsme12 > div > div > div > div > div:nth-child(2) > div > div > label:nth-child(2) > div.st-bc.st-br.st-bs.st-bt.st-bu.st-bv.st-bw.st-bx.st-by.st-bz.st-c0 > div > p{
            font-size : 23px;
            font-weight: bold;}
            
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.st-emotion-cache-1itdyc2.eczjsme18 > div.st-emotion-cache-6qob1r.eczjsme11 > div.st-emotion-cache-1gwvy71.eczjsme12 > div > div > div > div > div:nth-child(2) > div > div > label:nth-child(3) > div.st-bc.st-br.st-bs.st-bt.st-bu.st-bv.st-bw.st-bx.st-by.st-bz.st-c0 > div > p{
            font-size : 23px;
            font-weight: bold;}

    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.st-emotion-cache-1itdyc2.eczjsme18 > div.st-emotion-cache-6qob1r.eczjsme11 > div.st-emotion-cache-1gwvy71.eczjsme12 > div > div > div > div > div:nth-child(2) > div > div > label:nth-child(4) > div.st-bc.st-br.st-bs.st-bt.st-bu.st-bv.st-bw.st-bx.st-by.st-bz.st-c0 > div > p
            {
            font-size : 23px;
            font-weight: bold;}
    </style>""",
    unsafe_allow_html=True)


# Sidebar Navigation

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to ⬇️", ["🏠 Home", "📊 Prediction App", "📓 EDA Notebook", "ℹ️ About"])


# HOME PAGE

if page == "🏠 Home":
    st.markdown(
    """
    <style>
    .custom-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 300px;        /* change width */
        height: auto;        /* auto adjust height */
        border-radius: 15px; /* rounded corners */
        box-shadow: 0px 4px 12px rgba(0,0,0,0.4); /* shadow */
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
    f'<img src="data:image/png;base64,{img_base64}" class="custom-img">',
    unsafe_allow_html=True
    )


    st.markdown(
        """
        <h1 style="text-align: center; color: #2E86C1;">🎓 Student Performance Prediction</h1>
        <p style="text-align: center; font-size:18px;">
        Welcome to the <b>Student Performance Prediction App</b>.<br>
        This app uses <b>Machine Learning</b> to predict students' 
        <b>Average Score</b> and based on average score predict whether they <b>Pass</b> or <b>Fail</b>.
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )


# PREDICTION PAGE

elif page == "📊 Prediction App":
    st.header("📊 Student Performance Prediction")

    st.write("### 🔽 Enter Student Details:")

    # Input features
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    race = st.selectbox("Race/Ethnicity", encoders["race/ethnicity"].classes_)
    parent_edu = st.selectbox("Parental Level of Education", encoders["parental level of education"].classes_)
    lunch = st.selectbox("Lunch Type", encoders["lunch"].classes_)
    prep = st.selectbox("Test Preparation Course", encoders["test preparation course"].classes_)

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
    input_data = np.array([[gender_encoded, race_encoded, parent_edu_encoded,lunch_encoded, prep_encoded, math_score, 
    reading_score, writing_score]])

    # Prediction
    if st.button("🔮 Predict Performance"):
        avg_score = model.predict(input_data)[0]
        result = "✅ Pass" if avg_score >= 40 else "❌ Fail"

        # st.success(f"📊 Predicted Average Score: {avg_score:.2f}")
        # st.info(f"🎯 Result: {result}")
        st.subheader(f"📊 Predicted Average Score: {avg_score:.2f}")
        st.subheader(f"🎯 Result: {result}")

elif page == "📓 EDA Notebook":
    st.header("📓 Student Performance EDA Notebook")
    with open("eda.html", "r", encoding="utf-8") as f:
        notebook_html = f.read()
    st.components.v1.html(notebook_html, height=800, scrolling=True)


# ABOUT PAGE

elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")
    st.markdown(
        """
        **Dataset:** [StudentsPerformance - Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)  
        **Features Used:** Gender, Race/Ethnicity, Parental Education, Lunch, Test Preparation, Subject Scores  
        **Model:** Linear Regression (Pickle Saved)  
        **Prediction:** Average Score + Pass/Fail Status  

        🛠 Built with **Streamlit, Scikit-learn, and Python**  
        """
    )
