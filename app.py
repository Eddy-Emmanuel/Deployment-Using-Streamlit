import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import joblib

st.markdown("<center><h1>Diabetes Prediction</h1></center>", unsafe_allow_html=True)

col1, col2, col3 =  st.columns(3)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=None)

with col2:
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=None)

with col3:
    bloodpressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=None)

with col1:
    skinthickness = st.number_input("Skin thickness", min_value=0, max_value=99, value=None)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=846, value=None)

with col3:
    bmi = st.number_input("BMI", min_value=0, value=None)

with col1:
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=None, format="%f")

with col2:
    age = st.number_input("Age", min_value=1, value=None, format="%d")

collected_data = np.array([pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, DiabetesPedigreeFunction, age])

if None not in collected_data:
    scaled_data = zscore(collected_data)

    model = st.selectbox("Select Model",
                        ["LogisticRegression", "KNeighboursClassifier", "GaussianNB", "SVC"])

    if model == "LogisticRegression":
        model_ = joblib.load("LogisticRegressionModel.pkl", "r")
    elif model == "KNeighboursClassifier":
        model_ =joblib.load("KNeighboursClassifierModel.pkl", "r")
    elif model == "GaussianNB":
        model_ = joblib.load("GaussianNBModel.pkl", "r")
    else:
        model_ = joblib.load("SVCModel.pkl", "r")

    prediction = model_.predict(scaled_data.reshape(1, -1))[0]

    if st.button("Predict"):
        if prediction == 0:
            st.text("No Diabetes")
            st.balloons()
        else:
            st.text("Diabebtes present, do take care of yourself.")



