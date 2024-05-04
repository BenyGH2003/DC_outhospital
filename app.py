import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

columns= [['Age', 'MRS in discharge']]



model = joblib.load('DC.pkl')


st.title('Prediction of outhospital mortality of ischemic stroke patients undergoing DC :brain:')

age = st.number_input("Enter the patient age")
mrs = st.number_input("Enter the MRS of patients while discharging")



def predict(): 
    row = np.array([age,mrs]) 
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict_proba(X)
    if prediction[0][1] >= 0.3: 
        st.error('The patients is more likely not to survive, based on our model :heavy_exclamation_mark:')
    elif prediction[0][1] < 0.3:
        st.success('The patients is more likely to survive, based on our model :white_check_mark:') 
        

trigger = st.button('Predict', on_click=predict)
