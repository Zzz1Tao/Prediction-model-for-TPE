import streamlit as st
import joblib
import numpy as np
import warnings
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

model = joblib.load("CatBoost.pkl")

with st.form("my_form"):
   Age = st.number_input('Age(year)', step=1)
   sex_option = st.selectbox('Sex', ['Male', 'Female'])
   Sex = 1 if sex_option == 'Male' else 2
   NCC = st.number_input('Nucleated Cell Count (cells/µL)', step=1)
   Eosinophil = st.number_input('Eosinophil(%)', step=1)
   TBAb_option = st.selectbox('TB-Ab(Tubercle bacillus antibody)', ['Negative', 'Weak Positive','Positive'])
   TBAb = 0 if TBAb_option == 'Negative' else (1 if TBAb_option == 'Weak Positive' else 2)
   ADA = st.number_input('ADA(U/L)')
   Chloride = st.number_input('Chloride(mmol/L)')
   Protein = st.number_input('Protein(mg/dL)')
   CEA = st.number_input('CEA(ug/L)')
   CA199 = st.number_input('CA199(U/mL)')
   SCC = st.number_input('SCC(ng/mL)')
   CK19 = st.number_input('CK19(ng/mL)')
   
   submitted = st.form_submit_button("Predict")
   if submitted:
    x_train = np.array([[Age,Sex,NCC,Eosinophil,TBAb,ADA,Chloride,Protein,CEA,CA199,CK19,SCC]])
    explainer = shap.TreeExplainer(
    model,
    )
    temp = np.round(x_train, 2)
    shap_values = explainer.shap_values(x_train) 
    shap.force_plot(
    explainer.expected_value, 
    shap_values, 
    temp,
    feature_names = ['Age','Sex','NCC','Eosinophil','TBAb','ADA','Chloride','Protein','CEA','CA199','CK19','SCC'],matplotlib=True,show=False)
    plt.rcParams['font.size'] = 14 
    plt.xticks(size=15)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.savefig("TPE force plot.png",dpi=600)
    pred = model.predict_proba(x_train)
    st.markdown("#### _Based on feature values, predicted possibility of TPE is {}%_".format(round(pred[0][1], 4)*100))
    st.image('TPE force plot.png')
