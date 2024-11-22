import streamlit as st
import joblib
import numpy as np
import warnings
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


model = joblib.load("CatBoost.pkl")

with st.form("my_form"):
   Age = st.number_input('Age')
   Sex = st.number_input('Sex')
   NCC = st.number_input('NCC')
   Eosinophil = st.number_input('Eosinophil')
   TBAb = st.number_input('TB Ab')
   ADA = st.number_input('ADA')
   Chloride = st.number_input('Chloride')
   Protein = st.number_input('Protein')
   CEA = st.number_input('CEA')
   CA199 = st.number_input('CA199')
   SCC = st.number_input('SCC')
   CK19 = st.number_input('CK19')

   submitted = st.form_submit_button("Predict")
   if submitted:
      x_train = np.array([[Age,Sex,NCC,Eosinophil,TBAb,ADA,Chloride,Protein,CEA,CA199,CK19,SCC]])
      explainer = shap.TreeExplainer(
      model,
      data=x_train,
      feature_perturbation="interventional",
      model_output="probability",
      )
      shap_values = explainer.shap_values(x_train) 
      temp = np.round(x_train, 2)
      shap.force_plot(explainer.expected_value, shap_values,temp,
         features, matplotlib=True, show=False)
      # plt.xticks(fontproperties='Times New Roman', size=15)
      # plt.yticks(fontproperties='Times New Roman', size=20)
      plt.tight_layout()
      plt.savefig("TPE force plot.png",dpi=600)
      pred = model.predict_proba(x_train)
      st.markdown("#### _Based on feature values, predicted possibility of TPE is {}%_".format(round(pred[0][1], 4)*100))
      st.image('TPE force plot.png')
