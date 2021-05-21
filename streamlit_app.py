
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)
risk_thresh = 0.382
high_risk_thresh = risk_thresh * 2

def st_shap(ClientID, height=None):
    client = X2_comb_test.loc[X2_comb_test.index == ClientID, :]
    
    fig = get_waterfall(client)
    
    st.pyplot(fig, clear_figure=False)

uploaded_model = st.file_uploader("Choose a pickle model")
if uploaded_model is not None:
    modfit_xgb = pickle.load(uploaded_model)
    
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    X2_comb_test = pd.read_csv(uploaded_file, index_col=0)
    #st.write(X2_comb_test)

    #with open("C:/Users/Antoine/Documents/GitHub/DS_PRJ7/data/pickle_xgb_model", 'rb') as file:
            #modfit_xgb = pickle.load(file)
    
    client_id_list = tuple(X2_comb_test.index.unique().tolist())

    background = shap.maskers.Independent(X2_comb_test)
    def pred_wrapper(x):
        return shap.links.identity(modfit_xgb.predict_proba(x, validate_features=False)[:,1])
    pred_explainer = shap.Explainer(pred_wrapper, background, link=shap.links.logit)
        
    def get_waterfall(x):
        shap_pred_values = pred_explainer(x)
    
        # visualize the first prediction's explanation
        return shap.plots.waterfall(shap_pred_values[0])
       
    def prediction(ClientID):   
        
        client = X2_comb_test.loc[X2_comb_test.index == ClientID, :]
        
        pred = modfit_xgb.predict_proba(client)[:,1]
    
         
        if pred >= high_risk_thresh:
            answer = 'Client très risqué, score = {}'.format(pred)
            risk_type = 2
        elif pred >= risk_thresh:
            answer = 'Client modérément risqué, score = {}'.format(pred)
            risk_type = 1
        else:
            answer = 'Client sûr, score = {}'.format(pred)
            risk_type = 0
            
        
        return answer, risk_type
    
    
    def main():       
        # front end elements of the web page 
        html_temp = """ 
        <div style ="background-color:#ABBAEA;padding:13px"> 
        <h1 style ="color:black;text-align:center;">API - Prédiction d'octroi de crédit</h1> 
        </div> 
        """
          
        # display the front end aspect
        st.markdown(html_temp, unsafe_allow_html = True) 
          
        # following lines create boxes in which user can enter data required to make prediction 
        ClientID = st.selectbox('Client ID',client_id_list)
        result =""
        interpreter_plot = ""
          
        # when 'Predict' is clicked, make the prediction and store it 
        if st.button("Predict"): 
            result, risk = prediction(ClientID) 
            interpreter_plot = st_shap(ClientID)
            if risk == 0:
                st.success('{}'.format(result))
            elif risk  == 1:
                st.warning('{}'.format(result))
            elif risk == 2:
                st.error('{}'.format(result))
            
             
    if __name__=='__main__': 
        main()
