
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import shap


risk_thresh = 0.382
high_risk_thresh = risk_thresh * 2


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    X2_comb_test = pd.read_csv(uploaded_file, index_col=0)
    st.write(X2_comb_test)

with open("./data/pickle_xgb_model", 'rb') as file:
        modfit_xgb = pickle.load(file)

background = shap.maskers.Independent(X2_comb_test)
def pred_wrapper(x):
    return shap.links.identity(modfit_xgb.predict_proba(x, validate_features=False)[:,1])
pred_explainer = shap.Explainer(pred_wrapper, background, link=shap.links.logit)
    
def get_waterfall(x):
    shap_pred_values = pred_explainer(x)

    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_pred_values[0])
    
client_id_list = tuple(X2_comb_test.index.unique().tolist())


def prediction(ClientID):   
    
    client = X2_comb_test.loc[X2_comb_test.index == ClientID, :]
    
    pred = modfit_xgb.predict_proba(client)

     
    if pred >= high_risk_thresh:
        answer = 'Client très risqué'
    elif pred >= risk_thresh:
        answer = 'Client risqué'
    else:
        answer = 'Client peu risqué'
        
    get_waterfall(client)
    return answer


def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    ClientID = st.selectbox('Client ID',client_id_list)
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(ClientID) 
        st.success('{}'.format(result))
        
     
if __name__=='__main__': 
    main()
