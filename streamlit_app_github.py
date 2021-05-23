
import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import shap
from urllib import request
import cloudpickle as cp
import matplotlib
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
risk_thresh = 0.382
high_risk_thresh = risk_thresh * 2

def st_shap(ClientID):
    client = X2_comb_test.loc[X2_comb_test.index == ClientID, :]
    
    fig = get_waterfall(client)
    
    st.pyplot(fig, clear_figure=False)
    
pickle_url = "https://github.com/Anvil-Late/Default_risk_prediction/raw/main/data/pickle_xgb_model"
dataset_url = "https://github.com/Anvil-Late/Default_risk_prediction/raw/main/data/preprocessed_testing_set.csv"
modfit_xgb = cp.load(request.urlopen(pickle_url)) 
X2_comb_test = pd.read_csv(dataset_url, index_col=0)

explainer = shap.TreeExplainer(modfit_xgb)
shap_values = explainer.shap_values(X2_comb_test)


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
    ClientID = st.selectbox("Veuillez choisir l'ID d'un client ayant déposé un dossier d'emprunt",client_id_list)
    result =""
    interpreter_plot = ""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result, risk = prediction(ClientID) 
        if risk == 0:
            st.success('{}'.format(result))
        elif risk  == 1:
            st.warning('{}'.format(result))
        elif risk == 2:
            st.error('{}'.format(result))
        st.text('Explication du score : \n')
        interpreter_plot = st_shap(ClientID)
        st.text('Importance des variables : \n')
        #fig2 = shap.summary_plot(shap_values, X2_comb_test)
        #st.pyplot(fig2, clear_figure=True)
        xgb_feature_importance = modfit_xgb.get_booster().get_score(importance_type="gain")
        xgb_feature_importance = pd.DataFrame.from_dict(xgb_feature_importance, orient = "index")
        xgb_feature_importance.rename(columns = {0 : "importance"}, inplace = True)
        xgb_feature_importance.sort_values("importance", ascending = True, inplace = True)
        xgb_top_features = xgb_feature_importance.tail(21)
        fig2, ax = plt.subplots(figsize = (18, 18))
        #xgb_top_features.plot(kind = "barh", ax=ax)
        #plt.yticks(range(0,len(xgb_top_features.index)), 
        #           xgb_top_features.index.map(lambda X : str(X[:40]) + "[...]"), fontsize = 14);
       # plt.legend().set_visible(False)
        #st.pyplot(fig2, clear_figure=True)
        fig2 = shap.summary_plot(shap_values, X2_comb_test)
        st.pyplot(fig2, clear_figure=True)
        
         
if __name__=='__main__': 
    main()
