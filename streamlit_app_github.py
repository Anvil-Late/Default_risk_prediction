
import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import shap
from urllib import request
import cloudpickle as cp
import matplotlib
import matplotlib.pyplot as plt

# Removes error message for plots
st.set_option('deprecation.showPyplotGlobalUse', False)

# Parameters
risk_thresh = 0.382
high_risk_thresh = risk_thresh * 2

# Function that returns the prediction cascade plot
def st_shap(ClientID):
    client = X2_comb_test.loc[X2_comb_test.index == ClientID, :]
    
    fig = get_waterfall(client)
    
    st.pyplot(fig, clear_figure=False)

# Load dataset and model
pickle_url = "https://github.com/Anvil-Late/Default_risk_prediction/raw/main/data/pickle_xgb_model"
dataset_url = "https://github.com/Anvil-Late/Default_risk_prediction/raw/main/data/preprocessed_testing_set.csv"
modfit_xgb = cp.load(request.urlopen(pickle_url)) 
X2_comb_test = pd.read_csv(dataset_url, index_col=0)

# Retrieve global stats with which to compare a client
X2_comb_test_stats = X2_comb_test.describe().drop("count")

# Prepare feature importance tree
explainer = shap.TreeExplainer(modfit_xgb)
shap_values = explainer.shap_values(X2_comb_test)

# Extract all clients to create select box
client_id_list = tuple(X2_comb_test.index.unique().tolist())

with st.sidebar.beta_container():
    st.write("This is inside the container")
    # Display feature importance plot
    st.text('Importance des variables : \n')
    fig2, ax = plt.subplots(figsize = (18, 18))
    fig2 = shap.summary_plot(shap_values, X2_comb_test)
    st.pyplot(fig2, clear_figure=True)
    
    # Print final text that links to user manual
    st.markdown("Pour mieux comprendre [l'explication du score](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/main/Note%20M%C3%A9thodologique.ipynb#Cascade-d'interpr%C3%A9tation-de-pr%C3%A9diction),  [l'importance des variables](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/main/Note%20M%C3%A9thodologique.ipynb#Interpr%C3%A9teur-shap) ainsi que [la nature du modèle](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/main/Note%20M%C3%A9thodologique.ipynb#Entra%C3%AEnement-du-mod%C3%A8le),  veuillez consulter [la note méthodologique suivante](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/main/Note%20M%C3%A9thodologique.ipynb)",
                unsafe_allow_html=True)

# This will return the cascade plot
background = shap.maskers.Independent(X2_comb_test)
def pred_wrapper(x):
    return shap.links.identity(modfit_xgb.predict_proba(x, validate_features=False)[:,1])
pred_explainer = shap.Explainer(pred_wrapper, background, link=shap.links.logit)
    
def get_waterfall(x):
    shap_pred_values = pred_explainer(x)

    # visualize the first prediction's explanation
    return shap.plots.waterfall(shap_pred_values[0])

# Prediction function. Sets actions to perform when user clicks on "predict"
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
        
    
    return answer, risk_type, client

# Main API function
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:#ABBAEA;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Analyse financière pour octroi de crédit</h1> 
    </div> 
    """
      
    # Display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # Following lines create boxes in which user can enter data required to make prediction 
    ClientID = st.selectbox("Veuillez choisir l'ID d'un client ayant déposé un dossier d'emprunt",client_id_list)
    result =""
    interpreter_plot = ""
      
    # When 'Predict' is clicked, make the prediction, display it and display plots
    if st.button("Predict"): 
        result, risk, client_stats = prediction(ClientID)
        # Show client data and global statistics
        st.text('Données du client : \n')
        st.dataframe(client_stats)
        st.text("Statistiques de l'échantillon : \n")
        st.dataframe(X2_comb_test_stats)
        
        # Display score. Color depends on score
        if risk == 0:
            st.success('{}'.format(result))
        elif risk  == 1:
            st.warning('{}'.format(result))
        elif risk == 2:
            st.error('{}'.format(result))
            
        # Display waterfall plot
        st.text('Explication du score : \n')
        interpreter_plot = st_shap(ClientID)
        
        # Display feature importance plot
        st.text('Importance des variables : \n')
        fig2, ax = plt.subplots(figsize = (18, 18))
        fig2 = shap.summary_plot(shap_values, X2_comb_test)
        st.pyplot(fig2, clear_figure=True)
        
        # Print final text that links to user manual
        st.markdown("Pour mieux comprendre [l'explication du score](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/main/Note%20M%C3%A9thodologique.ipynb#Cascade-d'interpr%C3%A9tation-de-pr%C3%A9diction),  [l'importance des variables](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/main/Note%20M%C3%A9thodologique.ipynb#Interpr%C3%A9teur-shap) ainsi que [la nature du modèle](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/main/Note%20M%C3%A9thodologique.ipynb#Entra%C3%AEnement-du-mod%C3%A8le),  veuillez consulter [la note méthodologique suivante](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/main/Note%20M%C3%A9thodologique.ipynb)",
                    unsafe_allow_html=True)
        
         
if __name__=='__main__': 
    main()
