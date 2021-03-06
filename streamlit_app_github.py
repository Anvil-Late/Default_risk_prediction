
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

# Top Features
top_features = ['EXT_SOURCE_2', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED', 'bureau_DAYS_CREDIT_mean',
                'bureau_AMT_CREDIT_MAX_OVERDUE_sum', 'DAYS_BIRTH', 'AMT_CREDIT',
                'NAME_EDUCATION_TYPE_Higher education', 'CODE_GENDER_F', 
                'bureau_AMT_CREDIT_SUM_DEBT_mean', 'AMT_ANNUITY', 'CODE_GENDER_M',
                'SK_DPD_DEF_max', 'DAYS_ID_PUBLISH', 'bureau_DAYS_CREDIT_max',
                'DAYS_LAST_PHONE_CHANGE', 'TOTAL_AMT_CREDIT', 'NAME_FAMILY_STATUS_Married',
                'bureau_AMT_CREDIT_SUM_DEBT_sum', 'bureau_DAYS_CREDIT_ENDDATE_max']

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

# Prepare feature importance tree
explainer = shap.TreeExplainer(modfit_xgb)
shap_values = explainer.shap_values(X2_comb_test)

# Extract all clients to create select box
client_id_list = tuple(X2_comb_test.index.unique().tolist())

# Do all predictions
all_preds = modfit_xgb.predict_proba(X2_comb_test)[:,1]

# Subsetters for low, med and high risk clients
y_low_risk = all_preds < risk_thresh
y_med_risk = (all_preds >= risk_thresh) & (all_preds < high_risk_thresh)
y_high_risk = all_preds >= high_risk_thresh

# Retrieve global stats with which to compare a client
X2_comb_test_stats = X2_comb_test.loc[:, top_features].describe().drop("count")
X2_comb_test_stats_low = X2_comb_test.loc[y_low_risk, top_features].describe().drop("count")
X2_comb_test_stats_med = X2_comb_test.loc[y_med_risk, top_features].describe().drop("count")
X2_comb_test_stats_high = X2_comb_test.loc[y_high_risk, top_features].describe().drop("count")

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
        answer = 'Client tr??s risqu??, score = {}'.format(pred)
        risk_type = 2
    elif pred >= risk_thresh:
        answer = 'Client mod??r??ment risqu??, score = {}'.format(pred)
        risk_type = 1
    else:
        answer = 'Client s??r, score = {}'.format(pred)
        risk_type = 0
        
    
    return answer, risk_type, client.loc[:, top_features], round(pred[0],3)

# Main API function
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:#ABBAEA;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Analyse financi??re pour octroi de cr??dit</h1> 
    </div> 
    """
      
    # Display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # Following lines create boxes in which user can enter data required to make prediction 
    ClientID = st.selectbox("Veuillez choisir l'ID d'un client ayant d??pos?? un dossier d'emprunt",client_id_list)
    result =""
    interpreter_plot = ""
      
    # When 'Predict' is clicked, make the prediction, display it and display plots
    if st.button("Predict"): 
        result, risk, client_stats, clientscore = prediction(ClientID)
        # Show client data and global statistics
        st.text('Donn??es du client : \n')
        st.dataframe(client_stats)
        st.text("Statistiques globales sur l'ensemble des clients : \n")
        st.dataframe(X2_comb_test_stats)
        
        # Display score. Color depends on score
        if risk == 0:
            st.success('{}'.format(result))
        elif risk  == 1:
            st.warning('{}'.format(result))
        elif risk == 2:
            st.error('{}'.format(result))
        
        fig3, ax3 = plt.subplots(figsize = (18, 2))
        ax3.axhline(1, 0, risk_thresh, color="green", linewidth = 10)
        ax3.axhline(1, risk_thresh, high_risk_thresh, color="orange", linewidth = 10)
        ax3.axhline(1, high_risk_thresh, 1, color="red", linewidth = 10)
        ax3.text(risk_thresh, 1.6, "{} \n seuil de vigilance".format(risk_thresh), color = "orange", fontweight = "bold", fontsize = 16)
        ax3.text(high_risk_thresh, 1.6, "{} \n seuil de haute vigilance".format(high_risk_thresh), 
                 color = "red", fontweight = "bold", fontsize = 16)
        ax3.text(clientscore, 0.8, "X", color = "k", fontweight = "bold", fontsize = 30)
        ax3.text(clientscore, -0.1, 'Score du client \n %0.3f' % clientscore, color = "k", fontweight = "bold", fontsize = 16)
        plt.axis('off')
        ax3.set_ylim(-0.3, 2.5)
        st.pyplot(fig3, clear_figure=True)
        # Show global statistics for client's group

        st.text("Statistiques globales des clients s??rs : \n")
        st.dataframe(X2_comb_test_stats_low)

        st.text("Statistiques globales des clients mod??r??ment risqu??s : \n")
        st.dataframe(X2_comb_test_stats_med)

        st.text("Statistiques globales des clients tr??s risqu??s: \n")
        st.dataframe(X2_comb_test_stats_high)      
            
        # Display waterfall plot
        st.text('Explication du score : \n')
        interpreter_plot = st_shap(ClientID)
        
        # Display feature importance plot
        st.text('Importance des variables : \n')
        fig2, ax = plt.subplots(figsize = (18, 18))
        fig2 = shap.summary_plot(shap_values, X2_comb_test)
        st.pyplot(fig2, clear_figure=True)
        
        # Print final text that links to user manual
        st.markdown("Pour mieux comprendre [l'explication du score](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/5de0335e5d6442075ef22e57f40c347cd9c233ed/Note%20M??thodologique.ipynb#Cascade-d'interpr%C3%A9tation-de-pr%C3%A9diction),  [l'importance des variables](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/5de0335e5d6442075ef22e57f40c347cd9c233ed/Note%20M??thodologique.ipynb#Interpr%C3%A9teur-shap) ainsi que [la nature du mod??le](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/5de0335e5d6442075ef22e57f40c347cd9c233ed/Note%20M??thodologique.ipynb#Entra%C3%AEnement-du-mod%C3%A8le),  veuillez consulter [la note m??thodologique suivante](https://nbviewer.jupyter.org/github/Anvil-Late/Default_risk_prediction/blob/5de0335e5d6442075ef22e57f40c347cd9c233ed/Note%20M??thodologique.ipynb)",
                    unsafe_allow_html=True)
        
         
if __name__=='__main__': 
    main()

import matplotlib.transforms as transforms
import matplotlib.pyplot as plt


