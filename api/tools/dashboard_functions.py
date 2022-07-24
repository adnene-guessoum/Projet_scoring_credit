# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 20:10:19 2022

@author: Adnene
"""

import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#metrics
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import fbeta_score

from tools.preprocess import cleaning
import shap
shap.initjs()


#Load Dataframe
path_dftrain = '../../application_train.csv'
path_dftest = '../../application_test.csv'
dataframe_train = pd.read_csv(path_dftrain)
dataframe_test = pd.read_csv(path_dftest)

#Load model
path_model = '../../finalized_gbt_model.sav'
model = pickle.load(open(path_model,'rb'))

train = cleaning(dataframe_train)
#test = cleaning(dataframe_test)
model_explainer = shap.TreeExplainer(model, data = train)
exp_vals = model_explainer.expected_value
shap_vals = model_explainer.shap_values(train, check_additivity=False)

true_y = train["TARGET"]
predictions = model.predict(train.drop(["TARGET"], axis = 1).iloc[:,:-1])
probas_predictions = model.predict_proba(train.drop(["TARGET"], axis = 1).iloc[:,:-1])[:, 1]

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def interpretation_global(sample_nb):
                        
    #fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    
    auc_train_model = roc_auc_score(true_y, probas_predictions)
    st.write('AUC pour les données entraînement : '+str(auc_train_model))


    fpr_train_gbt, tpr_train_gbt, _ = roc_curve(true_y, probas_predictions)

    fig = plt.figure()
    plt.plot(fpr_train_gbt, tpr_train_gbt, color='blue', label='AUC_train = %0.2f' % auc_train_model)
    plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), color='red')
    plt.legend(loc = 'lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive Rate')
    plt.title('Credit Default- Gradient Boosting')
    #plt.savefig('roc_curve.png', dpi=300)
    plt.show()
    st.pyplot(fig)
    
    
    st.write("Explication globale du modèle avec SHAP:")
    #plot 1
    st.write("classement et résumé global de l'importance des features pour le modèle d'après leurs influences respectives dans l'octroie de crédit des clients:")
    fig1 = plt.figure()
    sum_plot = shap.summary_plot(shap_vals, train)
    st.pyplot(fig1)
    
    #plot 2
    st.write("description du processus de décision pour un sous-ensemble (10 clients):")
    fig2 = plt.figure()
    dec_plot_sample = shap.decision_plot(exp_vals.tolist(),
                               model_explainer.shap_values(train.sample(n = sample_nb)),
                               features=train,
                               highlight = [1]
                               )
    st.pyplot(fig2)
    
    
    
    st.write("influences respectives des features pour la décision d'octroyer le crédit:")
    fig3 = plt.figure()
    B_plot = shap.bar_plot(shap_vals[0], train)
    
    st.pyplot(fig3)
    
    
    
    
def interpretation_client(id_input):
    '''
    Fonction qui interpréte le score d'un client en utilisant SHAP'

    Returns
    -------
    None.

    '''
    individual_data = train[train['ID']==int(id_input)]
   
    
    #model_explainer = shap.TreeExplainer(model, data = train)
    shap_values = model_explainer.shap_values(individual_data)
    
    
    #fig3 = plt.figure()
    # Insert second SHAP plot here
    #waterfall_pl = shap.plots.waterfall(exp_vals,shap_values[0])
    #st.pyplot(fig3)
    
    fig2 = plt.figure()
    # Insert SHAP plot here
    #B_plot = shap.bar_plot(shap_values, individual_data)
    dec_plot = shap.decision_plot(exp_vals.tolist(),
                               shap_values,
                               features=train
                               )
    st.pyplot(fig2)
    
    fig = plt.figure()
    # Insert first SHAP plot here
    F_plot = shap.force_plot(model_explainer.expected_value, shap_values, individual_data)
    st_shap(F_plot, 400)
    #st.pyplot(fig)
    
def model_score(model):
    prediction = model.predict(train)
    proba = model.predict_proba(train)
    importance = model.feature_importances_
    
    for i,v in enumerate(importance):
        st.write('Feature: %0d, Score: %.5f' % (i,v))
    
    fig = plt.figure()
    plt.bar([x for x in range(len(importance))],importance)
    st.pyplot(fig)
    
    

    