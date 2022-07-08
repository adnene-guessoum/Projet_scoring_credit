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
model_explainer = shap.TreeExplainer(model, data = train)
exp_vals = model_explainer.expected_value

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def interpretation_global(sample_nb):
    #train = cleaning(df)
    #model_explainer = shap.TreeExplainer(model, data = train)
    
    shap_vals = model_explainer.shap_values(train)
    
    #fig, ax = plt.subplots(nrows=2, ncols=1)
    #sum_plot = shap.summary_plot(shap_vals, train)
    #dec_plot_sample = shap.decision_plot(exp_vals.tolist(),
    #                           model_explainer.shap_values(train.sample(n = sample_nb)),
    #                           features=train,
    #                           highlight = [1]
    #                           )
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    #plt.subplot(111)
    # Insert first SHAP plot here
    fig1 = plt.figure()
    sum_plot = shap.summary_plot(shap_vals, train)
    st.pyplot(fig1)
    
    fig2 = plt.figure()
    # Insert second SHAP plot here
    dec_plot_sample = shap.decision_plot(exp_vals.tolist(),
                               model_explainer.shap_values(train.sample(n = sample_nb)),
                               features=train,
                               highlight = [1]
                               )
    st.pyplot(fig2)
    
    fig3 = plt.figure()
    # Insert second SHAP plot here
    B_plot = shap.bar_plot(shap_vals[0], train)
    
    st.pyplot(fig3)
    
    #return st.pyplot(sum_plot), st.pyplot(dec_plot_sample, 400)

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
    
    

    