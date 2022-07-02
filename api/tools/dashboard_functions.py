# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 20:10:19 2022

@author: Adnene
"""

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
    
model_explainer = shap.TreeExplainer(model, data = train.drop(['ID'], axis = 1))
    


def forceplot_client(df):
    '''
    Fonction qui interpréte le score d'un client en utilisant SHAP'

    Returns
    -------
    None.

    '''
    
    shap_values = model_explainer.shap_values(df)
    
    return shap.force_plot(model_explainer.expected_value, shap_values, df)
    

    