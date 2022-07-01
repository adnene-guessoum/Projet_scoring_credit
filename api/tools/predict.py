# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 20:10:19 2022

@author: Adnene
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Load Dataframe
path_dftrain = '../../application_train.csv'
path_dftest = '../../application_test.csv'
dataframe_train = pd.read_csv(path_dftrain)
dataframe_test = pd.read_csv(path_dftest)

#Load model
path_model = '../../finalized_gbt_model.sav'
model = pickle.load(open(path_model,'rb'))


def predict_credit(dataframe, ID):     
    
    #retourne pour un client si crédit est accordé
    data = preprocess(dataframe)
    
    prediction = model.predict([[np.array(data['exp'])]])





def predict_flask(ID, dataframe):
    '''Fonction de prédiction utilisée par l\'API flask :
    a partir de l'identifiant et du jeu de données
    renvoie la prédiction à partir du modèle'''

    ID = int(ID)
    X = dataframe[dataframe['SK_ID_CURR'] == ID]

    X = X.drop(['Unnamed: 0', 'SK_ID_CURR', 'LABELS'], axis=1)
    proba = predict_function_xgb_stacking(X)
    print(proba)
    print(proba[0])
    print(proba[0][0])
    if proba[0][0] > 0.5:
        return 0, proba
    else:
        return 1, proba
    
if __name__ == "__main__":