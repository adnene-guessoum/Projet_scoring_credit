# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:10:30 2022

@author: Adnene
"""

import flask
#from flask import Flask, render_template, jsonify
import json
import requests

#from tools.predict import *
from tools.preprocess import cleaning
import pandas as pd
import xgboost
import pickle


app = flask.Flask(__name__)
app.config["DEBUG"] = True



#page d'accueil du site
@app.route('/', methods=['GET'])
def home():
    return flask.render_template("home.html")


#URL d'acces au dashboard:  http://localhost:5000/dashboard/  
@app.route('/dashboard/') 
def dashboard():
    return flask.render_template("dashboard.html")




# endpoint: prédiction octroie de credit (oui, non pour un client)

#Load Dataframe
path_dftrain = '../../application_train.csv'
path_dftest = '../../application_test.csv'
dataframe_train = pd.read_csv(path_dftrain)
dataframe_test = pd.read_csv(path_dftest)

#Load model
path_model = '../../finalized_gbt_model.sav'
model = pickle.load(open(path_model,'rb'))


@app.route('/api/credit/')
def predict_credit(dataframe, ID):     
    
    #retourne pour un client si crédit est accordé
    data = cleaning(dataframe)
    
    data_for_prediction = data[data['ID'] == ID].iloc[:,:-1] # use 1 row of data here. Could use multiple rows if desired
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

    
    prediction = model.predict(data_for_prediction_array)
    proba = model.predict_proba(data_for_prediction_array)
    
    dictionnaire = {
        'prediction' : int(prediction),
        'proba' : float(proba[0][0])
        }

    print('Nouvelle Prédiction : \n', dict_final)

    return flask.jsonify(dictionnaire)

def get_pred()

if __name__ == "__main__":
    # url à saisir: http://localhost:5000/ ou http://127.0.0.1:5000
    app.run(debug=True)