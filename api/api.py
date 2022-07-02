# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:10:30 2022

@author: Adnene
"""

import flask
import json
import requests


from tools.preprocess import cleaning
import pandas as pd
import xgboost
import pickle


#construction API Flask: api de prediction interrogé
#par le dashboard streamlit qui affichera les résultats
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
train = pd.read_csv(path_dftrain)
test = pd.read_csv(path_dftest)

#Load model
path_model = '../../finalized_gbt_model.sav'
model = pickle.load(open(path_model,'rb'))

#@app.route('/api/credit/<ID>', methods=['POST'])

@app.route('/api/credit/<int:ID>')
def predict_credit(ID):     
    
    #retourne pour un client si crédit est accordé
    #ID = int(request.args.get('ID'))
    data = cleaning(train)
    
    data_for_prediction = data[data['ID'] == ID].iloc[:,:-1] # use 1 row of data here. Could use multiple rows if desired
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    
    prediction = model.predict(data_for_prediction_array)
    proba = model.predict_proba(data_for_prediction_array)
    
    dictionnaire = {
        'individual_data' : data_for_prediction.to_json(),
        'prediction' : int(prediction),
        'proba' : float(proba[0][0])
        }

    print('Nouvelle Prédiction : \n', dictionnaire)

    return flask.jsonify(dictionnaire)


if __name__ == "__main__":
    # url à saisir: http://localhost:5000/ ou http://127.0.0.1:5000
    app.run(debug=True)