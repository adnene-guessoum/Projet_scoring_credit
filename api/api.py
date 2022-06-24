# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:10:30 2022

@author: Adnene
"""

import flask
#from flask import Flask, render_template, jsonify
import json
import requests

app = flask.Flask(__name__)
app.config["DEBUG"] = True


# endpoint: prédiction octroie de credit (oui, non pour un client)

@app.route('/', methods=['GET'])

def home():
 return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for scoring ML project.</p>"


#def credit(id_client):
    
    
    
 #   dictionnaire = {
 #       'prediction' : int(prediction),
 #       'proba' : float(proba[0][0])
 #       }

 #   print('Nouvelle Prédiction : \n', dict_final)

  #  return jsonify(dictionnaire)


app.run()

#if __name__ == "__main__":