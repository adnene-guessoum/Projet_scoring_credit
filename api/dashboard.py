# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:27:48 2022

@author: Adnene
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import urllib
from urllib.request import urlopen
import json
import requests

from tools.preprocess import cleaning
from tools.dashboard_functions import *


#Load Dataframe
path_dftrain = '../../application_train.csv'
path_dftest = '../../application_test.csv'

#Load model
path_model = '../../finalized_gbt_model.sav'
model = pickle.load(open(path_model,'rb'))



@st.cache #mise en cache de la fonction pour exécution unique
def load_data(PATH):
    data=pd.read_csv(PATH)
    return data

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

train = load_data(path_dftrain)
#test = load_data(path_dftest)
liste_id = train['SK_ID_CURR'].tolist()

#présentation du dashboard:
st.title("Dashboard modéle de scoring - crédit")
st.subheader("Prédictions de crédit client et comparaisons")
st.markdown("Dashboard explicatif du modéle de prédiction d'attribution de crédit:")

#menu:
st.sidebar.title("Analyse des résultats de prédiction d'offre de crédit:")
st.sidebar.markdown("information sur le modèle choisie:")


#choix du client:
id_input = st.text_input('identifiant client:', )

#mauvais identifiant: message d'erreur "veuillez verifier que l'identifiant saisi est correct"
if id_input == '':
    st.write('Veuillez saisir un identifiant.')

elif (int(id_input) not in liste_id):
    st.write('Veuillez vérifier si l\'identifiant saisie est correct.')
    st.write('Si oui, veuillez vérifier que les informations clients obligatoires ont bien été renseigné. Pour rappel les champs à renseigner sont:')
    
    
    st.write(['DAYS_BIRTH','EXT_SOURCE_3','EXT_SOURCE_2','NAME_EDUCATION_TYPE','REGION_RATING_CLIENT','NAME_INCOME_TYPE','CODE_GENDER','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','REG_CITY_NOT_WORK_CITY'])
    
#identifiant correct:
elif (int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API

    #Appel de l'API : 
    API_url = "http://127.0.0.1:5000/api/credit/" + str(int(id_input))
    
    with st.spinner('Chargement du score du client...'):
        json_url = urlopen(API_url)

        API_data = json.loads(json_url.read())
        classe_predite = API_data['prediction']
        if classe_predite == 1:
            etat = 'client à risque'
        else:
            etat = 'client peu risqué'
        proba = 1-API_data['proba'] 

        #affichage de la prédiction
        prediction = API_data['proba']
        classe_reelle = train[train['SK_ID_CURR']==int(id_input)]['TARGET'].values[0]
        classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
        chaine = 'Prédiction : **' + etat +  '** avec **' + str(round(proba*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')'
        
    st.markdown(chaine)

    st.subheader("Caractéristiques explicatives de la prédiction:")
    
    st.write('Caractéristiques globales du modèle:')
    
    with st.spinner('Chargement des caractéristiques globales du modèle...'):
         interpretation_global(10)
    st.success('Done!')

    st.write('Caractéristiques locales pour le client considéré:')

    with st.spinner('Chargement des détails de la prédiction...'):
        interpretation_client(id_input)
        
        
    st.success('Done!')

    