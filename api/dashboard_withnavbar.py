# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:27:46 2022

@author: Adnene
"""
#Import
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
#from tools.strimlitbook import read_ipynb
from tools.streamlitbook.strimlitbook import read_ipynb

import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from pathlib import Path
import urllib
from urllib.request import urlopen
import json
import requests

from tools.preprocess import cleaning
from tools.dashboard_functions import *

#functions and load
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


#dashboard display with navbar:
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options = ["Home",
                   "Comprendre nos clients",
                   "Comprendre le modèle",
                   "Prédire et expliquer"],
        icons = ["house", "book", "bar-chart", "bullseye"],
        menu_icon = "cast",
        styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#FF6F61", "font-size": "25px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#6B5B95"},
    })

    
    
if selected == "Home":

    #présentation du dashboard:
        st.title("Implémentez un modéle de scoring de credit")
        st.subheader(" Bienvenue sur le Dashboard ")
        st.markdown(" Projet par Adnène Guessoum ")

if selected == "Comprendre nos clients":
    st.title(f"Analyse exploratoire des données clients:")
    
    #nb = read_ipynb('../analysis_notebooks/Exploratory_data_analysis_vers1.ipynb')
    #nb.display()
    
    #EDA = open("Exploratory_data_analysis_true.html", encoding="utf8")
    #components.html(EDA.read())
    
    def read_markdown_file(markdown_file):
        return Path(markdown_file).read_text()
    
    EDA_markdown = read_markdown_file("Exploratory_data_analysis_vers1\Exploratory_data_analysis_vers1.md")
    st.markdown(EDA_markdown, unsafe_allow_html=True)
                    
if selected == "Comprendre le modèle":
    st.title(f"Comprendre le modèle de score-crédit:")
    st.markdown(f"Informations sur le modèle choisie:")
    
    st.write('Caractéristiques globales du modèle:')
    
    with st.spinner('Chargement des caractéristiques globales du modèle...'):
         interpretation_global(10)
    st.success('Done!')
    
    
if selected == "Prédire et expliquer":
    st.title(f"Prédire et expliquer le risque de défaut d'un client:")
    st.markdown("Analyse des résultats de prédiction d'offre de crédit:")


    #choix du client:
    id_input = st.text_input('identifiant client:', )
    
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
        
        st.write('Caractéristiques locales pour le client considéré:')

        with st.spinner('Chargement des détails de la prédiction...'):
            interpretation_client(id_input)
        
        
        st.success('Done!')
        