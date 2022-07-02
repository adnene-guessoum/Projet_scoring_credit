# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:27:48 2022

@author: Adnene
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#Load Dataframe
path_dftrain = '../../../application_train.csv'
path_dftest = '../../../application_test.csv'


@st.cache #mise en cache de la fonction pour exécution unique
def load_data(PATH):
    data=pd.read_csv(PATH)
    return data

train = load_data(path_dftrain)
#test = load_data(path_dftest)
liste_id = train['SK_ID_CURR'].tolist()


st.title("Dashboard modéle de scoring - crédit")
st.subheader("Prédictions de crédit client et comparaisons")
st.sidebar.title("Analyse des résultats de prédiction d'offre de crédit:")
st.markdown("Dashboard explicatif du modéle de prédiction d'attribution de crédit:")
st.sidebar.markdown("Dashboard explicatif du modéle de prédiction d'attribution de crédit::")
