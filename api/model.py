# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:21:38 2022

@author: Adnene
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json


path_model = '../../finalized_gbt_model.sav'
model = pickle.load(open(path_model,'rb'))