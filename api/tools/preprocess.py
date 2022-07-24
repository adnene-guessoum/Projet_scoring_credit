# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:53:28 2022

@author: Adnene
"""

import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler

import os
import warnings
warnings.filterwarnings('ignore')

def cleaning(data):
    ID = data['SK_ID_CURR']

    df = data[['DAYS_BIRTH','EXT_SOURCE_3','EXT_SOURCE_2','NAME_EDUCATION_TYPE','REGION_RATING_CLIENT','NAME_INCOME_TYPE','CODE_GENDER','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','REG_CITY_NOT_WORK_CITY', 'TARGET']]

    df_d = pd.get_dummies(df)
    df_d['ID'] = ID
    df_d = df_d.dropna()
    x = df_d
    
    print(x.shape, df_d.shape)
    
    return x

#if __name__ == "__main__":
 