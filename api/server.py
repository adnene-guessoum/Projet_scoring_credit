# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:20:53 2022

@author: Adnene
"""

import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

path_model = '../../finalized_gbt_model.sav'
model = pickle.load(open(path_model,'rb'))

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(data['exp'])]])
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)