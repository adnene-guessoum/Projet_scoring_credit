# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:04:05 2022

@author: Adnene
"""

#Pour tester l'application avant de construire le dashboard
import os
import tempfile
import pytest
from api import app

    
def test_api():
    response = app.test_client().get('/api/credit/<ID>')

    assert response.status_code == 200
    assert response.data == b'!'
    
    
