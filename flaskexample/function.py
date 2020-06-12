#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:15:31 2020

@author: macbook
"""

from joblib import dump, load
import numpy as np
import cv2
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

#import the clustered ds swatches
hsv_knn = load('static/ds_h_knn_3.joblib')

 username = request.args.get('username', 'World')
    
    for filename in os.listdir('static/'):
        if filename.startswith('image2_'):  # not to remove other images
            os.remove('static/' + filename)

    urllib.request.urlretrieve(username, "static/image2.jpg")