#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:37:02 2020

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
img = cv2.imread('/Users/macbook/Documents/GitHub/flask/flaskexample/static/image2.jpg')
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
pixels = np.float32(img_HSV.reshape(-1, 3))

def shift_h_remove(data, v_thresh, s_thresh):
    """Produces shifted H values for color segmentation and removed neutral tones
    Inputs: data - list of pixel H, S, V values one entry per pixel
    Outputs: H, H120, H240
    """
    shifted_colors = []
    for i in range(0,len(data)):
        H = data[i][0]
        s = data[i][1]
        v = data[i][2]
        V_thres = 255*v_thresh
        S_thres = 255*s_thresh
        if (v > V_thres and s > S_thres):
            if H >= 120:
                H120 = H - 120
            else:
                H120 = H + 60
            if H >= 60:
                H240 = H - 60
            else:
                H240 = H + 120
            shifted_colors.append([H, H120, H240, s, v])        
        else:
           pass
        
        
    return shifted_colors


pixels_shift = shift_h_remove(pixels, .25, .25)
pixels_df = pd.DataFrame(pixels_shift, columns=['h','H120','H240','s','v'])
X_pixels = pixels_df[['h']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=42, algorithm = 'full')
kmeans.fit(X_pixels)
image2show = kmeans.cluster_centers_[kmeans.labels_]

kmeans_df = pd.DataFrame(image2show, columns=['h'])

X = kmeans_df[['h']]

predict_colors = hsv_knn.predict(X)

colors2 = np.array(np.unique(predict_colors, return_counts=True)).T

j = colors2[:,0]

print(j)