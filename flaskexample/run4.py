#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:43:54 2020

@author: macbook
"""

from flask import Flask, request
from datetime import datetime
import urllib.request
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import urllib.request
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os


app = Flask(__name__)

@app.route('/')
def home():
    return """
         <html><body>
             <h2>Let's get some colors</h2>
             <form action="/greet">
                 What is the url of your image ? <input type='text' name='username'><br>
                 
                 <input type='submit' value='Continue'>
             </form>
         </body></html>
         """

@app.route('/greet')
def greet():
    username = request.args.get('username', 'World')
    
    for filename in os.listdir('static/'):
        if filename.startswith('image2_'):  # not to remove other images
            os.remove('static/' + filename)

    urllib.request.urlretrieve(username, "static/image2.jpg")
    dbname = 'colors'
    username = 'macbook'
    pswd = 'DarwinRulez!1'

    engine = create_engine('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname))
    con = None
    con = psycopg2.connect(database = dbname, user = username, host='localhost', password=pswd)

    X_sql = sql_query = """
    SELECT h FROM pigment_hsv2;
    """
    X = pd.read_sql_query(X_sql,con)

    y_sql  = """
    SELECT image_number FROM pigment_hsv2;
    """
    y = pd.read_sql_query(y_sql,con)

    y_2 = y["image_number"] 

    X_train,X_test,y_train,y_test = train_test_split(X,y_2,random_state=42)

    knn = KNeighborsClassifier(n_neighbors=26)

    knn.fit(X_train, y_train)

    img = io.imread('static/image2.jpg')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_PP_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]



    #avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)

    indices = np.argsort(counts)[::-1]   
    freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
    rows = np.int_(img.shape[0]*freqs)

    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    plt.imshow(dom_patch)
    plt.axis('off')
    for filename in os.listdir('static/'):
        if filename.startswith('dom2_'):  # not to remove other images
            os.remove('static/' + filename)
    plt.savefig("static/dom2.png")

    dom_pred = knn.predict(palette)

    sql_query2 = """
    SELECT color_name, image_number FROM pigment_info;
    """

    color_names = pd.read_sql_query(sql_query2, con)

    j = color_names.loc[color_names['image_number'].isin(dom_pred)]
    
    j_str = str(j)

    return """
         <html><body>
             <h2>Here are your colors</h2>
             {0}
             <h2>Your Original Image</h2>
             
            <img src="static/image2.jpg"/>
             <h2>Your dominant colors</h2>
             <img src="static/dom2.png"/>
            
         </body></html>
         """.format(j_str)

 # Launch the FlaskPy dev server
app.run(host="localhost", debug=True)