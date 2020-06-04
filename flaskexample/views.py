#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:07:24 2020

@author: macbook
"""
from flask import render_template, request, Flask 
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flaskexample.a_Model import ModelIt
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename




user = 'macbook'
host = 'localhost'
dbname = 'colors'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_files():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
                           title = 'Home', user = { 'nickname': 'Miguel'},
                           )
@app.route('/db')
def color_page():
    sql_query = """
    SELECT * FROM pigment_info"""
    query_results = pd.read_sql_query(sql_query,con)
    colors = ""
    for i in range (0,10):
        colors += query_results.iloc[i]["warm_cool"]
        colors += "<br>"
    return colors

@app.route('/db_fancy')
def cesareans_page_fancy():
   sql_query = """
              SELECT index, color_name, warm_cool FROM pigment_info;
               """
   query_results=pd.read_sql_query(sql_query,con)
   births = []
   for i in range(0,query_results.shape[0]):
       births.append(dict(index=query_results.iloc[i]['index'], 
                     attendant=query_results.iloc[i]['color_name'], 
                     birth_month=query_results.iloc[i]['warm_cool']))
   return render_template('colors.html',births=births)

@app.route('/input')
def cesareans_input():
   return render_template("input.html")

@app.route('/output')
def cesareans_output():
 #pull 'birth_month' from input field and store it
 patient = request.args.get('color_name')
   #just select the Cesareans  from the birth dtabase for the month that the user inputs
 query = "SELECT index, color_name, warm_cool FROM pigment_info WHERE color_name='%s'" % patient
 print(query)
 query_results=pd.read_sql_query(query,con)
 print(query_results)
 births = []
 for i in range(0,query_results.shape[0]):
     births.append(dict(index=query_results.iloc[i]['index'], color_name=query_results.iloc[i]['color_name'], warm_cool=query_results.iloc[i]['warm_cool']))
     the_result = ModelIt(patient,births)
 return render_template("output.html", births = births, the_result = the_result)

