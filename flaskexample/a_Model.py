#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:39:18 2020

@author: macbook
"""
import urllib.request

def ModelIt(fromUser  = 'Default', url = ""):
 urllib.request.urlretrieve(url, "image.jpg")
 result = "hi!"
 if fromUser != 'Default':
   return result
 else:
   return 'check your input'