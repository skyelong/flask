#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:39:18 2020

@author: macbook
"""

def ModelIt(fromUser  = 'Default', births = []):
 in_month = len(births)
 print('The number born is %i' % in_month)
 result = in_month
 if fromUser != 'Default':
   return result
 else:
   return 'check your input'