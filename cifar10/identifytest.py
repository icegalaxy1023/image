# -*- coding: utf-8 -*-
"""
Created on Tue May 08 16:44:25 2018

@author: icegalaxy
"""
from identify import predict
y_pred=predict(60)
labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
for i in range(60):
    print i
    print y_pred[i]
