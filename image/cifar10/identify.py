# -*- coding: utf-8 -*-
"""
Created on Sat May 05 21:00:31 2018

@author: icegalaxy
"""
import cPickle as pickle
import numpy as np
from cnn import ThreeLayerConvNet
# =============================================================================
# 参数 n 图片识别的数目
#返回 图像标签
# =============================================================================
def predict(n):
    #读入测试数据
    DIR = 'C:/image/cifar10/datasets/cifar-10-batches/data_batch_6'
    with open(DIR, 'r') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
    mean_image = np.mean(X, axis=0)
    X -= mean_image
    X = X.transpose(0, 3, 1, 2).copy()
    
    #读入训练模型 
    model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, filter_size = 3,use_batchnorm=True)
    for key in model.params:
        model.params[key]=np.load('C:/image/cifar10/model/'+key+'.npy')
    
    #图像识别 
    labels=[]
    scores = model.loss(X[0:n])
    y_pred = np.argmax(scores, axis=1)
    label=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    for i in range(60):
        labels.append(label[y_pred[i]])
    return labels
