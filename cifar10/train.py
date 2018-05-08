# -*- coding: utf-8 -*-
"""
Created on Fri May 04 19:27:54 2018

@author: icegalaxy
"""
import numpy as np
from data_utils import get_CIFAR10_data
import matplotlib.pyplot as plt
from cnn import ThreeLayerConvNet
from solver import Solver

def trainmodel():
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    
    def rel_error(x, y):
      """ 返回相对误差 """
      return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
    
    # 加载 CIFAR10 数据.
    DIR = 'C:/image/cifar10'
    data = get_CIFAR10_data(DIR)
    for k, v in data.iteritems():
      print '%s: ' % k, v.shape
    
    #训练CIFAR-10模型
    model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, filter_size = 3,use_batchnorm=True)
    
    
    solver = Solver(model, data,
                     num_epochs=10, batch_size=50,
                     update_rule='adam',
                     optim_config={
                       'learning_rate': 1e-3,
                     },
                     verbose=True, print_every=20)
    solver.train()

