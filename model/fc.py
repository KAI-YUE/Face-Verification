# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:49:09 2020

@author: pyd
"""

import numpy as np
from functools import reduce
import math

class FullyConnect(object):
    def __init__(self,shape ,weights,bias,output_num,autoinit=True):
        self.input_shape=shape
        self.batchsize=shape[0]
        self.output_shape=[self.batchsize,output_num]
        input_len=reduce(lambda x,y: x*y, shape[1:])
        if autoinit:
            self.weights=np.random.standard_normal((input_len,output_num))/100
            self.bias=np.random.standard_normal(output_num)/100
        else:
            if (input_len != weights.shape[0]) or (output_num !=weights.shape[1]) or (output_num!=bias.shape[0]):
                print("fc param shape not suitable!")
                return
            self.weights=weights
            self.bias=bias
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        
    def forward(self,x):
        self.x=x.reshape([self.batchsize,-1])
        output=np.dot(self.x,self.weights)+self.bias
        return output
        
    def gradient(self, eta):
        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = eta[i][:, np.newaxis].T
            self.w_gradient += np.dot(col_x, eta_i)
            self.b_gradient += eta_i.reshape(self.bias.shape)

        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)

        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias
        # zero gradient
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)