# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:53:59 2020

@author: pyd
"""

import numpy as np

class Softmax(object):
    def __init__(self,shape):
        self.softmax=np.zeros(shape)
        self.eta=np.zeros(shape)
        self.batchsize=shape[0]
        
    def predict(self,prediction):
        exp_prediction=np.zeros(prediction.shape)
        self.softmax=np.zeros(prediction.shape)
        for i in range(self.batchsize):
            prediction[i,:]-=np.max(prediction[i,:])
            exp_prediction[i]=np.exp(prediction[i])
            self.softmax[i]=exp_prediction[i]/np.sum(exp_prediction[i])
        return self.softmax
    
    def cal_loss(self, prediction, label):
        self.label = label
        loss = 0
        for i in range(label.shape[0]):
            loss += -prediction[i,:,:,label[i]] + np.log(np.sum(np.exp(label)))
        return loss
    
    def gradient(self):
        self.eta = self.softmax.copy()
        for i in range(self.batchsize):
            self.eta[i, self.label[i]] -= 1
        return self.eta