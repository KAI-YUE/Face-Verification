# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:50:10 2020

@author: pyd
"""

import numpy as np
from functools import reduce
import math

class Conv2D(object):
    def __init__(self,shape,weights,bias,output_channels,ksize=3,stride=1,method='VALID',autoinit=True):
        self.input_shape=shape
        self.output_channels=output_channels
        self.input_channels=shape[-1]
        self.batchsize=shape[0]
        self.stride=stride
        self.ksize=ksize
        self.method=method
        if autoinit:
            weights_scale=math.sqrt(reduce(lambda x,y:x*y,shape)/self.output_channels)
            self.weights=np.random.standard_normal((ksize,ksize,self.input_channels,self.output_channels))/weights_scale
            self.bias=np.random.standard_normal(self.output_channels)/weights_scale
        else:
            if (weights.shape[0]!=ksize or weights.shape[1]!=ksize or weights.shape[2]!=self.input_channels or weights.shape[3]!=self.output_channels) or (bias.shape[0]!=self.output_channels):
                print("conv params shape not suitable!")
                return
            self.weights=weights
            self.bias=bias
        if method == 'VALID':
            self.eta = np.zeros((shape[0], (shape[1] - ksize + 1) // self.stride, (shape[2] - ksize + 1) // self.stride,
             self.output_channels))

        if method == 'SAME':
            self.eta = np.zeros((shape[0], shape[1]//self.stride, shape[2]//self.stride,self.output_channels))

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

        if (shape[1] - ksize) % stride != 0:
            print('input tensor width can\'t fit stride')
        if (shape[2] - ksize) % stride != 0:
            print('input tensor height can\'t fit stride')
            
    def forward(self,x):
        
        col_weights=self.weights.reshape([-1,self.output_channels])
       
        if self.method=='SAME':
            x=np.pad(x,((0,0),(self.ksize//2,self.ksize//2),(self.ksize//2,self.ksize//2),(0,0)),'constant',constant_values=0)
        
        self.col_image=[]
        conv_out=np.zeros(self.output_shape)
       
        for i in range(self.batchsize):
            img_i=x[i][np.newaxis,:]
            self.col_image_i=im2col(img_i,self.ksize,self.stride)
            
            conv_out[i]=np.reshape(np.dot(self.col_image_i,col_weights)+self.bias,self.output_shape[1:])
            
            self.col_image.append(self.col_image_i)
        #mtt=mt.stop()
        self.col_image=np.array(self.col_image)
        #ttt=tt.stop()
       # print(ptt,mtt,ttt)
        return conv_out
    
    def gradient(self, eta):
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array([im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        
def im2col(image,ksize,stride):
    image_col=[]
    for i in range(0,image.shape[1]-ksize+1,stride):
        for j in range(0,image.shape[2]-ksize+1,stride):
            col=image[:,i:i+ksize,j:j+ksize,:].reshape([-1])
            image_col.append(col)
    image_col=np.array(image_col)
    return image_col