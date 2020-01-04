# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:51:22 2020

@author: pyd
"""

import numpy as np

class MaxPooling(object):
    def __init__(self,shape,ksize=2,stride=2):
        self.input_shape=shape
        self.ksize=ksize
        self.stride=stride
        self.output_channels=shape[-1]
        self.index=np.zeros(shape)
        self.output_shape = [shape[0], shape[1] // self.stride, shape[2] // self.stride, self.output_channels]
    
    def forward(self,x):
        out = np.zeros([x.shape[0], x.shape[1] // self.stride, x.shape[2] // self.stride, self.output_channels])
        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0,x.shape[1]-self.stride+1,self.stride):
                    for j in range(0,x.shape[2]-self.stride+1,self.stride):
                        out[b, i // self.stride, j // self.stride, c] = np.max(x[b, i:(i + self.ksize), j:(j + self.ksize), c])
                        index=np.argmax(x[b,i:i+self.ksize,j:j+self.ksize,c])
                        self.index[b,i+index//self.stride,j+index%self.stride,c]=1
        return out
    
    # def forward(self,x):
    #     out = np.zeros([x.shape[0], x.shape[1] // self.stride, x.shape[2] // self.stride, self.output_channels])
    #     for i in range(0,x.shape[1]-self.stride+1,self.stride):
    #         for j in range(0,x.shape[2]-self.stride+1,self.stride):
    #             out[:, i // self.stride, j // self.stride, :] = np.max(x[:, i:(i + self.ksize), j:(j + self.ksize), :],axis=(1,2))
    #             #index=np.argmax(x[:,i:i+self.ksize,j:j+self.ksize,:],axis=(1,2))
    #             #self.index[:,i+index//self.stride,j+index%self.stride,:]=1
    #     return out

    def gradient(self, eta):
        a=np.repeat(np.repeat(eta, self.stride, axis=1), self.stride, axis=2)
        sa=a.shape
        si=self.index.shape
        return np.pad(a,((0,0),(0,si[1]-sa[1]),(0,si[2]-sa[2]),(0,0)), 'constant') * self.index
