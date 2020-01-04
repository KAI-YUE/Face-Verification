# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:13:06 2020

@author: pyd
"""
# Python Libraries
import numpy as np
import scipy.io
import math
import pickle

# My Libraries
from model.fc import FullyConnect
from model.relu import Relu
from model.softmax import Softmax
from model.conv import Conv2D
from model.pool import MaxPooling
from src.loadConfig import DictClass

class Net(object):
    def __init__(self, in_channels=3, outFeatures=2622, **kwargs):
        """
        Constructor
        """
        if "config" in kwargs:
            config = kwargs["config"]
        else:
            config = DictClass(default_config)

        self.conv1_1 = Conv2D([config.batch_size,250,250,in_channels],0,0,64,3,1,'SAME')
        self.relu1_1 = Relu(self.conv1_1.output_shape)
        self.conv1_2 = Conv2D(self.relu1_1.output_shape,0,0,64,3,1,'SAME')
        self.relu1_2 = Relu(self.conv1_2.output_shape)
        self.pool1 = MaxPooling(self.relu1_2.output_shape)

        self.conv2_1 = Conv2D(self.pool1.output_shape,0,0,128,3,1,'SAME')
        self.relu2_1 = Relu(self.conv2_1.output_shape)
        self.conv2_2 = Conv2D(self.relu2_1.output_shape,0,0,128,3,1,'SAME')
        self.relu2_2 = Relu(self.conv2_2.output_shape)
        self.pool2 = MaxPooling(self.relu2_2.output_shape)

        self.conv3_1 = Conv2D(self.pool2.output_shape,0,0,256,3,1,'SAME')
        self.relu3_1=Relu(self.conv3_1.output_shape)
        self.conv3_2=Conv2D(self.relu3_1.output_shape,0,0,256,3,1,'SAME')
        self.relu3_2=Relu(self.conv3_2.output_shape)
        self.conv3_3=Conv2D(self.relu3_2.output_shape,0,0,256,3,1,'SAME')
        self.relu3_3=Relu(self.conv3_3.output_shape)
        self.pool3=MaxPooling(self.relu3_3.output_shape)

        self.conv4_1=Conv2D(self.pool3.output_shape,0,0,512,3,1,'SAME')
        self.relu4_1=Relu(self.conv4_1.output_shape)
        self.conv4_2=Conv2D(self.relu4_1.output_shape,0,0,512,3,1,'SAME')
        self.relu4_2=Relu(self.conv4_2.output_shape)
        self.conv4_3=Conv2D(self.relu4_2.output_shape,0,0,512,3,1,'SAME')
        self.relu4_3=Relu(self.conv4_3.output_shape)
        self.pool4=MaxPooling(self.relu4_3.output_shape)

        self.conv5_1=Conv2D(self.pool4.output_shape,0,0,512,3,1,'SAME')
        self.relu5_1=Relu(self.conv5_1.output_shape)
        self.conv5_2=Conv2D(self.relu5_1.output_shape,0,0,512,3,1,'SAME')
        self.relu5_2=Relu(self.conv5_2.output_shape)
        self.conv5_3=Conv2D(self.relu5_2.output_shape,0,0,512,3,1,'SAME')
        self.relu5_3=Relu(self.conv5_3.output_shape)
        self.pool5=MaxPooling(self.relu5_3.output_shape)

        self.fc6=Conv2D(self.pool5.output_shape,0,0,4096,7,1,'VALID')
        self.relu6=Relu(self.fc6.output_shape)
        self.fc7=Conv2D(self.relu6.output_shape,0,0,4096,1,1,'SAME')
        self.relu7=Relu(self.fc7.output_shape)
        self.fc8=Conv2D(self.relu7.output_shape,0,0,2622,1,1,'SAME')
        self.sf=Softmax(self.fc8.output_shape)

    ####################### Public Methods ########################
    def set_param(self, **kwargs):
        """
        Set parameters of the model
        """
        for arg, value in kwargs.items():
            command = "self.{}".format(arg)
            exec(command + "={}".format(value)) 

    def netforward(self, x):
        y=self.pool1.forward(self.relu1_2.forward(self.conv1_2.forward(self.relu1_1.forward(self.conv1_1.forward(x)))))
        y=self.pool2.forward(self.relu2_2.forward(self.conv2_2.forward(self.relu2_1.forward(self.conv2_1.forward(y)))))
        y=self.pool3.forward(self.relu3_3.forward(self.conv3_3.forward(self.relu3_2.forward(self.conv3_2.forward(self.relu3_1.forward(self.conv3_1.forward(y)))))))
        y=self.pool4.forward(self.relu4_3.forward(self.conv4_3.forward(self.relu4_2.forward(self.conv4_2.forward(self.relu4_1.forward(self.conv4_1.forward(y)))))))
        y=self.pool5.forward(self.relu5_3.forward(self.conv5_3.forward(self.relu5_2.forward(self.conv5_2.forward(self.relu5_1.forward(self.conv5_1.forward(y)))))))
        y=self.fc8.forward(self.relu7.forward((self.fc7.forward(self.relu6.forward(self.fc6.forward(y))))))
        return y 
    
    def train(self, X, Y, **kwargs):
        if "config" in kwargs:
            config = kwargs["config"]
        else:
            config = DictClass(default_config)
        
        batch_size = config.batch_size
        lr = config.lr
        for epoch in range(config.max_epoch):
            for i in range(X.shape[0]//batch_size):
                img = X[i*batch_size:(i+1)*batch_size].reshape([batch_size,X.shape[1],X.shape[2],3])
                label = Y[i*batch_size:(i+1)*batch_size]
                
                # Forward propagation
                out = self.netforward(img)
                loss = self.sf.cal_loss(out, label)
                
                # Back propagation
                self.sf.gradient()
                eta = self.fc8.gradient(self.sf.eta)
                self.fc8.backward(lr)

                eta = self.fc7.gradient(self.relu7.gradient(eta))
                self.fc7.backward(lr)

                eta = self.fc6.gradient(self.relu6.gradient(eta))
                self.fc6.backward(lr)

                eta = self.conv5_3.gradient(self.relu5_3.gradient(self.pool5.gradient(eta)))
                self.conv5_3.backward(lr)

                eta = self.conv5_2.gradient(self.relu5_2.gradient(eta))
                self.conv5_2.backward(lr)

                eta = self.conv5_1.gradient(self.relu5_1.gradient(eta))
                self.conv5_1.backward(lr)

                eta = self.conv4_3.gradient(self.relu4_3.gradient(self.pool4.gradient(eta)))
                self.conv4_3.backward(lr)

                eta = self.conv4_2.gradient(self.relu4_2.gradient(eta))
                self.conv4_2.backward(lr)

                eta = self.conv4_1.gradient(self.relu4_1.gradient(eta))
                self.conv4_1.backward(lr)

                eta = self.conv3_3.gradient(self.relu4_3.gradient(self.pool4.gradient(eta)))
                self.conv4_3.backward(lr)

                eta = self.conv4_2.gradient(self.relu4_2.gradient(eta))
                self.conv4_2.backward(lr)

                eta = self.conv3_3.gradient(self.relu3_3.gradient(self.pool3(eta)))
                self.conv3_3.backward(lr)

                eta = self.conv3_2.gradient(self.relu3_2.gradient(eta))
                self.conv3_2.backward(lr)

                eta = self.conv3_1.gradient(self.relu3_1.gradient(eta))
                self.conv3_1.backward(lr)

                eta = self.conv2_2.gradient(self.relu2_2.gradient(self.pool2(eta)))
                self.conv2_2.backward()

                eta = self.conv2_1.gradient(self.relu2_1.gradient(eta))
                self.conv2_1.backward(lr)

                eta = self.conv1_2.gradient(self.relu1_2.gradient(self.pool1(eta)))
                self.conv1_2.backward()

                eta = self.conv1_1.gradient(self.relu1_1.gradient(eta))
                self.conv1_1.backward(lr)

    def load(self, src_dir):
        with open(src_dir, 'rb') as fp:
            a_set = pickle.load(fp)

            self.conv1_1.weights = a_set["conv1_1w"]
            self.conv1_1.bias = a_set["conv1_1b"]
            self.conv1_2.weights = a_set["conv1_2w"]
            self.conv1_2.bias = a_set["conv1_2b"]

            self.conv2_1.weights = a_set["conv2_1w"]
            self.conv2_1.bias = a_set["conv2_1b"]
            self.conv2_2.weights = a_set["conv2_2w"]
            self.conv2_2.bias = a_set["conv2_2b"]

            self.conv3_1.weights = a_set["conv3_1w"]
            self.conv3_1.bias = a_set["conv3_1b"]
            self.conv3_2.weights = a_set["conv3_2w"]
            self.conv3_2.bias = a_set["conv3_2b"]
            self.conv3_3.weights = a_set["conv3_3w"]
            self.conv3_3.bias = a_set["conv3_3b"]

            self.conv4_1.weights = a_set["conv4_1w"]
            self.conv4_1.bias = a_set["conv4_1b"]
            self.conv4_2.weights = a_set["conv4_2w"]
            self.conv4_2.bias = a_set["conv4_2b"]
            self.conv4_3.weights = a_set["conv4_3w"]
            self.conv4_3.bias = a_set["conv4_3b"]
            
            self.conv5_1.weights = a_set["conv5_1w"]
            self.conv5_1.bias = a_set["conv5_1b"]
            self.conv5_2.weights = a_set["conv5_2w"]
            self.conv5_2.bias = a_set["conv5_2b"]
            self.conv5_3.weights = a_set["conv5_3w"]
            self.conv5_3.bias = a_set["conv5_3b"]

            self.fc6.weights = a_set["fc6_w"]
            self.fc6.bias = a_set["fc6_b"]
            self.fc7.weights = a_set["fc7_w"]
            self.fc7.bias = a_set["fc7_b"]
            self.fc8.weights = a_set["fc8_w"]
            self.fc8.bias = a_set["fc8_b"]

    def save(self, dst_dir):
        params_dict = \
            {
                "conv1_1w": self.conv1_1.weights,
                "conv1_1b": self.conv1_1.bias,
                "conv1_2w": self.conv1_2.weights,
                "conv1_2b": self.conv1_2.bias,

                "conv2_1w": self.conv2_1.weights,
                "conv2_1b": self.conv2_1.bias,
                "conv2_2w": self.conv2_2.weights,
                "conv2_2b": self.conv2_2.bias,

                "conv3_1w": self.conv3_1.weights,
                "conv3_1b": self.conv3_1.bias,
                "conv3_2w": self.conv3_2.weights,
                "conv3_2b": self.conv3_2.bias,
                "conv3_3w": self.conv3_3.weights,
                "conv3_3b": self.conv3_3.bias,

                "conv4_1w": self.conv4_1.weights,
                "conv4_1b": self.conv4_1.bias,
                "conv4_2w": self.conv4_2.weights,
                "conv4_2b": self.conv4_2.bias,
                "conv4_3w": self.conv4_3.weights,
                "conv4_3b": self.conv4_3.bias,

                "conv5_1w": self.conv5_1.weights,
                "conv5_1b": self.conv5_1.bias,
                "conv5_2w": self.conv5_2.weights,
                "conv5_2b": self.conv5_2.bias,
                "conv5_3w": self.conv5_3.weights,
                "conv5_3b": self.conv5_3.bias,

                "fc6_w": self.fc6.weights,
                "fc6_b": self.fc6.bias,
                "fc7_w": self.fc7.weights,
                "fc7_b": self.fc7.bias,
                "fc8_w": self.fc8.weights,
                "fc8_b": self.fc8.bias,
            }
        
        with open (params_dict, 'wb') as fp:
            pickle.dump(params_dict, fp)
        
    
default_config = \
{
    "batch_size": 2,
    "lr":         1e-4,
    "max_epoch":  10
}

if __name__ == '__main__':
    src_dir = r'D:\Projects\Face-Verification-release\checkpoints\model.dat'
    NN = Net()
    NN.load(src_dir)
    
    NN.train(img1, y)

