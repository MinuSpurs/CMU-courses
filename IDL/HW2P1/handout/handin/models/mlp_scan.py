# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()] 

    def init_weights(self, weights):
        w1, w2, w3 = weights
        weight_list = [w1, w2, w3]
        conv_layers = [self.conv1, self.conv2, self.conv3]

        for conv_layer, weight in zip(conv_layers, weight_list):
            reshaped_weight = weight.T.reshape(conv_layer.out_channels, conv_layer.kernel_size, conv_layer.in_channels)
            conv_layer.conv1d_stride1.W = reshaped_weight.transpose(0, 2, 1)

            conv_layer.conv1d_stride1.b = np.zeros(conv_layer.out_channels)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=2, stride=2)
        self.conv2 = Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=2)
        self.conv3 = Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()] 

    def __call__(self, A):
        return self.forward(A)

    def init_weights(self, weights):

        w1,w2,w3 = weights
        self.conv1.conv1d_stride1.W = w1[:,:2].T.reshape((2,8,24))[:,:2,:].transpose(0,2,1)
        self.conv2.conv1d_stride1.W = w2[:,:8].T.reshape((8,4,2))[:,:2,:].transpose(0,2,1)
        self.conv3.conv1d_stride1.W = w3[:,:].T.reshape((4,2,8)).transpose(0,2,1)
        
    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel, out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA