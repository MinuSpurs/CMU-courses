import numpy as np
from layers import *


class MLP():
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            self.layers.append(Linear(in_size, out_size))
            self.layers.append(ReLU())
        self.layers = self.layers[:-1] 

    def init_weights(self, weights):
        for i in range(len(weights)):
            self.layers[i * 2].W = weights[i].T

    def forward(self, A):
        Z = A
        for layer in self.layers:
            Z = layer(Z)
        return Z

    def backward(self, dLdZ):
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


if __name__ == '__main__':
    D = 24  
    layer_sizes = [8 * D, 8, 16, 4]
    mlp = MLP([8 * D, 8, 16, 4])