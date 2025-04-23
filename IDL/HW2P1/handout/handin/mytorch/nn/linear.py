import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features)) 
        self.b = np.zeros((out_features, 1)) 

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  
        self.N = A.shape[0] 
        Z = np.dot(self.W, A.T) + self.b  

        return Z.T

    def backward(self, dLdZ):

        dLdA = np.dot(dLdZ, self.W)  
        self.dLdW = np.dot(dLdZ.T, self.A)  
        self.dLdb = np.sum(dLdZ.T, axis=1, keepdims=True)  

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA