import numpy as np
import math
import scipy

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z)) 
        return self.A

    def backward(self, dLdA):
        dAdZ = self.A * (1 - self.A)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):
        self.A = np.tanh(Z) 
        return self.A

    def backward(self, dLdA):
        dAdZ = 1 - np.square(self.A) 
        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = np.where(self.A > 0, 1, 0) 
        dLdZ = dLdA * dAdZ
        return dLdZ


class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def forward(self, Z):
        self.Z = Z

        self.A = 0.5 * Z * (1 + np.array([math.erf(z / np.sqrt(2)) for z in Z.flatten()]).reshape(Z.shape))
        return self.A
    
    def backward(self, dLdA):

        Z_squared = np.power(self.Z, 2)

        dAdZ = 0.5 * (1 + np.array([math.erf(z / np.sqrt(2)) for z in self.Z.flatten()]).reshape(self.Z.shape)) + \
               (self.Z * np.exp(-0.5 * Z_squared)) / np.sqrt(2 * np.pi)

        dLdZ = dLdA * dAdZ
        return dLdZ

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """

        self.A = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True) 

        return self.A
    
    def backward(self, dLdA):

        N = self.A.shape[0] 
        C = self.A.shape[1] 

        dLdZ = np.zeros_like(dLdA) 

        for i in range(N):

            J = np.zeros((C, C))

            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m, n] = self.A[i, m] * (1 - self.A[i, m]) 
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            dLdZ[i,:] = np.dot(J, dLdA[i, :]) 
            
        return dLdZ