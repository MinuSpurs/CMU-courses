import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]  
        self.C = A.shape[1]  
        se = (A - Y) ** 2 
        sse = se.sum()  
        mse = sse / (self.N * self.C)  

        return mse

    def backward(self):

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape[0] 
        C = A.shape[1]  

        Ones_C = np.ones((1, C), dtype='f')  
        Ones_N = np.ones((N, 1), dtype='f')  

        self.softmax = np.exp(A) / np.dot(np.sum(np.exp(A), axis=1, keepdims=True), Ones_C)  
        crossentropy = -np.sum(Y * np.log(self.softmax + 1e-12), axis=1)  
        sum_crossentropy = np.dot(Ones_N.T, crossentropy)[0]  
        L = sum_crossentropy / N

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y) / self.A.shape[0]  

        return dLdA