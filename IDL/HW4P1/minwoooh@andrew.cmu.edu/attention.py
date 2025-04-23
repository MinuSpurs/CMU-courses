import torch
import math

class Softmax:

    def forward(self, Z):
        z_original_shape = Z.shape
        self.N = Z.shape[0] * Z.shape[1]
        self.C = Z.shape[2]
        Z = Z.reshape(self.N, self.C)

        Ones_C = torch.ones((self.C, 1))
        self.A = torch.exp(Z) / (torch.exp(Z) @ Ones_C)

        return self.A.reshape(z_original_shape)

    def backward(self, dLdA):
        dLdA_original_shape = dLdA.shape
        dLdA = dLdA.reshape(self.N, self.C)

        dLdZ = torch.zeros((self.N, self.C))
        
        for i in range(self.N):
            J = torch.zeros((self.C, self.C))
            for m in range(self.C):
                for n in range(self.C):
                    if n == m:
                        J[m, n] = self.A[i][m] * (1 - self.A[i][m])
                    else:
                        J[m, n] = -self.A[i][m] * self.A[i][n]
            dLdZ[i, :] = dLdA[i, :] @ J

        return dLdZ.reshape(dLdA_original_shape)

class Attention:
    
    def __init__(self, weights_keys, weights_queries, weights_values):
        self.W_k = weights_keys
        self.W_q = weights_queries
        self.W_v = weights_values
        self.softmax = Softmax()
        
    def forward(self, X):
        self.X = X
        
        self.Q = torch.matmul(self.X, self.W_q)  
        self.K = torch.matmul(self.X, self.W_k)  
        self.V = torch.matmul(self.X, self.W_v)  

        self.A_w = torch.matmul(self.Q, self.K.transpose(-1, -2)) / math.sqrt(self.Q.shape[-1])  

        attn_mask = torch.triu(torch.ones_like(self.A_w), diagonal=1) * -1e9  
        self.A_w = self.A_w + attn_mask

        self.A_sig = self.softmax.forward(self.A_w)  

        X_new = torch.matmul(self.A_sig, self.V)  

        return X_new
    
    def backward(self, dLdXnew):
        """
        Backpropagation through the self-attention layer.
        Calculates gradients for weights, keys, queries, values, and input.
        """
        dLdA_sig = torch.matmul(dLdXnew, self.V.transpose(-2, -1))

        dLdA_w = self.softmax.backward(dLdA_sig)  

        scale_factor = 1 / torch.sqrt(torch.tensor(self.K.size(-1), dtype=torch.float32))  
        dLdA_w_scaled = dLdA_w * scale_factor  

        self.dLdV = torch.matmul(self.A_sig.transpose(-2, -1), dLdXnew)
        self.dLdK = torch.matmul(dLdA_w_scaled.transpose(-2, -1), self.Q)  
        self.dLdQ = torch.matmul(dLdA_w_scaled, self.K)  

        self.dLdWq = torch.matmul(self.X.transpose(-2, -1), self.dLdQ)  
        self.dLdWk = torch.matmul(self.X.transpose(-2, -1), self.dLdK)  
        self.dLdWv = torch.matmul(self.X.transpose(-2, -1), self.dLdV)  

        dLdX_from_Q = torch.matmul(self.dLdQ, self.W_q.transpose(0, 1)) 
        dLdX_from_K = torch.matmul(self.dLdK, self.W_k.transpose(0, 1))  
        dLdX_from_V = torch.matmul(self.dLdV, self.W_v.transpose(0, 1))  

        dLdX = dLdX_from_Q + dLdX_from_K + dLdX_from_V 

        return dLdX
