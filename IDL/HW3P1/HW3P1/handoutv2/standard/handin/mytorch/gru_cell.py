import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.
        """
        self.x = x
        self.hidden = h_prev_t

        self.r = self.r_act.forward(np.dot(self.Wrx, self.x) + np.dot(self.Wrh, self.hidden) + self.brx + self.brh)

        self.z = self.z_act.forward(np.dot(self.Wzx, self.x) + np.dot(self.Wzh, self.hidden) + self.bzx + self.bzh)

        self.n = self.h_act.forward(np.dot(self.Wnx, self.x) + self.r * (np.dot(self.Wnh, self.hidden) + self.bnh) + self.bnx)

        h_t = (1 - self.z) * self.n + self.z * self.hidden

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) 

        return h_t


    def backward(self, delta):
        """GRU cell backward.

        This calculates the gradients wrt the parameters and returns the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
            summation of derivative wrt loss from next layer at
            the same time-step and derivative wrt loss from same layer at
            next time-step.

        Returns
        -------
        dx: (input_dim,)
            derivative of the loss wrt the input x.

        dh_prev_t: (hidden_dim,)
            derivative of the loss wrt the input hidden h.
        """
        d_h_next = delta

        d_n = d_h_next * (1 - self.z)  
        d_z = d_h_next * (-self.n + self.hidden)  
        d_h_prev = d_h_next * self.z 

        d_n_t_pre = self.h_act.backward(d_n, state=self.n)  

        self.dWnx += np.outer(d_n_t_pre, self.x)
        self.dbnx += d_n_t_pre
        d_x_n = np.dot(self.Wnx.T, d_n_t_pre)  

        d_Wnh_part = np.outer(d_n_t_pre, self.hidden)
        self.dWnh += d_Wnh_part * self.r.reshape(-1, 1)
        self.dbnh += d_n_t_pre * self.r
        d_h_prev += np.dot(self.Wnh.T, (d_n_t_pre * self.r)) 

        d_r = d_n_t_pre * (np.dot(self.Wnh, self.hidden) + self.bnh)  
        d_r_t_pre = self.r_act.backward(d_r)  

        self.dWrx += np.outer(d_r_t_pre, self.x)
        self.dbrx += d_r_t_pre
        d_x_r = np.dot(self.Wrx.T, d_r_t_pre)  

        self.dWrh += np.outer(d_r_t_pre, self.hidden)
        self.dbrh += d_r_t_pre
        d_h_prev += np.dot(self.Wrh.T, d_r_t_pre)  

        d_z_t_pre = self.z_act.backward(d_z)  

        self.dWzx += np.outer(d_z_t_pre, self.x)
        self.dbzx += d_z_t_pre
        d_x_z = np.dot(self.Wzx.T, d_z_t_pre)

        self.dWzh += np.outer(d_z_t_pre, self.hidden)
        self.dbzh += d_z_t_pre
        d_h_prev += np.dot(self.Wzh.T, d_z_t_pre)  

        d_x = d_x_n + d_x_r + d_x_z  

        assert d_x.shape == (self.d,)
        assert d_h_prev.shape == (self.h,)

        return d_x, d_h_prev
