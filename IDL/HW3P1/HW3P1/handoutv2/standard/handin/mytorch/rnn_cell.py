import numpy as np
from mytorch.nn.activation import *


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).

        Input
        -----
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """
        h_t = self.activation.forward(
            np.dot(x, self.W_ih.T) + self.b_ih + np.dot(h_prev_t, self.W_hh.T) + self.b_hh
        )
        
        return h_t



    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        Input
        -----
        delta: (batch_size, hidden_size)
            Gradient w.r.t. the current hidden layer.

        h_t: (batch_size, hidden_size)
            Hidden state of the current time step.

        h_prev_l: (batch_size, input_size)
            Input at the current time step.

        h_prev_t: (batch_size, hidden_size)
            Hidden state at the previous time step.

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t. the input x.

        dh_prev_t: (batch_size, hidden_size)
            Derivative w.r.t. the previous hidden state h_prev_t.
        """
        batch_size = delta.shape[0]
        
        dz = delta * (1 - h_t**2)

        self.dW_ih += np.dot(dz.T, h_prev_l) / batch_size
        self.dW_hh += np.dot(dz.T, h_prev_t) / batch_size
        self.db_ih += np.sum(dz, axis=0) / batch_size
        self.db_hh += np.sum(dz, axis=0) / batch_size

        dx = np.dot(dz, self.W_ih)
        dh_prev_t = np.dot(dz, self.W_hh)

        return dx, dh_prev_t

