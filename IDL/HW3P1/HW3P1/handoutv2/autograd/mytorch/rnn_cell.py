import numpy as np
from .nn.linear import *
from .nn.activation import *
from mytorch.functional import *
from mytorch.autograd_engine import *

"""
NOTE: Since you shoud have already implemented(or recieved) the Linear class,
We can model RNNCell's as composable Linear transformations.
An Elman RNN cell with some activation function ('act_fn') is given by: 

ht = act_fn(Wih xt + bih + Whh ht−1 + bhh)

where,
xt   : input features at timestep t
ht-1 : hidden state at timestep t-1
Wih  : input-to-hidden weights
bih  : input-to-hidden bias
Whh  : hidden-to-hidden weights
bhh  : hidden-to-hidden bias
ht   : hidden state at timestep t
"""


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size, autograd_engine, act_fn=Tanh):
        """DO NOT MODIFY!"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.autograd_engine = autograd_engine
        self.activation = act_fn(self.autograd_engine)

        # TODO: Init two Linear layers
        # NOTE: Make sure you pass the Autograd Engine
        self.ih = NotImplemented
        self.hh = NotImplemented

        """DO NOT MODIFY"""
        self.zero_grad()

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        """DO NOT MODIFY"""
        self.ih.init_weights(W_ih, b_ih)
        self.hh.init_weights(W_hh, b_hh)

    def zero_grad(self):
        """DO NOT MODIFY"""
        self.ih.zero_grad()
        self.hh.zero_grad()

    def __call__(self, x, h_prev_t, scale_hidden=None):
        """DO NOT MODIFY"""
        return self.forward(x, h_prev_t, scale_hidden)

    def forward(self, x, h_prev_t, scale_hidden=None):
        """
        RNN Cell forward (single time step).

        Input (see writeup for explanation)
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

        """
        ht = act_fn(Wih xt + bih + Whh ht−1 + bhh)
        """

        # TODO: Apply the Linear Transformation on the input features
        input_transform = NotImplemented

        # TODO: Apply the Linear Transformation on the hidden features
        hidded_transform = NotImplemented

        # TODO: Multiply the hidden transformation with optional scale factor
        # NOTE: Remember to add any operations.
        # NOTE: Also remember np.ndarrays with the same views cannot be added to the gradient buffer.
        # NOTE: This is done to be able to later use RNNCell's to create GRUCells
        if scale_hidden is not None:
            scale_hidden = NotImplemented
        else:
            scale_hidden = NotImplemented

        # TODO: Add the input Linear Transformation and the hidden Linear Transformation
        total_transform = NotImplemented

        # TODO: Apply the activation function
        h_t = NotImplemented

        raise NotImplementedError
