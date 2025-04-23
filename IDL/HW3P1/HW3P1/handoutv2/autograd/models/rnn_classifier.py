import sys

sys.path.append("./")
from mytorch.nn.linear import *
from mytorch.rnn_cell import *
from mytorch.autograd_engine import *
import numpy as np


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(
        self, input_size, hidden_size, output_size, autograd_engine, num_layers=2
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.autograd_engine = autograd_engine

        # TODO: Understand then uncomment this code :)
        """
        self.rnn = [
            RNNCell(input_size, hidden_size, self.autograd_engine) if i == 0
            else RNNCell(hidden_size, hidden_size, self.autograd_engine)
            for i in range(num_layers)
        ]
        
        self.output_layer = Linear(hidden_size, output_size, self.autograd_engine)
        """
        # store hidden states at each time step, [(seq_len+1) * (num_layers) * (batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """
        DO NOT MODIFY!
        Initialize weights.

        Parameters
        ----------
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.init_weights(linear_weights[0], linear_weights[1])

    def __call__(self, x, h_0=None):
        """DO NOT MODIFY!"""
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """
        RNN forward, multiple layers, multiple time steps.

        Parameters
        ----------
        x: (batch_size, seq_len, input_size)
            Input

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits: (batch_size, output_size)

        Output (y): logits

        """
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            # NOTE: Initialize hidden state for each layer for timestep 0.
            # A list of length num_layers with np.ndarrays of shape (batch_size, hidden_size) initialized with zeros.
            # Since, the same np.ndarray with diffrent views cannot be added to the gradient buffer,
            # This creates an independent np.ndarray for each hidden state which can
            # be succesfully added to the computational graph for gradient tracking
            hidden = [
                np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)
            ]
        else:
            hidden = h_0

        # Save x and append a copy(!!) of the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())
        logits = None

        # Add your code here --->
        # (More specific pseudocode may exist in lecture slides)
        # Iterate through the sequence
        #   Iterate over the length of your self.rnn (through the layers)
        #       Run the rnn cell with the correct parameters and update
        #       the parameters as needed. Update hidden.
        #   Similar to above, append a copy of the current hidden array to the hiddens list

        # TODO

        # Get the outputs from the last time step using the linear layer and return it
        # <--------------------------
        # return logits
        raise NotImplementedError
