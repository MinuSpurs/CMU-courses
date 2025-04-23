import numpy as np
from mytorch.functional import *


class Activation(object):
    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an
    # abstract base class for the others

    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self, autograd_engine):
        self.state = None
        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.state


class Identity(Activation):
    """
    Identity function (already implemented).
    This class is a gimme as it is already implemented for you as an example.
    Just complete the forward by returning self.state.
    """

    def __init__(self, autograd_engine):
        super(Identity, self).__init__(autograd_engine)

    def forward(self, x):

        self.state = 1 * x
        self.autograd_engine.add_operation(inputs=[np.ones_like(x), x], output=self.state,
                                           gradients_to_update=[None, None],
                                           backward_operation=mul_backward)

        return self.state


class Sigmoid(Activation):
    """
    Sigmoid activation.
    Feel free to enumerate primitive operations here (you may find it helpful).
    Feel free to refer back to your implementation from part 1 of the HW for equations.
    """

    def __init__(self, autograd_engine):
        super(Sigmoid, self).__init__(autograd_engine)

    def forward(self, x):

        # TODO Compute forward with primitive operations
        # TODO Add operations to the autograd engine as you go
        i1 = -1.0 * x
        self.autograd_engine.add_operation(inputs=[-np.ones_like(x), x], output=i1,
                                           gradients_to_update=[None, None],
                                           backward_operation=mul_backward)

        i2 = np.exp(i1)
        self.autograd_engine.add_operation(inputs=[i1], output=i2,
                                           gradients_to_update=[None],
                                           backward_operation=exp_backward)

        i3 = 1.0 + i2
        self.autograd_engine.add_operation(inputs=[np.ones_like(i2), i2], output=i3,
                                           gradients_to_update=[None, None],
                                           backward_operation=add_backward)

        self.state = 1.0 / i3
        self.autograd_engine.add_operation(inputs=[np.ones_like(i3), i3], output=self.state,
                                           gradients_to_update=[None, None],
                                           backward_operation=div_backward)

        return self.state


class Tanh(Activation):
    """
    Tanh activation.
    Feel free to enumerate primitive operations here (you may find it helpful).
    Feel free to refer back to your implementation from part 1 of the HW for equations.
    """

    def __init__(self, autograd_engine):
        super(Tanh, self).__init__(autograd_engine)

    def forward(self, x):

        # TODO Compute forward with primitive operations
        # TODO Add operations to the autograd engine as you go
        # NOTE: Modification equivalent to HW3P1 modification
        self.state = np.tanh(x)
        self.autograd_engine.add_operation(inputs=[x, self.state], output=self.state,
                                           gradients_to_update=[None, None],
                                           backward_operation=tanh_backward)

        return self.state


class ReLU(Activation):
    """
    ReLU activation.
    Feel free to enumerate primitive operations here (you may find it helpful).
    Feel free to refer back to your implementation from part 1 of the HW for equations.
    """

    def __init__(self, autograd_engine):
        super(ReLU, self).__init__(autograd_engine)

    def forward(self, x):

        # TODO Compute forward with primitive operations
        # TODO Add operations to the autograd engine as you go
        self.state = np.maximum(0, x)

        self.autograd_engine.add_operation(inputs=[x], output=self.state,
                                           gradients_to_update=[None],
                                           backward_operation=max_backward)
        return self.state
