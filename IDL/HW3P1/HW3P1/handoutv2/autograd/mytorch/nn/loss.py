import numpy as np
from mytorch.functional import (
    matmul_backward,
    add_backward,
    sub_backward,
    mul_backward,
    div_backward,
    SoftmaxCrossEntropy_backward,
)
from mytorch.autograd_engine import *

class LossFN(object):
    """
    Interface for loss functions.

    The class serves as an abstract base class for different loss functions.
    The forward() method should be completed by the derived classes.

    This class is similar to the wrapper functions for the activations
    that you wrote in functional.py with a couple of key differences:
        1. Notice that instead of passing the autograd object to the forward
            method, we are instead saving it as a class attribute whenever
            an LossFN() object is defined. This is so that we can directly
            call the backward() operation on the loss as follows:
                >>> loss_fn = LossFN(autograd_object)
                >>> loss_val = loss_fn(y, y_hat)
                >>> loss_fn.backward()

        2. Notice that the class has an attribute called self.loss_val.
            You must save the calculated loss value in this variable.
            This is so that we do not explicitly pass the divergence to
            the autograd engine's backward method. Rather, calling backward()
            on the LossFN object will take care of that for you.

    WARNING: DO NOT MODIFY THIS CLASS!
    """

    def __init__(self, autograd_engine):
        self.autograd_engine = autograd_engine
        self.loss_val = None

    def __call__(self, y, y_hat):
        return self.forward(y, y_hat)

    def forward(self, y, y_hat):
        """
        Args:
            - y (np.ndarray) : the ground truth,
            - y_hat (np.ndarray) : the output computed by the network,

        Returns:
            - self.loss_val : the calculated loss value
        """
        raise NotImplementedError

    def backward(self):
        # Call autograd's backward here.
        self.autograd_engine.backward(self.loss_val)


class MSELoss(LossFN):
    def __init__(self, autograd_engine):
        super(MSELoss, self).__init__(autograd_engine)

    def forward(self, y, y_hat):
        # TODO: Use the primitive operations to calculate the MSE Loss
        # TODO: Remember to use add_operation to record these operations in
        #       the autograd engine after each operation

        # self.loss_val = ...
        # return self.loss_val
        self.loss_val = np.mean((y - y_hat) ** 2)
        return self.loss_val


# Hint: To simplify things you can just make a backward for this loss and not
# try to do it for every operation.
class SoftmaxCrossEntropy(LossFN):
    """
    :param A: Output of the model of shape (N, C)
    :param Y: Ground-truth values of shape (N, C)
    * To be removed
    self.A = A
    self.Y = Y
    self.N = A.shape[0]
    self.C = A.shape[-1]

    Ones_C = np.ones((self.C, 1))
    Ones_N = np.ones((self.N, 1))

    self.softmax = np.exp(self.A) / np.sum(np.exp(self.A), axis=1, keepdims=True)
    crossentropy = (-1 * self.Y * np.log(self.softmax)) @ Ones_C
    sum_crossentropy = Ones_N.T @ crossentropy
    L = sum_crossentropy / self.N
    """

    def __init__(self, autograd_engine):
        super(SoftmaxCrossEntropy, self).__init__(autograd_engine)

    def forward(self, y, y_hat):
        # TODO: calculate loss value and set self.loss_val
        # To simplify things, add a single operation corresponding to the
        # backward function created for this loss

        # y     -> ground truth
        # y_hat -> model output

        N = y_hat.shape[0]
        C = y_hat.shape[-1]
        Ones_C = np.ones((C, 1))
        Ones_N = np.ones((N, 1))
        y_exp = np.exp(y_hat)
        y_exp_sum =  np.sum(y_exp, keepdims=True)
        softmax = y_exp / y_exp_sum
        ce = (-1 * y * np.log(softmax)) @ Ones_C
        sum_ce = Ones_N.T @ ce
        self.loss_val = sum_ce / N
        self.loss_val = self.loss_val.reshape(1,)
        self.autograd_engine.add_operation(inputs=[y_hat, y],
                                           output=self.loss_val,
                                           gradients_to_update=[None, None],
                                           backward_operation=SoftmaxCrossEntropy_backward)

        return self.loss_val
