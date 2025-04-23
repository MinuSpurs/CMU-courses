import numpy as np
from mytorch.functional import *
from mytorch.autograd_engine import *


class Linear():
    def __init__(self, in_features, out_features, autograd_engine):
        """
        Do not modify
        """
        self.W = np.random.uniform(-np.sqrt(1 / in_features),
                                   np.sqrt(1 / in_features),
                                   size=(out_features, in_features))  # flip this to out x in to mimic pytorch
        self.b = np.random.uniform(-np.sqrt(1 / in_features),
                                   np.sqrt(1 / in_features),
                                   size=(out_features,))  # just change this to 1-d after implementing broadcasting
        
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.momentum_W = np.zeros(self.W.shape)
        self.momentum_b = np.zeros(self.b.shape)
        self.autograd_engine = autograd_engine
    
    def init_weights(self, W, b):
        self.W = W
        self.b = b

    def zero_grad(self):
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
            Computes the affine transformation forward pass of the Linear Layer

            Args:
                - x (np.ndarray): the input array,

            Returns:
                - (np.ndarray), the output of this forward computation.
        """

        # NOTE: Handle batched/unbatched inputs. Possibly include in starter code (?)
        is_batched = np.ndim(x) == 2
        if not is_batched:
            x_internal = np.expand_dims(x, axis=0).copy()
            self.autograd_engine.add_operation(inputs=[x, np.array([0])], output=x_internal,
                                               gradients_to_update=[None, None],
                                               backward_operation=expand_dims_backward)
        else:
            x_internal = x.copy()
            self.autograd_engine.add_operation(inputs=[x], output=x_internal,
                                               gradients_to_update=[None],
                                               backward_operation=identity_backward)

        # TODO: Use the primitive operations to calculate the affine transformation
        #      of the linear layer

        # TODO: Remember to use add_operation to record these operations in
        #      the autograd engine after each operation

        h = x_internal @ self.W.T
        self.autograd_engine.add_operation(inputs=[x_internal, self.W.T], output=h,
                                           gradients_to_update=[None, self.dW.T],
                                           backward_operation=matmul_backward)

        y = h + self.b.T
        self.autograd_engine.add_operation(inputs=[h, self.b.T], output=y,
                                           gradients_to_update=[None, self.db.T],
                                           backward_operation=add_backward)
        
        # TODO: remember to return the computed value
        if not is_batched:
            ret = np.squeeze(y, 0).copy()
            self.autograd_engine.add_operation(inputs=[y, np.array([0])], output=ret,
                                               gradients_to_update=[None, None],
                                               backward_operation=squeeze_backward)
        else:
            ret = y.copy()
            self.autograd_engine.add_operation(inputs=[y], output=ret,
                                               gradients_to_update=[None],
                                               backward_operation=identity_backward)
        
        return ret
