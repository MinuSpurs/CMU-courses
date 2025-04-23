import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        batch_size, in_channels, input_size = A.shape
        out_channels, _, kernel_size = self.W.shape
        output_size = input_size - kernel_size + 1

        Z = np.zeros((batch_size, out_channels, output_size))

        for n in range(batch_size):
            for c_out in range(out_channels):
                for i in range(output_size):
                    Z[n, c_out, i] = np.sum(
                        A[n, :, i:i + kernel_size] * self.W[c_out]) + self.b[c_out]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        _, in_channels, kernel_size = self.W.shape
        input_size = self.A.shape[2]

        dLdA = np.zeros(self.A.shape)

        self.dLdW = np.zeros_like(self.W)
        self.dLdb = np.zeros_like(self.b)

        for n in range(batch_size):
            for c_out in range(out_channels):
                for i in range(output_size):
                    self.dLdW[c_out] += dLdZ[n, c_out, i] * self.A[n, :, i:i + kernel_size]
                    dLdA[n, :, i:i + kernel_size] += dLdZ[n, c_out, i] * self.W[c_out]

        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):

        self.stride = stride
        self.pad = padding

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride) if stride > 1 else None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        if self.pad > 0:
            A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant', constant_values=0)

        Z = self.conv1d_stride1.forward(A)

        if self.downsample1d is not None:
            Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        if self.downsample1d is not None:
            dLdZ = self.downsample1d.backward(dLdZ)

        dLdA = self.conv1d_stride1.backward(dLdZ)

        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad:-self.pad]

        return dLdA