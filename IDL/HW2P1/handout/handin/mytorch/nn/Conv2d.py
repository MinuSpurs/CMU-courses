import numpy as np
from resampling import *

class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        batch_size, in_channels, input_height, input_width = A.shape
        out_channels, _, kernel_size, _ = self.W.shape
        output_height = input_height - kernel_size + 1
        output_width = input_width - kernel_size + 1

        Z = np.zeros((batch_size, out_channels, output_height, output_width)) 

        for n in range(batch_size):
            for c_out in range(out_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        patch = A[n, :, h:h + kernel_size, w:w + kernel_size]
                        Z[n, c_out, h, w] = np.sum(patch * self.W[c_out]) + self.b[c_out]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        
        batch_size, out_channels, output_height, output_width = dLdZ.shape
        _, in_channels, kernel_size, _ = self.W.shape
        input_height, input_width = self.A.shape[2], self.A.shape[3]

        dLdA = np.zeros(self.A.shape)

        self.dLdW = np.zeros_like(self.W)
        self.dLdb = np.zeros_like(self.b)

        for n in range(batch_size):
            for c_out in range(out_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        self.dLdW[c_out] += dLdZ[n, c_out, h, w] * self.A[n, :, h:h + kernel_size, w:w + kernel_size]
                        self.dLdb[c_out] += dLdZ[n, c_out, h, w]
                        dLdA[n, :, h:h + kernel_size, w:w + kernel_size] += dLdZ[n, c_out, h, w] * self.W[c_out]

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.pad = padding

        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)   # TODO
        self.downsample2d = Downsample2d(stride) if stride > 1 else None   # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        if self.pad > 0:
            A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant', constant_values=0) # TODO

        Z = self.conv2d_stride1.forward(A)

        if self.downsample2d is not None:
            Z = self.downsample2d.forward(Z) 

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        if self.downsample2d is not None:
            dLdZ = self.downsample2d.backward(dLdZ)

        dLdA = self.conv2d_stride1.backward(dLdZ) 

        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad] 

        return dLdA