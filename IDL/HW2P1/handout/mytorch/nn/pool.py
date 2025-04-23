import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape
        kernel_size = self.kernel
        output_height = input_height - kernel_size + 1
        output_width = input_width - kernel_size + 1

        Z = np.zeros((batch_size, in_channels, output_height, output_width))

        for n in range(batch_size):
            for c in range(in_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        Z[n, c, h, w] = np.max(A[n, c, h:h + kernel_size, w:w + kernel_size])

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_height, output_width = dLdZ.shape
        kernel_size = self.kernel
        input_height, input_width = self.A.shape[2], self.A.shape[3]

        dLdA = np.zeros_like(self.A)

        for n in range(batch_size):
            for c in range(in_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        
                        region = self.A[n, c, h:h + kernel_size, w:w + kernel_size]
                        max_idx = np.unravel_index(np.argmax(region, axis=None), region.shape)
                        dLdA[n, c, h + max_idx[0], w + max_idx[1]] += dLdZ[n, c, h, w]

        return dLdA
        


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape
        kernel_size = self.kernel
        output_height = input_height - kernel_size + 1
        output_width = input_width - kernel_size + 1

        Z = np.zeros((batch_size, in_channels, output_height, output_width))

        for n in range(batch_size):
            for c in range(in_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        Z[n, c, h, w] = np.mean(A[n, c, h:h + kernel_size, w:w + kernel_size])

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        batch_size, in_channels, output_height, output_width = dLdZ.shape
        kernel_size = self.kernel
        input_height, input_width = self.A.shape[2], self.A.shape[3]

        dLdA = np.zeros_like(self.A)

        for n in range(batch_size):
            for c in range(in_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        dLdA[n, c, h:h + kernel_size, w:w + kernel_size] += dLdZ[n, c, h, w] / (kernel_size * kernel_size)

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride) if stride > 1 else None  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)  

        if self.downsample2d is not None:
            Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        if self.downsample2d is not None:
            dLdZ = self.downsample2d.backward(dLdZ)
            
        dLdA = self.maxpool2d_stride1.backward(dLdZ)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  
        self.downsample2d = Downsample2d(stride) if stride > 1 else None 

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)  

        if self.downsample2d is not None:
            Z = self.downsample2d.forward(Z)

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        if self.downsample2d is not None:
            dLdZ = self.downsample2d.backward(dLdZ)

        dLdA = self.meanpool2d_stride1.backward(dLdZ)

        return dLdA