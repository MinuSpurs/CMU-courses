import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape 
        k = self.upsampling_factor 
        output_width = k * (input_width - 1) + 1 
        
        Z = np.zeros((batch_size, in_channels, output_width), dtype=A.dtype)  # TODO
        Z[:, :, ::k] = A 
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        batch_size, in_channels, output_width = dLdZ.shape
        k = self.upsampling_factor
        input_width = (output_width - 1) // k + 1
        
        dLdA = dLdA = np.zeros((batch_size, in_channels, input_width), dtype=dLdZ.dtype)  # TODO
        dLdA = dLdZ[:, :, ::k]
        
        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape
        k = self.downsampling_factor
        output_width = input_width // k
        
        Z = A[:, :, ::k]
        self.input_width = input_width

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size, in_channels, output_width = dLdZ.shape
        k = self.downsampling_factor
        
        dLdA = np.zeros((batch_size, in_channels, self.input_width), dtype=dLdZ.dtype) # TODO
        dLdA[:, :, ::k] = dLdZ
        
        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        batch_size, in_channels, input_height, input_width = A.shape
        k = self.upsampling_factor
        
        output_height = k * (input_height - 1) + 1
        output_width = k * (input_width - 1) + 1
        
        Z = np.zeros((batch_size, in_channels, output_height, output_width), dtype=A.dtype)  # TODO
        Z[:, :, ::k, ::k] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, output_height, output_width = dLdZ.shape
        k = self.upsampling_factor
        
        input_height = (output_height - 1) // k + 1
        input_width = (output_width - 1) // k + 1
        
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width), dtype=dLdZ.dtype)  # TODO
        dLdA = dLdZ[:, :, ::k, ::k]
        
        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        batch_size, in_channels, input_height, input_width = A.shape
        k = self.downsampling_factor

        output_height = input_height // k
        output_width = input_width // k
        
        Z = A[:, :, ::k, ::k]   # TODO
    
        self.input_height = input_height
        self.input_width = input_width

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, output_height, output_width = dLdZ.shape
        k = self.downsampling_factor

        dLdA = np.zeros((batch_size, in_channels, self.input_height, self.input_width), dtype=dLdZ.dtype) # TODO

        dLdA[:, :, ::k, ::k] = dLdZ

        return dLdA