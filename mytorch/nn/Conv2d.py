import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

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
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        Z = np.zeros((self.A.shape[0],
                    self.out_channels,
                    self.A.shape[2]-self.kernel_size+1,
                    self.A.shape[3]-self.kernel_size+1), dtype='float64') 

        for i in range(Z.shape[2]):
            for j in range(Z.shape[3]):
                window=self.A[:,:,i:i+self.kernel_size,j:j+self.kernel_size]
                Z[:,:,i,j]=np.tensordot(window,self.W,axes=((1,2,3),(1,2,3)))+self.b
        
        # print('shape of A=', A.shape)
        # print('shape of W=', self.W.shape)
        # print('shape of Z = ',Z.shape)
        # print('shape of b = ',self.b.shape)
        # print('shape of window',window.shape)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        self.dLdW = np.zeros(self.W.shape,dtype='float64')  # TODO
        self.dLdb = np.sum(dLdZ,axis=(0,2,3))  # TODO
        self.dLdA = np.zeros(self.A.shape,dtype='float64') # TODO

        #convolving for dldw
        for i in range(self.dLdW.shape[2]):
            for j in range(self.dLdW.shape[3]):
                window=self.A[:,:,i:i+dLdZ.shape[2],j:j+dLdZ.shape[3]]
                self.dLdW[:,:,i,j]=np.tensordot(dLdZ,window,axes=((0,2,3),(0,2,3)))
        
        #pad dldz
        dldz_pad=np.pad(dLdZ,((0,0),(0,0),(self.kernel_size-1,self.kernel_size-1),(self.kernel_size-1,self.kernel_size-1)),mode='constant',constant_values=0)

        #flipping weights
        flip_w=np.flip(self.W,3)
        flip_w=np.flip(flip_w,2)
        
        #convulsion for dlda 
        for i in range(self.dLdA.shape[2]):
            for j in range(self.dLdA.shape[3]):
                window=dldz_pad[:,:,i:i+flip_w.shape[2],j:j+flip_w.shape[3]]
                self.dLdA[:,:,i,j]=np.tensordot(window,flip_w,axes=((1,2,3),(0,2,3)))
        
        return self.dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
       
        F = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(F) 
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        B = self.downsample2d.backward(dLdZ)
        
        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(B)
        
        return dLdA
