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

        self.dLdW = np.zeros(self.W.shape,dtype='float64')
        self.dLdb = np.zeros(self.b.shape,dtype='float64')

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        Z = np.zeros((self.A.shape[0],self.out_channels,self.A.shape[2]-self.kernel_size+1),dtype='float64') # TODO
        
        #convolusion for Z
        for i in range(Z.shape[2]):
            window=self.A[:,:,i:i+self.kernel_size]
            Z[:,:,i]=np.tensordot(window,self.W,axes=((1,2),(1,2)))+self.b
        
        # print('shape of A=', A.shape)
        # print('shape of W=', self.W.shape)
        # print('shape of Z = ',Z.shape)
        # print('shape of Z = ',Z.shape)
        # print('shape of b = ',self.b.shape)
        # print('shape of window',window.shape)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        self.dLdb=np.sum(np.sum(dLdZ,axis=0),axis=1)         
        self.dLdA = np.zeros(self.A.shape,dtype='float64')  
        
        #convolution for dLdW
        for i in range(self.W.shape[2]):
            self.dLdW[:,:,i]=np.tensordot(dLdZ,self.A[:,:,i:i+dLdZ.shape[2]],axes=((2,0),(2,0)))
        
        #pad dldz
        dLdZ_pad=np.pad(dLdZ,((0,0),(0,0),(self.kernel_size-1,self.kernel_size-1)),mode='constant',constant_values=0)
        
        #flip filter
        flip_w=np.flip(self.W,2)

        #convolution for dLdA
        for i in range(self.A.shape[2]):
            self.dLdA[:,:,i]=np.tensordot(dLdZ_pad[:,:,i:i+flip_w.shape[2]],flip_w,axes=((1,2),(0,2)))
        
        # print('shape of A',self.A.shape)
        # print('shape of dldz',dLdZ.shape)
        # print('shape of dldb',self.dLdb.shape)
        # print('shape of b =' ,self.b.shape)
        # print('kernel size',self.kernel_size)
        # print(' shape of dldw =', self.dLdW.shape)
        # print('shape of dlda=', self.dLdA.shape)
       

        return self.dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels,out_channels,kernel_size,weight_init_fn=None, bias_init_fn=None) 
        self.downsample1d = Downsample1d(self.stride) 

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        F = self.conv1d_stride1.forward(A)
        # downsample
        Z = self.downsample1d.forward(F)  

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        B = self.downsample1d.backward(dLdZ)
        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(B)  

        return dLdA
