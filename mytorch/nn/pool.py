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
        self.A=A
        Z=np.zeros((A.shape[0],A.shape[1],A.shape[2]-self.kernel+1,A.shape[3]-self.kernel+1),dtype='float64')
        self.positionx=np.zeros((A.shape[0],A.shape[1],A.shape[2]-self.kernel+1,A.shape[3]-self.kernel+1),dtype='float64')
        self.positiony=np.zeros((A.shape[0],A.shape[1],A.shape[2]-self.kernel+1,A.shape[3]-self.kernel+1),dtype='float64')
        
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):                 #can optimize this portion
                for x in range(Z.shape[2]):
                    for y in range(Z.shape[3]):
                        window=self.A[i,j,x:x+self.kernel,y:y+self.kernel]
                        # getting arg max of maximum element
                        positions=list(np.unravel_index(np.argmax(window),window.shape))
                        # saving positions 
                        self.positionx[i,j,x,y]=positions[0]
                        self.positiony[i,j,x,y]=positions[1]
                        #forward
                        Z[i,j,x,y]=window[positions[0],positions[1]]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA=np.zeros(self.A.shape,dtype='float64')

        for i in range(dLdA.shape[0]):
            for j in range(dLdA.shape[1]):
                for x in range(dLdA.shape[2]-self.kernel+1):            
                    for y in range(dLdA.shape[3]-self.kernel+1):
                        m=int(self.positionx[i,j,x,y])
                        n=int(self.positiony[i,j,x,y])
                        dLdA[i,j,x:x+self.kernel,y:y+self.kernel][m,n]+=dLdZ[i,j,x,y]
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
        self.A=A
        Z=np.zeros((A.shape[0],A.shape[1],A.shape[2]-self.kernel+1,A.shape[3]-self.kernel+1),dtype='float64')
        
        # for i in range(Z.shape[0]):
        #     for j in range(Z.shape[1]):
        #         for x in range(Z.shape[2]):
        #             for y in range(Z.shape[3]):
        #                 window=self.A[i,j,x:x+self.kernel,y:y+self.kernel]
        #                 Z[i,j,x,y]=np.mean(window)
        for i in range(A.shape[2]-self.kernel+1):
            for j in range(A.shape[3]-self.kernel+1):
                Z[:,:,i,j]=np.mean(A[:,:,i:i+self.kernel,j:j+self.kernel],axis=(2,3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA=np.zeros(self.A.shape,dtype='float64')
        # for i in range(dLdA.shape[0]):
        #     for j in range(dLdA.shape[1]):
        #         for x in range(dLdA.shape[2]-self.kernel+1):
        #             for y in range(dLdA.shape[3]-self.kernel+1):
        #                 for n in range(self.kernel):
        #                     for m in range(self.kernel):
        #                         dLdA[i,j,x+n,y+m]+=(1/self.kernel**2)*dLdZ[i,j,x,y]
        for i in range(dLdA.shape[2]-self.kernel+1):
            for j in range(dLdA.shape[3]-self.kernel+1):
                dLdA[:,:,i:i+self.kernel,j:j+self.kernel]+=dLdZ[:,:,i,j].reshape((self.A.shape[0],self.A.shape[1],1,1))*(1/self.kernel**2)
        

        return dLdA 


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        F=self.maxpool2d_stride1.forward(A)
        Z=self.downsample2d.forward(F)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        B=self.downsample2d.backward(dLdZ)
        dLdA=self.maxpool2d_stride1.backward(B)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        F=self.meanpool2d_stride1.forward(A)
        Z=self.downsample2d.forward(F)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        B=self.downsample2d.backward(dLdZ)
        dLdA=self.meanpool2d_stride1.backward(B)
        return dLdA
