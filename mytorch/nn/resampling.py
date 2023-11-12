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
        self.A=A

        Z = np.zeros((A.shape[0],A.shape[1],self.upsampling_factor*(A.shape[2]-1)+1),dtype='float64') 
        for i in range(Z.shape[0]):
            k=0                         #future np.kron
            for j in Z[i]:
                j[::self.upsampling_factor]=A[i,k,:]
                k+=1
        
        # print('shape of A=',A.shape)
        # print('shape of Z',Z.shape)
        # print('inside Z',Z[::self.upsampling_factor].shape)
        # print('upsampling factor=',self.upsampling_factor)
        # print('final shape of Z=',Z.shape)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = np.zeros(self.A.shape, dtype='float64') # TODO
        for i in range(dLdA.shape[0]): # iteration batches
            for j in range(dLdA.shape[1]):
                dLdA[i,j,:]=dLdZ[i,j,::self.upsampling_factor]
               

        # print('resamplingdlda shape=',dLdA.shape)
        # print('resampling dldz shape=',dLdZ.shape)
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

        Z = np.zeros((A.shape[0],A.shape[1],int((A.shape[2]-1)/self.downsampling_factor+1)), dtype='float64') 
        
        for i in range(A.shape[0]): # iteration batches
            k=0
            for j in A[i]:  # elements in each batches
                Z[i,k,:]=j[::self.downsampling_factor]
                k+=1
        # print('k=',self.downsampling_factor)
        # print('shape of A=',A.shape)
        # print('shape of Z=',Z.shape)

        self.req_shap=A.shape   # using this dimension later for backward pass

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = np.zeros((self.req_shap[0],self.req_shap[1],self.req_shap[2]),dtype='float64') 
        for i in range(dLdZ.shape[0]):
            k=0
            for j in dLdA[i]:
                j[::self.downsampling_factor]=dLdZ[i,k,:]
                k+=1  
        
        # print('shape of dldz=',dLdZ.shape)
        # print('shape of dlda', dLdA.shape)
        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        self.A=A
        Z = np.zeros((A.shape[0],A.shape[1],self.upsampling_factor*(A.shape[2]-1)+1,self.upsampling_factor*(A.shape[3]-1)+1),dtype='float64') 

        for i in range(Z.shape[0]):          # iteration over batches
            for j in range(Z.shape[1]):      # iteration over channels
                Z[i,j,::self.upsampling_factor,::self.upsampling_factor]=A[i,j,:,:]    # every alternate rows and columns of Z matched to A
                
        
        # print('shape of A=',A.shape)
        # print('shape of Z',Z.shape)
        # print('upsampling factor=',self.upsampling_factor)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros(self.A.shape,dtype='float64')  
        for i in range(dLdZ.shape[0]):          # iteration over batches
            for j in range(dLdZ.shape[1]):      # iteration over channels
                dLdA[i,j,:,:]=dLdZ[i,j,::self.upsampling_factor,::self.upsampling_factor]   # every alternate rows and columns of dldz matched to dldA       

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        Z = np.zeros((A.shape[0],A.shape[1],int((A.shape[2]-1)/self.downsampling_factor+1),int((A.shape[3]-1)/self.downsampling_factor+1)), dtype='float64') 
        
        # NOTE  used own method here to implement the 3rd dimension of z. 
        #  According to handout 3rd dimension =  A.shape[2]//self.downsampling_factor+1 
        # but handout didnt work :(

        for i in range(A.shape[0]):          # iteration over batches
            for j in range(A.shape[1]):      # iteration over channels
                Z[i,j,:,:]=A[i,j,::self.downsampling_factor,::self.downsampling_factor]   # every alternate rows and columns of dldz matched to dldA       
        
        self.req_dim=A.shape # required dimension , later used for backward pass.

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros((self.req_dim[0],self.req_dim[1],self.req_dim[2],self.req_dim[3]),dtype='float64')  
        for i in range(dLdA.shape[0]):          # iteration over batches
            for j in range(dLdA.shape[1]):      # iteration over channels
                dLdA[i,j,::self.downsampling_factor,::self.downsampling_factor]=dLdZ[i,j,:,:]    # every alternate rows and columns of Z matched to A
                
        return dLdA
