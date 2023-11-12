import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        self.Z = Z
        # print('shape of Z',self.Z.shape)
        self.N = self.Z.shape[0]*self.Z.shape[2]*self.Z.shape[3] 
        self.M = np.sum(self.Z,axis=(0,2,3),keepdims=True)/self.N
        self.V =  np.sum((self.Z-self.M)**2,axis=(0,2,3),keepdims=True)/self.N 
       
        if eval == False:
            # training mode
            self.NZ = (self.Z-self.M)/np.sqrt(self.V+self.eps) 
            self.BZ =  (self.NZ*self.BW)+self.Bb 
            # print('shape of NZ,x',self.NZ.shape,self.M.shape)
            self.running_M = self.alpha*self.running_M+(1-self.alpha)*self.M  
            self.running_V = self.alpha*self.running_V+(1-self.alpha)*self.V 

            

        if eval:
            # inference mode
            NZ = (self.Z-self.running_M)/np.sqrt(self.running_V+self.eps)  # TODO
            self.BZ =  (NZ*self.BW)+self.Bb # TODO

            
        # self.NZ =  # TODO
        # self.BZ = None  # TODO

        # self.running_M = None  # TODO
        # self.running_V = None  # TODO
        
            

        return self.BZ

    def backward(self, dLdBZ):
  
        self.dLdBb = np.sum(dLdBZ,axis=(0,2,3),keepdims=True)
        self.dLdBW = np.sum(dLdBZ*self.NZ,axis=(0,2,3),keepdims=True) # TODO
         # TODO


        dLdNZ = dLdBZ*self.BW  # TODO
        dLdV = -0.5*(np.sum(dLdNZ*(self.Z-self.M)*((self.V+self.eps)**-(1.5)),axis=(0,2,3),keepdims=True)) # TODO
        dNZdM= -((self.V+self.eps)**(-0.5))-0.5*(self.Z-self.M)*((self.V+self.eps)**(-1.5))*(-2/self.N)*np.sum(self.Z-self.M, axis=(0,2,3),keepdims=True)# TODO
        dLdM=np.sum(dLdNZ*dNZdM,axis=(0,2,3),keepdims=True)
        dLdZ = dLdNZ*((self.V+self.eps)**(-0.5))+dLdV*((2/self.N)*(self.Z-self.M))+dLdM/self.N # TODO
        # print('shape of V =',self.V.shape)
        # print('shape of dldv =',dLdV.shape)
        # print('N =',self.N)
        # print('shape of Z =',self.Z.shape)
        # print('shape of M =',self.M.shape)
        # print('shape of dldm =',dLdM.shape)
        # print('shape of dldbz =', dLdBZ.shape)
        # print('shape of NZ = ',self.NZ.shape)

        return dLdZ


        dLdNZ = None  # TODO
        dLdV = None  # TODO
        dLdM = None  # TODO

        dLdZ = None  # TODO

        raise NotImplemented
