import numpy as np

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.A=A
        print('shape of A in flatten function =', self.A.shape)
        Z = np.reshape(A,(A.shape[0],A.shape[1]*A.shape[2])) 
        print('shape of z after flatten function =',Z.shape)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = dLdZ.reshape(self.A.shape)  

        return dLdA
