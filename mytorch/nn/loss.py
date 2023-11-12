import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = self.A.shape[0] # TODO
        self.C = self.A.shape[1]  # TODO
        se = (np.subtract(self.A,self.Y))*(np.subtract(self.A,self.Y))  # TODO
        sse = (np.ones((self.N,1)).T)@se@np.ones((self.C,1)) # TODO
        mse = sse/(2*self.N*self.C)  # TODO

        print('mse =',mse)
        # print('se',se.shape)
        # print('N',self.N)
        # print('C',self.C)
        return mse

    def backward(self):

        dLdA = np.subtract(self.A,self.Y)/(self.N*self.C) #TODO

        return dLdA.astype(dtype='f')


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = self.A.shape[0]  # TODO
        C = self.A.shape[1]  # TODO

        Ones_C = np.ones((C,1))  # TODO
        Ones_N = np.ones((N,1))  # TODO

        self.softmax = np.array([np.exp(i)/sum(np.exp(i)) for i in self.A]) # TODO
        
        print('Y',self.Y.shape)
        print('softmax',self.softmax.shape)
        print('Ones C',Ones_C.shape)
        crossentropy = -1*(self.Y*np.log(self.softmax))@Ones_C # TODO
        sum_crossentropy = np.dot(Ones_N.T,crossentropy) # TODO
        L = sum_crossentropy / N

        return L

    def backward(self):

        dLdA = self.softmax-self.Y  # TODO
        return dLdA
