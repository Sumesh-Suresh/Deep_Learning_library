import numpy as np
import os

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
        self.N = self.A.shape[0] 
        self.C = self.A.shape[1]  
        se = (np.subtract(self.A,self.Y))*(np.subtract(self.A,self.Y))  
        sse = (np.ones((self.N,1)).T)@se@np.ones((self.C,1)) 
        mse = sse/(2*self.N*self.C) 

        print('mse =',mse)
        # print('se',se.shape)
        # print('N',self.N)
        # print('C',self.C)
        return mse

    def backward(self):

        dLdA = np.subtract(self.A,self.Y)/(self.N*self.C) 

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
        N = self.A.shape[0] 
        C = self.A.shape[1] 

        Ones_C = np.ones((C,1))
        Ones_N = np.ones((N,1))
        self.softmax = np.array([np.exp(i)/sum(np.exp(i)) for i in self.A]) 
        
        print('Y',self.Y.shape)
        print('softmax',self.softmax.shape)
        print('Ones C',Ones_C.shape)
        crossentropy = -1*(self.Y*np.log(self.softmax))@Ones_C 
        sum_crossentropy = np.dot(Ones_N.T,crossentropy) 
        L = sum_crossentropy / N

        return L

    def backward(self):

        dLdA = self.softmax-self.Y  
        return dLdA


# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss

    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        
        self.logits = x
        self.labels = y
        N=self.logits.shape[0]
        C=self.logits.shape[1]

        Ones_C=np.ones((C,1))
        Ones_N=np.ones((N,1))

        self.softmax= np.array([np.exp(i)/np.sum(np.exp(i)) for i in self.logits])
        self.cross_entropy = -1*(self.labels*np.log(self.softmax))@Ones_C
        sum_crossentropy = np.dot(Ones_N.T,self.cross_entropy) 
        self.loss = sum_crossentropy / N
        

        return self.loss

    def backward(self):
        """
        TODO: Implement this function similar to how you did for HW1P1 or HW2P1.
        Return:
            out (np.array): (batch size, 10)
        """

        self.gradient = self.softmax-self.labels

        return self.gradient
