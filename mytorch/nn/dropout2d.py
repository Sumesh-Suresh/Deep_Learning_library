import numpy as np


class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p
        self.mask=[]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, eval=False):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          eval (boolean): whether the model is in evaluation mode
        Return:
          np.array of same shape as input x
        """
        # 1) Get and apply a per-channel mask generated from np.random.binomial
        # 2) Scale your output accordingly
        # 3) During test time, you should not apply any mask or scaling.
        #self.x= x
        batch, inp_channel,_,_ = x.shape
        
        if not eval:
            
            self.mask = np.random.binomial(1, 1 - self.p, size=(x.shape[0], x.shape[1], 1, 1))
            self.mask = np.tile(self.mask, (1, 1, x.shape[2], x.shape[3]))
            self.x=x * self.mask / (1 - self.p)
        elif eval:
            self.x=x
          
        return self.x

          
        
        

    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """
        # 1) This method is only called during training.
        # 2) You should scale the result by chain rule
        
        return self.mask*delta/(1-self.p)

