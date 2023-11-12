import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.r = self.r_act(self.Wrx@self.x+self.brx+self.Wrh@self.hidden+self.brh)
        self.z = self.z_act(self.Wzx@self.x+self.bzx+self.Wzh@self.hidden+self.bzh)
        self.n = self.h_act(self.Wnx@self.x+self.bnx+self.r*(self.Wnh@self.hidden+self.bnh))
        h_t = (1-self.z)*self.n+self.z*self.hidden
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t
        # raise NotImplementedError

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        self.x= np.reshape(self.x,(self.d,1))
        self.hidden = np.reshape(self.hidden,(self.h,1))
        self.r = np.reshape(self.r,(self.h,1))
        self.z= np.reshape(self.z,(self.h,1))
        self.n = np.reshape(self.n,(self.h,1))
        delta = np.reshape(delta,(self.h,1))

        self.bzh = np.reshape(self.bzh,(self.h,1))
        self.bnh = np.reshape(self.bnh,(self.h,1))
        self.brh = np.reshape(self.brh,(self.h,1))
        
        self.bzx = np.reshape(self.bzx,(self.h,1))
        self.bnx = np.reshape(self.bnx,(self.h,1))
        self.brx = np.reshape(self.brx,(self.h,1))

        dz=delta*(self.hidden-self.n)
        dn=delta*(1-self.z)
        
        dr=dn*self.h_act.backward(self.n)*(self.Wnh@self.hidden+self.bnh)

        #for r
        dr_act= self.r_act.backward(self.r)
        dr_act.reshape((self.h,1))
        print('shape of dr',dr.shape)
        print('sape of dr_act',dr_act.shape)
        print('shape of dr*dr_act',(dr*dr_act).shape)
        print('shape of hidden.t',(self.hidden.T).shape)
        
        self.dWrh = (dr*dr_act)@self.hidden.T
        self.dWrx=  (dr*dr_act)@self.x.T
        self.dbrx=  np.reshape(dr*dr_act,(self.h,))
        self.dbrh=  np.reshape(dr*dr_act,(self.h,))

        #for z
        dz_act= np.reshape(self.z_act.backward(self.z),(self.h,1))
        self.dWzh = dz*dz_act@self.hidden.T
        self.dWzx=  dz*dz_act@self.x.T
        self.dbzx=  np.reshape(dz*dz_act,(self.h,))
        self.dbzh=  np.reshape(dz*dz_act,(self.h,))

        #for n
        dh_act = np.reshape(self.h_act.backward(self.n),(self.h,1))
        self.dWnh = self.r*dn*dh_act@self.hidden.T
        self.dWnx =  dn*dh_act@self.x.T
        self.dbnx =  np.reshape(dn*dh_act,(self.h,))
        self.dbnh =  np.reshape(self.r*dn*dh_act,(self.h,))

        dx = (((dn*dh_act).T)@self.Wnx) + (((dr*dr_act).T)@self.Wrx) + (((dz*dz_act).T)@self.Wzx)
        dh_prev_t = (delta*self.z).T + (((dn*dh_act*self.r).T)@self.Wnh) + (((dz*dz_act).T)@self.Wzh) + (((dr*dr_act).T)@self.Wrh)
        
        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t
        # raise NotImplementedError
