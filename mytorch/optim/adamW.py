# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class AdamW():
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.l = model.layers
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.weight_decay=weight_decay

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]

        self.m_what= [np.zeros(l.W.shape, dtype="f") for l in model.layers]
        self.v_what=[np.zeros(l.b.shape, dtype="f") for l in model.layers]
        self.m_bhat= [np.zeros(l.W.shape, dtype="f") for l in model.layers]
        self.v_bhat=[np.zeros(l.b.shape, dtype="f") for l in model.layers]

    def step(self):

        self.t += 1
        for layer_id, layer in enumerate(self.l):

            # TODO: Calculate updates for weight
            self.m_W[layer_id]= self.beta1*self.m_W[layer_id]+(1-self.beta1)*layer.dLdW
            self.v_W[layer_id]=self.beta2*self.v_W[layer_id]+(1-self.beta2)*np.power(layer.dLdW,2)
            # TODO: calculate updates for bias
            self.m_b[layer_id]= self.beta1*self.m_b[layer_id]+(1-self.beta1)*layer.dLdb
            self.v_b[layer_id]=self.beta2*self.v_b[layer_id]+(1-self.beta2)*np.power(layer.dLdb,2)

            #initialize
            self.m_what[layer_id]=self.m_W[layer_id]/(1-self.beta1**self.t)
            self.v_what[layer_id]=self.v_W[layer_id]/(1-self.beta2**self.t)
            self.m_bhat[layer_id]=self.m_b[layer_id]/(1-self.beta1**self.t)
            self.v_bhat[layer_id]=self.v_b[layer_id]/(1-self.beta2**self.t)
            
            # TODO: Perform weight and bias updates with weight decay
            layer.W=layer.W-self.lr*self.m_what[layer_id]/np.sqrt(self.v_what[layer_id]+self.eps)
            # print('weight update performed')
            layer.b=layer.b-self.lr*self.m_bhat[layer_id]/np.sqrt(self.v_bhat[layer_id]+self.eps)
            #adamW extra step with weight decay
            layer.W=layer.W-layer.W*self.lr*self.weight_decay
            layer.b=layer.b-layer.b*self.lr*self.weight_decay
            # print('bias update performed')
            #raise NotImplementedError("AdamW Not Implemented")
