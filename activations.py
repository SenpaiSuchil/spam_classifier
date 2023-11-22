import numpy as np 

#funciones de activacion para capas de salida
def linear (z,derivative=False):
    a=z
    if derivative:
        da=a*(1-a)
        return a,da
    return a

def logistic (z,derivative=False):
    a=1/(1+np.exp(-z))
    if derivative:
        da=a*(1-a)
        return a,da
    return a

def softmax(z, derivative=False):
    ez=np.exp(z-np.max(z,axis=0))
    a=ez/np.sum(ez,axis=0)
    if derivative:
        da=np.ones(z.shape)
        return a, da
    return a 

#funciones de activacion para capas ocultas
def tanh (z,derivative=False):
    a=np.tanh(z)
    if derivative:
        da=(1+a)*(1-a)
        return a,da
    return a

def relu (z,derivative=False):
    a=z*(z>=0)
    if derivative:
        da=1.0*(z>=0)
        return a,da
    return a

def logistic_hidden(z, derivative=False):
    a=1/(1+np.exp(-z))
    if derivative:
        da=a*(1-a)
        return a,da
    return a


