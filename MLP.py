import numpy as np
from activations import *

class MLP:
    def __init__(self,layers_dims,
                 hidden_activation=tanh,
                 output_activation=logistic):
        
        #atributos
        self.L=len(layers_dims)-1
        self.w=[None]*(self.L+1)
        self.b=[None]*(self.L+1)
        self.f=[None]*(self.L+1)

        #inicialización
        for l in range(1,self.L+1):
            self.w[l]=-1+2*np.random.rand(layers_dims[l],
                                          layers_dims[l-1])
            self.b[l]=-1+2*np.random.rand(layers_dims[l],1)

            if l==self.L:
                self.f[l]=output_activation
            else:
                self.f[l]=hidden_activation

    def predict(self,X):
        a=np.asanyarray(X)
        for l in range(1,self.L+1):
            z=np.dot(self.w[l],a) + self.b[l]
            a=self.f[l](z)
        return a

    def fit (self,X,Y,epochs=500,lr=0.1):
        P=X.shape[1]
        for _ in range(epochs):
            for p in range(P):
                #inicializar activaciones
                a=[None]*(self.L+1)
                da=[None]*(self.L+1)
                lg=[None]*(self.L+1)

                #propagacion
                
                a[0]=X[:,p].reshape(-1,1)

                for l in range(1, self.L+1):
                    z=np.dot(self.w[l],a[l-1])+self.b[l]
                    a[l],da[l]=self.f[l](z,derivative=True)

                #backpropagation
                for l in range(self.L,0,-1):
                    if l==self.L:
                        
                        lg[l]=(Y[:,p].reshape(-1,1)-a[l])*da[l]
                        
                    else:
                        lg[l]=np.dot(self.w[l+1].T,lg[l+1])*da[l]
                    
                #gradiente descendente
                for l in range(1,self.L+1):
                    self.w[l]+=lr*np.dot(lg[l],a[l-1].T)
                    self.b[l]+=lr*lg[l]


