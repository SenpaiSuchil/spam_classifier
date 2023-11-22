from activations import *
import numpy as np
import time


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
        A=X
        for l in range(1,self.L+1):
            Z=self.w[l]@A+self.b[l]
            A=self.f[l](Z)
        return A
    
    def fit(self,X,Y,epochs=900,lr=0.2):
        p=X.shape[1]
        for e in range(epochs):
            begin= time.time()#tiempo inicial
            #print(f"iniciando epoca: {e}")
            #inicializar contenedores
            A=[None]*(self.L+1)
            dA=[None]*(self.L+1)
            lg=[None]*(self.L+1)

            #propagación
            A[0]=X
            for l in range(1,self.L+1):
                Z=self.w[l]@A[l-1]+self.b[l]
                A[l],dA[l]=self.f[l](Z,derivative=True)

            #back propagation
            for l in range(self.L,0,-1):
                if l==self.L:
                    lg[l]=(Y-A[l])*dA[l]
                else:
                    lg[l]=(self.w[l+1].T @ lg[l+1])*dA[l]
            
            #actualizacion de pesos y bias

            for l in range(1,self.L+1):
                self.w[l]+=(lr/p)*(lg[l] @ A[l-1].T)
                self.b[l]+=(lr/p)*np.sum(lg[l])
            end=time.time()

            print(f"tiempo total de ejecucion en epoca {e}: {end-begin:.2f}")#check para ver cuanto tardó


