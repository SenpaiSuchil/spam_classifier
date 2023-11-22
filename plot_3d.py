import matplotlib.pyplot as plt
import numpy as np

def MLP_binary_calss_2d(X,Y,net):
    plt.figure()
    for i in range(X.shape[1]):
        if Y[0,i]==0:
            plt.plot(X[0,i],X[1,i], 'ro',markersize=9)
        else:
            plt.plot(X[0,i],X[1,i], 'bo',markersize=9)
    xmin,ymin=np.min(X[0,:])-0.5,np.min(X[1,:])-0.5
    xmax,ymax=np.max(X[0,:])+0.5,np.max(X[1,:])+0.5

    xx,yy=np.meshgrid(np.linspace(xmin,xmax,100),
                      np.linspace(ymin,ymax,100))
    
    data=[xx.ravel(),yy.ravel()]
    zz=net.predict(data)
    zz=zz.reshape(xx.shape)
    plt.contour(xx,yy,zz,[0.5], colors='k',linestyle='--',linewidths=2)
    plt.contourf(xx,yy,zz,alpha=0.8,cmap=plt.cm.RdBu)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.grid()
    plt.show()
