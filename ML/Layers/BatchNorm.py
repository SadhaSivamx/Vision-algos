import numpy as np
import matplotlib.pyplot as plt

#representation : (batchsize,channels,height,width)
imgbatch=np.random.rand(32,10,80,80)

#inits
batchsize,channels,height,width=imgbatch.shape
gamma=np.random.rand(1,channels,1,1)
beta=np.random.rand(1,channels,1,1)

#Standardize
def Standardize(batch):
    xmean=np.mean(batch,axis=(0,2,3),keepdims=True)
    xvar=np.var(batch,axis=(0,2,3),keepdims=True)
    return (batch-xmean)/(np.sqrt(xvar+1e-15))

#Output Y=G*X+B
St=Standardize(imgbatch)
output=gamma*St+beta

#BackPropogation
err=np.random.rand(batchsize,channels,height,width)

def GetDerivatives(batch, error):

    xmean = np.mean(batch, axis=(0,2,3), keepdims=True)
    xvar = np.var(batch, axis=(0,2,3), keepdims=True)
    std = np.sqrt(xvar + 1e-5)

    xhat = (batch - xmean) / std

    dw = np.sum(xhat * error, axis=(0,2,3), keepdims=True)
    db = np.sum(error, axis=(0,2,3), keepdims=True)

    dxhat = error * gamma

    n = batch.shape[0] * batch.shape[2] * batch.shape[3]
    sumdx = np.sum(dxhat, axis=(0,2,3), keepdims=True)
    sumdxhat = np.sum(dxhat * xhat, axis=(0,2,3), keepdims=True)
    dimg = (1 / (n * std)) * (n * dxhat - sumdx - xhat * sumdxhat)
    return dw, db, dimg

#for Backward
dw,db,dimg=GetDerivatives(imgbatch,err)

#update : gamma-=alpha*dw
alpha=0.05
gamma-=alpha*dw
beta-=alpha*db

print(dimg.shape)

