import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def func(m, b, x):
    return sigmoid(np.matmul(m,(x.T))+b)


def berror(ytrue, ypred):
    n = len(ytrue)
    ypred = np.clip(ypred, 1e-15, 1 - (1e-15))
    error = ytrue * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred)
    return -np.sum(error) / n

#x :  (6, 2) y:  (6, 1)
datapointsx = np.array([[1,2],[3,1],[1,1],[5,6],[3,9],[10,2]], dtype=np.float64)
datapointsy = np.array([[0],[0],[0],[1],[1],[1]], dtype=np.float64)
print("x : ",datapointsx.shape,"y: ",datapointsy.shape)

#m :  (1, 2) b:  (1, 1)
m = np.array([[1,2]],dtype=np.float64)
b = np.array([[1]],dtype=np.float64)
print("m : ",m.shape,"b: ",b.shape)

epochs = 10000
alpha = 0.03

n = len(datapointsx)

for _ in range(epochs):
    ytrue = datapointsy
    ypred = func(m, b, datapointsx)

    err = berror(ytrue, ypred.T)

    dm = np.matmul((ypred - ytrue.T),datapointsx)/n
    db = np.sum(ypred - ytrue.T)/n

    m -= alpha * dm
    b -= alpha * db

    if _ % 10 == 0:
        print("At epoch {} error is {}".format(_, round(err, 4)))
