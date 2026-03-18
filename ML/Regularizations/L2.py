import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def reludr(x):
    return (x>0).astype(float)

datapointsx=np.array([[1,2],[2,3],[3,2],[1,1],[4,5],[6,2],[3,3],[6,6]],dtype=np.float64)
datapointsy=np.array([[3],[5],[5],[2],[9],[8],[6],[12]],dtype=np.float64)

np.random.seed(35)
a1=np.random.rand(3,2)
b1=np.random.rand(3,1)
a2=np.random.rand(2,3)
b2=np.random.rand(2,1)
a3=np.random.rand(1,2)
b3=np.random.rand(1,1)

#param
alp=0.005
epochs=100
lambdaa=0.5
loss=[]
weights=[]

#iteration
for _ in range(epochs):

    #length
    n=len(datapointsx)
    er=0

    for idx in range(len(datapointsx)):
        #Setup
        x=datapointsx[idx].reshape(1,2)
        ytrue = datapointsy[idx].reshape(1, 1)

        #Forward-Prop
        z1=np.matmul(x,a1.T)+b1.T
        out1=relu(z1)

        z2=np.matmul(out1,a2.T)+b2.T
        out2=relu(z2)

        z3=np.matmul(out2,a3.T)+b3.T
        out3=relu(z3)

        #Backward-Prop
        dw3=(-2*(ytrue-out3)*reludr(z3)*out2)+(2*lambdaa*a3)
        dw2=np.matmul((np.matmul(-2*(ytrue-out3)*reludr(z3),a3)*reludr(z2)).T,out1)+(2*lambdaa*a2)
        dw1=np.matmul((np.matmul(-2*(ytrue-out3)*reludr(z3)*a3*reludr(z2),a2)*reludr(z1)).T,x)+(2*lambdaa*a1)

        db3=reludr(z3)*-2*(ytrue-out3)
        db2=(np.matmul(-2*(ytrue-out3)*reludr(z3),a3)*reludr(z2)).T
        db1=(np.matmul(-2*(ytrue-out3)*reludr(z3)*a3*reludr(z2),a2)*reludr(z1)).T

        #corrections
        a3-=alp*dw3
        a2-=alp*dw2
        a1-=alp*dw1

        b3-=alp*db3
        b2-=alp*db2
        b1-=alp*db1

        #update
        er+=(ytrue[0][0] - out3[0][0])**2

    if _%5==0:
        print("at epoch {} error is {}".format(_,round(er/n,2)))
        loss.append(er / n)
        weights.append((np.sum(a1) + np.sum(a2) + np.sum(a3)))

print(weights)