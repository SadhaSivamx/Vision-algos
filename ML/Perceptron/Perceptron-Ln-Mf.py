import numpy as np

#points
datapointsx=np.array([[1,2],[3,5],[5,1],[2,3],[1,1]],dtype=np.float64)
datapointsy=np.array([2*x+1.5*y for (x,y) in datapointsx],dtype=np.float64)

#weights&bias
m=np.array([[1,1]],dtype=np.float64)
b=np.array([[1]],dtype=np.float64)

#iteration
epochs=100
alpha=0.05

for epoch in range(epochs):
    n=len(datapointsx)
    pred=np.sum(m*datapointsx,axis=1)+b

    #update
    error=datapointsy-pred
    m-=alpha*-2*(np.matmul(error,datapointsx))/n
    b-=alpha*(np.sum(-2*error)/n)

    #error
    mse=np.sum(error**2)/n
    print("At ep {} er is {}".format(epoch,round(mse,2)))

print("Done...")