import numpy as np
import matplotlib.pyplot as plt

xdata = np.array(list(range(1, 11)))
ydata = (xdata * 10) + 3


m1 = np.random.randn(100, 1) * 0.01
m2 = np.random.randn(20, 100) * 0.01
m3 = np.random.randn(1, 20) * 0.01

b1 = np.zeros((100, 1))
b2 = np.zeros((20, 1))
b3 = np.zeros((1, 1))

p = 0.1
alp = 0.00001
epochs = 1500

for epoch in range(epochs):
    epochloss = 0

    for dataidx in range(xdata.shape[0]):
        datax = xdata[dataidx].reshape((1, 1))
        datay = ydata[dataidx].reshape((1, 1))

        a1 = np.matmul(m1, datax) + b1
        mask1 = (np.random.rand(*a1.shape) > p) / (1.0 - p)
        a1drop = a1 * mask1

        a2 = np.matmul(m2, a1drop) + b2
        mask2 = (np.random.rand(*a2.shape) > p) / (1.0 - p)
        a2drop = a2 * mask2

        a3 = np.matmul(m3, a2drop) + b3

        epochloss += (datay[0, 0] - a3[0, 0]) ** 2

        e3 = -2 * (datay - a3)
        dw3 = np.dot(e3, a2drop.T)
        db3 = e3

        e2 = np.matmul(m3.T, e3)
        e2drop = e2 * mask2
        dw2 = np.dot(e2drop, a1drop.T)
        db2 = e2drop

        e1 = np.matmul(m2.T, e2drop)
        e1drop = e1 * mask1
        dw1 = np.dot(e1drop, datax.T)
        db1 = e1drop

        m1 -= alp * dw1
        b1 -= alp * db1
        m2 -= alp * dw2
        b2 -= alp * db2
        m3 -= alp * dw3
        b3 -= alp * db3

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {epochloss / 10:.4f}")

print("Training Complete...")
