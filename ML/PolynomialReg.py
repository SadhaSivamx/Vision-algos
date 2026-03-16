import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

fig, ax = plt.subplots(figsize=(8, 6))
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('Out.avi', fourcc, 10.0, (800, 600))

def gety(a, b, c, d, x):
    return a * (x ** 3) + b * (x ** 2) + c * (x) + d

datapoints = np.array([[1.5, 4.65], [6.0, 4.93], [3.5, 6.0], [5.5, 5.6], [4.5, 6.70]])
x = datapoints[:, 0]
y = datapoints[:, 1]

a = 0.02
b = 0.12
c = 0.35
d = 0.24

epochs = 50000
alpha = 0.00001
n = len(datapoints)

for _ in range(epochs):
    error = y - gety(a, b, c, d, x)

    dw1 = np.sum(-2 * error * (x * x * x)) / n
    dw2 = np.sum(-2 * error * (x * x)) / n
    dw3 = np.sum(-2 * error * x) / n
    db = np.sum(-2 * error) / n

    a -= alpha * dw1
    b -= alpha * dw2
    c -= alpha * dw3
    d -= alpha * db


    mse = np.sum(error ** 2) / n
    if _ % 500 == 0:
        print("At ep {} MSE is {}".format(_, round(mse, 4)))
        xp = np.linspace(0, 6, 20)
        yp = gety(a, b, c, d, xp)
        ax.plot(xp, yp, c="b")
        ax.scatter(x, y, marker="x", c="r")
        ax.set_xlim((0, 10))
        ax.set_ylim((0, 10))
        ax.set_title(f"Fitting a Curve.. Epoch {_}")
        fig.canvas.draw()
        imgrgb = np.asarray(fig.canvas.buffer_rgba())
        imgbgr = cv.cvtColor(imgrgb, cv.COLOR_RGBA2BGR)
        out.write(imgbgr)
        ax.clear()

out.release()
plt.close(fig)
print("Donee...")