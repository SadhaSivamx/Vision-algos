import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

fig, ax = plt.subplots(figsize=(8, 6),dpi=100)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('Out.avi', fourcc, 10.0, (800, 600))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def func(m, b, x):
    return sigmoid(m * x + b)


def berror(ytrue, ypred):
    n = len(ytrue)
    ypred = np.clip(ypred, 1e-15, 1 - (1e-15))
    error = ytrue * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred)
    return -np.sum(error) / n


datapointsx = np.array([-5, -3, 1, 3, 5, 6, 8, 10, 12, 13], dtype=np.float64)
datapointsy = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float64)

m = 0.03
b = 0.5

epochs = np.random.randint()
alpha = 0.05


n = len(datapointsx)

for _ in range(epochs):
    ytrue = datapointsy
    ypred = func(m, b, datapointsx)

    err = berror(ytrue, ypred)

    dm = np.sum((ypred - ytrue) * datapointsx) / n
    db = np.sum(ypred - ytrue) / n

    m -= alpha * dm
    b -= alpha * db

    if _ % 10 == 0:
        print("At epoch {} error is {}".format(_, round(err, 4)))
        xp = np.linspace(-5, 15, 50)
        yp = func(m, b, xp)

        ax.plot(xp, yp, c="r")
        ax.scatter(datapointsx, datapointsy, marker="x", c="b")
        ax.set_title(f"Fitting an S-Curve.. Epoch {_}")

        fig.canvas.draw()
        imgrgb = np.asarray(fig.canvas.buffer_rgba())
        imgbgr = cv.cvtColor(imgrgb, cv.COLOR_RGBA2BGR)

        imgbgr = cv.resize(imgbgr, (800, 600))
        out.write(imgbgr)
        ax.clear()

out.release()
plt.close(fig)
print("Done.. Video Saved!")