import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

m=1.0
b=0.4

fig, ax = plt.subplots(figsize=(8, 6))
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('Out.avi', fourcc, 10.0, (800, 600))

datapoints=np.array([[2.5,5.65],[3.3,4.93],[6,2.5],[7,1.6],[8.0,0.70]])
x=datapoints[:,0]
y=datapoints[:,1]

epochs=1000
alpha=0.055

for _ in range(1,epochs+1):
    n=len(datapoints)
    tx=datapoints[:,0]
    ty=datapoints[:,1]
    pred=((m*tx)+b)
    error=(ty-pred)
    dm=np.sum((-2*tx*error))/n
    db=np.sum((-2*error))/n
    m-=(alpha*dm)
    b-=(alpha*db)
    mse = np.sum(error ** 2) / n
    print("At epoch {} Error is {}".format(_, round(mse, 4)))

    xp = np.linspace(0, 10, 20)
    yp = (m * xp) + b
    ax.plot(xp, yp, c="r")
    ax.scatter(x, y, marker="x", c="b")
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 10))
    ax.set_title(f"Fitting a line.. Epoch {_}")
    fig.canvas.draw()
    imgrgb = np.asarray(fig.canvas.buffer_rgba())
    imgbgr = cv.cvtColor(imgrgb, cv.COLOR_RGBA2BGR)
    out.write(imgbgr)
    ax.clear()

out.release()
plt.close(fig)
print("Doneee....")