import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def softmax(out):
    expz = np.exp(out)
    return expz / np.sum(expz)

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0) * 1.0

img1 = cv.imread("Src/0.png", 0)
img1 = cv.resize(img1, (15, 15))

img2 = cv.imread("Src/3.png", 0)
img2 = cv.resize(img2, (15, 15))

img3 = cv.imread("Src/9.png", 0)
img3 = cv.resize(img3, (15, 15))

plt.imshow(img1,cmap="gray")
plt.show()

xdata = np.array([img1,img2,img3])
ydata = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
ldp = 15*15

m1 = np.random.randn(225, ldp) * np.sqrt(2.0 / ldp)
m2 = np.random.randn(50, 225) * np.sqrt(2.0 / 225)
m3 = np.random.randn(3, 50) * np.sqrt(2.0 / 50)

b1 = np.zeros((225, 1))
b2 = np.zeros((50, 1))
b3 = np.zeros((3, 1))

epochs = 500
alpha = 0.05

for epoch in range(epochs):
    totalloss = 0

    for idx in range(xdata.shape[0]):
        xtrue = np.ravel(xdata[idx]).reshape(-1, 1) / 255.0
        ytrue = ydata[idx].reshape(-1, 1)

        z1 = np.matmul(m1, xtrue) + b1
        out1 = relu(z1)

        z2 = np.matmul(m2, out1) + b2
        out2 = relu(z2)

        z3 = np.matmul(m3, out2) + b3
        probs = softmax(z3)

        loss_val = -np.log(probs[np.argmax(ytrue)] + 1e-15)
        totalloss += loss_val[0]

        E3 = probs - ytrue
        dw3 = np.matmul(E3, out2.T)
        db3 = E3

        E2 = np.matmul(m3.T, E3) * drelu(z2)
        dw2 = np.matmul(E2, out1.T)
        db2 = E2

        E1 = np.matmul(m2.T, E2) * drelu(z1)
        dw1 = np.matmul(E1, xtrue.T)
        db1 = E1

        for weight, grad in zip([m3, m2, m1, b3, b2, b1], [dw3, dw2, dw1, db3, db2, db1]):
            np.clip(grad, -1.0, 1.0, out=grad) 
            weight -= alpha * grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total Loss: {totalloss:.4f}")

def predict(img):
    x = np.ravel(img).reshape(-1, 1) / 255.0
    z1 = np.matmul(m1, x) + b1
    o1 = relu(z1)
    z2 = np.matmul(m2, o1) + b2
    o2 = relu(z2)
    z3 = np.matmul(m3, o2) + b3
    p = softmax(z3)
    return np.argmax(p)

print("\nFinal Predictions:")
classes = ["Digit 0", "Digit 3", "Digit 9"]
for i in range(3):
    res = predict(xdata[i])
    print(f"Input {classes[i]} -> Predicted Index: {res} ({classes[res]})")
