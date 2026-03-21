import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

'''
Sigmoid(x) = 1 / (1 + e^(-x))
np.clip limits bounds to [-1000, 1000] to prevent np.exp overflow
'''

def Sigmoid(res):
    res = np.clip(res, -1000, 1000)
    return 1 / (1 + np.exp(-res))


'''
ReLU(x) = max(0, x)
'''

def relu(x):
    return np.maximum(0, x)


'''
dReLU(x)/dx = 1.0 if x > 0 else 0.0
'''

def drelu(x):
    return (x > 0) * 1.0


'''
Backward pass for Convolution
dW = (Loss / kernal)
dimg = (Loss / orig_img)
'''

def Convbackward(dz, orig_img, kernal):
    print(dz.shape, orig_img.shape, kernal.shape)
    filters, kdepth, kheight, kwidth = kernal.shape
    _, out_h, out_w = dz.shape

    dW = np.zeros_like(kernal)
    dimgp = np.zeros((kdepth, orig_img.shape[1] + kheight - 1, orig_img.shape[2] + kwidth - 1))

    padreq = (kheight - 1) // 2
    imgp = np.pad(orig_img, ((0, 0), (padreq, padreq), (padreq, padreq)))

    for f in range(filters):
        for x in range(out_h):
            for y in range(out_w):
                patch = imgp[:, x:x + kheight, y:y + kheight]
                dW[f] += patch * dz[f, x, y]
                dimgp[:, x:x + kheight, y:y + kheight] += kernal[f] * dz[f, x, y]

    if padreq > 0:
        dimg = dimgp[:, padreq:-padreq, padreq:-padreq]
    else:
        dimg = dimgp

    return dW, dimg


'''
Reverse Pooling for Backpropagation
Routes dz to argmax(patch)
'''

def RevPool(dz, input, ksize, stride):
    d, w, h = input.shape
    od, oh, ow = dz.shape

    mem = np.zeros(input.shape)
    for dp in range(od):
        for r in range(oh):
            for c in range(ow):
                i = r * stride
                j = c * stride
                patch = input[dp, i:i + ksize, j:j + ksize]
                maximum = np.max(patch)
                mask = (patch == maximum)
                mem[dp, i:i + ksize, j:j + ksize] += (mask * dz[dp, r, c])
    return mem


'''
Forward pass for Convolution
res = img * kernal
'''

def convolveme(kernal, img):
    filters, kdepth, kheight, kwidth = kernal.shape
    print(kernal.shape)
    padreq = (kheight - 1) // 2

    _, imgw, imgh = img.shape
    img = np.pad(img, ((0, 0), (padreq, padreq), (padreq, padreq)))
    sz = (imgw - kheight + 2 * padreq) + 1

    res = np.zeros((filters, sz, sz))

    for f in range(filters):
        ckernal = np.ravel(kernal[f])
        for x in range(sz):
            for y in range(sz):
                patch = np.ravel(img[:, x:x + kheight, y:y + kheight])
                cvresult = np.dot(patch, ckernal)
                res[f][x][y] = cvresult
    return res


'''
Forward pass for Max Pooling
res = max(patch)
'''

def Maxpoolme(img, ksize, stride):
    d, h, w = img.shape
    size = ((h - ksize) // stride) + 1

    res = np.zeros((d, size, size))

    for dp in range(d):
        for r in range(size):
            for c in range(size):
                i = r * stride
                j = c * stride
                patch = img[dp, i: i + ksize, j: j + ksize]
                res[dp, r, c] = np.max(patch)
    return res


# fortraining..
img1 = cv.imread("Src/dog.png", 0)
img1 = cv.resize(img1, (28, 28))
img2 = cv.imread("Src/cat.png", 0)
img2 = cv.resize(img2, (28, 28))
# fortesting..
img3 = cv.imread("Src/img.png", 0)
img3 = cv.resize(img3, (28, 28))

# DataSetup
images = np.array([img1, img2]) / 255.0
result = np.array([[1], [0]])

# training-params
epochs = 30
alpha = 0.05

'''
Kernal Initialization
Shape: (Filters, Depth, Height, Width)
'''
Out0d = 1
kernal1 = np.random.randn(3, Out0d, 3, 3) * np.sqrt(2 / (3 * Out0d * 3 * 3))
kernal3 = np.random.randn(1, 3, 3, 3) * np.sqrt(2 / 27)

# Flatterning
flattened = 1 * 7 * 7

# Weights Initialization
ANN5 = np.random.randn(100, flattened) * np.sqrt(2 / 4900)
ANN6 = np.random.randn(20, 100) * np.sqrt(2 / 2000)
ANN7 = np.random.randn(1, 20) * np.sqrt(2 / 20)

# Bias Initialization
b5 = np.zeros((100, 1))
b6 = np.zeros((20, 1))
b7 = np.zeros((1, 1))

# epochs loop
for _ in range(epochs):
    print(f"\n--- Starting epoch {_ + 1} ---")
    for i in range(images.shape[0]):
        ''' 
        FORWARD PASS
        '''
        Out0 = images[i].reshape(1, 28, 28)
        ytrue = result[i]

        # Output shape : (3,28,28)
        r1 = convolveme(kernal1, Out0)
        Out1 = r1

        # Output shape : (3,14,14)
        r2 = Maxpoolme(r1, 2, 2)
        Out2 = r2

        # Output shape : (1,14,14)
        r3 = convolveme(kernal3, Out2)
        Out3 = r3

        # Output shape : (1,7,7)
        r4 = Maxpoolme(r3, 2, 2)
        # Output shape : (1*7*7)
        Out4 = np.ravel(r4).reshape(-1, 1)

        # Output shape : (100,1)
        r5 = np.matmul(ANN5, Out4) + b5
        Out5 = relu(r5)

        # Output shape : (20,1)
        r6 = np.matmul(ANN6, Out5) + b6
        Out6 = relu(r6)

        # Output shape : (1,1)
        r7 = np.matmul(ANN7, Out6) + b7
        r8 = Sigmoid(r7)

        print(f"Image {i + 1} | Target: {ytrue[0]} | Prediction: {r8[0][0]:.4f}")

        ''' 
        BACKWARD PASS
        Notation: d(x) = ( ∂Loss / ∂x )
        '''

        '''
        LAYER 7
        dz7 = (Loss / r7) = (r8 - ytrue)
        '''
        dz7 = (r8 - ytrue)

        '''
        (Loss / ANN7) = (Loss / r7) * (r7 / ANN7)
        (r7 / ANN7) = Out6
        '''
        ANN7 -= alpha * np.dot(dz7, Out6.T)
        b7 -= alpha * dz7

        '''
        LAYER 6
        dz6 = (Loss / r6) = (Loss / Out6) * (Out6 / r6)
        (Loss / Out6) = np.dot(ANN7.T, dz7)
        (Out6 / r6) = drelu(r6)
        '''
        dz6 = np.dot(ANN7.T, dz7) * drelu(r6)

        '''
        (Loss / ANN6) = (Loss / r6) * (r6 / ANN6)
        (r6 / ANN6) = Out5
        '''
        ANN6 -= alpha * np.dot(dz6, Out5.T)
        b6 -= alpha * dz6

        '''
        LAYER 5
        dz5 = (Loss / r5) = (Loss / Out5) * (Out5 / r5)
        (Loss / Out5) = np.dot(ANN6.T, dz6)
        (Out5 / r5) = drelu(r5)
        '''
        dz5 = np.dot(ANN6.T, dz6) * drelu(r5)

        '''
        (Loss / ANN5) = (Loss / r5) * (r5 / ANN5)
        (r5 / ANN5) = Out4
        '''
        ANN5 -= alpha * np.dot(dz5, Out4.T)
        b5 -= alpha * dz5

        ''' 
        dz4_flat = (Loss / Out4)
        '''
        dz4_flat = np.dot(ANN5.T, dz5)

        '''
        Reshape (1, 7, 7)
        '''
        dz4 = dz4_flat.reshape(r4.shape)

        '''
        dz3 = (Loss / r3)
        '''
        dz3 = RevPool(dz4, r3, ksize=2, stride=2)

        '''
        dw2 = (Loss / kernal3)
        dimg2 = (Loss / Out2)
        '''
        dw2, dimg2 = Convbackward(dz3, r2, kernal3)
        kernal3 -= dw2 * alpha

        '''
        dz2 = (Loss / r1)
        '''
        dz2 = RevPool(dimg2, r1, ksize=2, stride=2)

        '''
        dw1 = (Loss / kernal1)
        dimg1 = (Loss / Out0)
        '''
        dw1, dimg1 = Convbackward(dz2, Out0, kernal1)
        kernal1 -= dw1 * alpha

    print(f"Ending epoch {_ + 1}")


def Predict(image):
    Out0 = image.reshape(1, 28, 28)

    r1 = convolveme(kernal1, Out0)
    Out1 = r1

    r2 = Maxpoolme(r1, 2, 2)
    Out2 = r2

    r3 = convolveme(kernal3, Out2)
    Out3 = r3

    r4 = Maxpoolme(r3, 2, 2)
    Out4 = np.ravel(r4).reshape(-1, 1)

    r5 = np.matmul(ANN5, Out4) + b5
    Out5 = relu(r5)

    r6 = np.matmul(ANN6, Out5) + b6
    Out6 = relu(r6)

    r7 = np.matmul(ANN7, Out6) + b7
    r8 = Sigmoid(r7)

    if r8 < 0.5:
        print("its an Cat... conf {}".format(r8))
    else:
        print("its an Dog... conf {}".format(r8))


# Predictions...
Predict(img1)
Predict(img2)
Predict(img3)
