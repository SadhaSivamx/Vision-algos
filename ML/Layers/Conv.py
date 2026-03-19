import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def Convolve(k,img):
    #Size&Pad
    kernelsize = len(k)
    padreq = (kernelsize-1) // 2
    row, hgt = img.shape

    #OutputSize Est
    img = np.pad(img, padreq)
    sz = (row-kernelsize+2*padreq)+1
    res = np.zeros((sz, sz))
    kernel = np.ravel(k)

    #Convolve
    for i in range(sz):
        for j in range(sz):
            patch = np.ravel(img[i:i + kernelsize, j:j + kernelsize])
            cvresult = np.dot(patch, kernel)
            res[i][j] = cvresult
    return res

img=cv.imread("Src/dog.png",0)
img=cv.resize(img,(150,150))
k=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
plt.imshow(Convolve(k,img))
plt.show()