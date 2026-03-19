import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from Conv import Convolve

#Getmax
def MaxPool(img,ksize,stsize):
    hsize,w=img.shape
    size=((hsize-ksize)//stsize)+1
    res = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            iloc=i*stsize
            jloc=j*stsize
            patch = img[iloc: iloc + ksize, jloc: jloc + ksize]
            res[i, j] = np.max(patch)
    return res

img=cv.imread("Src/dog.png",0)
img=cv.resize(img,(150,150))
k=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
plt.imshow(MaxPool(Convolve(k,img),2,4))
plt.show()