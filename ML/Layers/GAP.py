import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread("Src/dog.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (80, 80))
imgstack = np.transpose(img, (2, 0, 1))

def AvgPooling(input,kernalsize,stride):
    channnels,height,width=input.shape
    outputshape=((height-kernalsize)//stride)+1
    memory=np.zeros((channnels,outputshape,outputshape))
    for dp in range(channnels):
        for ht in range(outputshape):
            for wt in range(outputshape):
                rval=ht*stride
                cval=wt*stride
                imgpatch=input[dp,rval:rval+kernalsize,cval:cval+kernalsize]
                memory[dp,ht,wt]=np.average(imgpatch)
    return memory

def ReverseAvgPooling(error,input,kernalsize,stride):
    channnels, height, width = input.shape
    outputshape=((height-kernalsize)//stride)+1
    memory=np.zeros_like(input)
    for dp in range(channnels):
        for ht in range(outputshape):
            for wt in range(outputshape):
                rval=ht*stride
                cval=wt*stride
                memory[dp,rval:rval+kernalsize,cval:cval+kernalsize]+=1/(kernalsize**2)*error[dp,ht,wt]
    return memory

def GlobalAvgPooling(input):
    channnels,height,width=input.shape
    memory=np.zeros((channnels,1,1))
    for dp in range(channnels):
        val=np.average(input[dp,:,:])
        memory[dp]=val
    return memory

def ReverseGlobalAvgPooling(Error,input):
    channels,height,width=input.shape
    memory=np.zeros_like(input)
    for dp in range(channels):
        memory[dp,:,:]+=(1/(height*width))*Error[dp]
    return memory

out=AvgPooling(imgstack,2,2)
Out = np.transpose(out, (1, 2, 0))
Out = Out.astype(np.uint8)
plt.imshow(Out)
plt.show()
