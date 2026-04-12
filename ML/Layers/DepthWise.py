import numpy as np
import matplotlib.pyplot as plt

def Conv(kernal,image):
    ksize,ksize=kernal.shape
    imgH,imgW=image.shape
    P=(ksize-1)//2
    imgpadded = np.pad(image, ((P,P), (P,P)))
    outsize = (imgH-ksize+(2*P))+1
    memory = np.zeros((outsize,outsize))
    for h in range(outsize):
        for w in range(outsize):
            imgpatch=np.ravel(imgpadded[h:h+ksize,w:w+ksize])
            kernalunr=np.ravel(kernal)
            memory[h,w]=np.dot(kernalunr,imgpatch)
    return memory

def DwiseConv(batch,kernal):
    bt,height,width=batch.shape
    filters,ksize,ksize=kernal.shape
    mem=np.zeros((bt,height,width))
    for f in range(filters):
        mem[f]=Conv(kernal[f],batch[f])
    return mem

def Conv1x1(image,kernal):
    channels,height,width=image.shape
    filters,channels,ksize,ksize=kernal.shape
    memory=np.zeros((filters,height,width))
    for f in range(filters):
        for h in range(height):
            for w in range(width):
                img=np.ravel(image[:,h,w])
                ker=np.ravel(kernal[f])
                memory[f,h,w]=np.dot(img,ker)
    return memory

def RevConv1x1(input,kernal,error):
    inpchannels,inpheight,inpwidth=input.shape
    errfilters,errheight,errwidth=error.shape
    kerfilters,kerchannels,kerheight,kerwidth=kernal.shape
    dk=np.zeros(kernal.shape)
    dimg=np.zeros(input.shape)
    for f in range(kerfilters):
        for h in range(inpheight):
            for w in range(inpwidth):
                dk[f,:,0,0] += input[:,h,w] * error[f,h,w]
                dimg[:,h,w]+=kernal[f,:,0,0] * error[f,h,w]
    return dk,dimg

def RevConv(kernal,image,error):
    ksize, ksize = kernal.shape
    imgH, imgW = image.shape
    P = (ksize - 1) // 2
    imgpadded = np.pad(image, ((P, P), (P, P)))
    outsize = (imgH - ksize + (2 * P)) + 1
    dk = np.zeros(kernal.shape)
    dimg = np.zeros(image.shape)
    dimgp = np.pad(dimg, ((P, P), (P, P)))
    for h in range(outsize):
        for w in range(outsize):
            imgpatch=imgpadded[h:h+ksize,w:w+ksize]
            dk+=imgpatch*error[h,w]
            dimgp[h:h+ksize,w:w+ksize]+=kernal*error[h,w]
    return dk,dimgp[P:P+imgH,P:P+imgW]

def RevDwiseConv(input,kernal,error):
    filters, ksize, ksize = kernal.shape
    dkn=np.zeros(kernal.shape)
    dimgn=np.zeros(input.shape)
    for f in range(filters):
       kk,img=RevConv(kernal[f],input[f],error[f])
       dkn[f]+=kk
       dimgn[f]+=img
    return dkn,dimgn

#forwardprop
input=np.random.rand(32,150,150)
kernal1=np.random.rand(32,3,3)
kernal2=np.random.rand(64,32,1,1)
Out1=DwiseConv(input,kernal1)
Out2=Conv1x1(Out1,kernal2)

#backprop
error=np.random.rand(*Out2.shape)
dk2,dimg2=RevConv1x1(Out1,kernal2,error)
dk1,dimg1=RevDwiseConv(input,kernal1,dimg2)
print(dk1.shape,dimg1.shape)
