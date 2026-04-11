import numpy as np
def Conv(kernal, input, stride):
    kerfilters, kerdepth, kersize, _ = kernal.shape
    batch, depth, height, width = input.shape
    p = (kersize - 1) // 2
    outsize = ((height - kersize + 2 * p) // stride) + 1
    inputp = np.pad(input, ((0, 0), (0, 0), (p, p), (p, p)))
    memory = np.zeros((batch, kerfilters, outsize, outsize))
    for b in range(batch):
        for f in range(kerfilters):
            for h in range(outsize):
                for w in range(outsize):
                    i, j = h * stride, w * stride
                    imgpatch = np.ravel(inputp[b, :, i:i+kersize, j:j+kersize])
                    kernalunravelled = np.ravel(kernal[f])
                    memory[b, f, h, w] = np.dot(imgpatch, kernalunravelled)
    return memory

def Standardize(batch, gamma, beta):
    xmean = np.mean(batch, axis=(0, 2, 3), keepdims=True)
    xvar = np.var(batch, axis=(0, 2, 3), keepdims=True)
    xhat = (batch - xmean) / (np.sqrt(xvar + 1e-15))
    return (gamma * xhat) + beta

def ReLU(x):
    return np.maximum(0, x)

input = np.random.rand(5, 1, 50, 50)
k1 = np.random.rand(10, 1, 3, 3) * 0.1
gamma1 = np.ones((1, 10, 1, 1))
beta1 = np.zeros((1, 10, 1, 1))

k2 = np.random.rand(10, 10, 3, 3) * 0.1
gamma2 = np.ones((1, 10, 1, 1))
beta2 = np.zeros((1, 10, 1, 1))

kp = np.random.rand(10, 1, 1, 1) * 0.1
gammap = np.ones((1, 10, 1, 1))
betap = np.zeros((1, 10, 1, 1))


print("Input shape:", input.shape)

out1 = Conv(k1, input, 1)
norm1 = Standardize(out1, gamma1, beta1)
relu1 = ReLU(norm1)

out2 = Conv(k2, relu1, 1)
norm2 = Standardize(out2, gamma2, beta2)

shortcut = Conv(kp, input, 1)
norms= Standardize(shortcut, gammap, betap)
final = ReLU(norm2 + norms)

print("Final Output shape:", final.shape)

def GetDerivatives(batch,gamma,error):
    xmean = np.mean(batch, axis=(0,2,3), keepdims=True)
    xvar = np.var(batch, axis=(0,2,3), keepdims=True)
    std = np.sqrt(xvar + 1e-5)

    xhat = (batch - xmean) / std

    dw = np.sum(xhat * error, axis=(0,2,3), keepdims=True)
    db = np.sum(error, axis=(0,2,3), keepdims=True)

    dxhat = error * gamma

    n = batch.shape[0] * batch.shape[2] * batch.shape[3]
    sumdx = np.sum(dxhat, axis=(0,2,3), keepdims=True)
    sumdxhat = np.sum(dxhat * xhat, axis=(0,2,3), keepdims=True)
    dimg = (1 / (n * std)) * (n * dxhat - sumdx - xhat * sumdxhat)
    return dw, db, dimg

def RevConv(kernal, input, error, stride):
    batch, inpdepth, inpheight, inpwidth = input.shape
    kerfilters, kerdepth, kerheight, kerwidth = kernal.shape
    batcherr, errdepth, errheight, errwidth = error.shape
    requiredpad = kerheight // 2
    pinp = np.pad(input, ((0, 0), (0, 0), (requiredpad, requiredpad), (requiredpad, requiredpad)), mode='constant')
    dimg = np.zeros_like(input)
    dimg = np.pad(dimg, ((0, 0), (0, 0), (requiredpad, requiredpad), (requiredpad, requiredpad)), mode='constant')
    dk = np.zeros_like(kernal)

    for b in range(batch):
        for f in range(kerfilters):
            for i in range(0, errheight):
                for j in range(0, errwidth):
                    istart, jstart = i * stride, j * stride
                    imgpatch = pinp[b, :, istart:istart + kerheight, jstart:jstart + kerwidth]
                    dimg[b, :, istart:istart + kerheight, jstart:jstart + kerwidth] += kernal[f] * error[b, f, i, j]
                    errval = error[b, f, i, j]
                    dk[f] += imgpatch * errval
    return dk, dimg[:, :, requiredpad:requiredpad + inpheight, requiredpad:requiredpad + inpwidth]

def reludr(inp):
    return (inp > 0).astype(float)

errorf = np.random.rand(*final.shape) * reludr(norm2 + norms)
dgp, dbp, dimgshort = GetDerivatives(shortcut, gammap, errorf)
dk, dimg = RevConv(kp, input, dimgshort, 1)

dg2, db2, dimgs2 = GetDerivatives(out2, gamma2, errorf)
dk2, dimg2 = RevConv(k2, relu1, dimgs2, 1)

drl2 = dimg2 * reludr(norm1)

dg1, db1, dimgs1 = GetDerivatives(out1, gamma1, drl2)
dk1, dimg1 = RevConv(k1, input, dimgs1, 1)

#Updates

alpha=0.05

gammap-=alpha*dgp
betap-=alpha*dbp
kp-=alpha*dk

gamma2-=alpha*dg2
beta2-=alpha*db2
k2-=alpha*dk2

gamma1-=alpha*dg1
beta1-=alpha*db1
k1-=alpha*dk1





