'''
K-Means Clustering with Spatial Features

Step-1
We represent each pixel not just by color (R, G, B),
but by its position and color (Y, X, R, G, B).
This ensures clusters are spatially coherent
(pixels near each other are more likely to stay together).

Step-2
We pick 10 initial centroids. To ensure they aren't all clumped in one corner,
we only accept a new centroid if it is a specific radial distance (300 units)
away from existing ones.

Step-3
We map every pixel to its nearest centroid using the Euclidean distance formula:
dist = sqrt{(y1-y2)^2 + (x1-x2)^2 + (r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2

Step-4
We recalculate the center of each cluster by taking the mean
of all pixels assigned to it.

Step-5
We repeat steps 3 and 4 until the centroids stabilize (move less than 1 unit)
or we hit our 50-epoch limit.

Step-6
Each pixel in the image is repainted with the color of its assigned centroid
creating a "posterized" or segmented effect.
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

#Load-Image
img=cv.imread("img.png")
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
img=cv.resize(img,(255,255))
ori=img.copy()

#Stacking Based on Color and Spatial
yrds, xrds = np.indices((255, 255))
pixels = np.dstack((yrds, xrds, img)).reshape(-1, 5)

#KNN++ Choicer
idx = np.random.randint(0, len(pixels))
centroids = [pixels[idx]]
while len(centroids)<10:
    Newidx = np.random.randint(0, len(pixels))
    Ncentroid = pixels[Newidx]
    Vals=np.linalg.norm(np.array(centroids)-Ncentroid,axis=1)
    Score=np.all(Vals<300)
    if Score:
        centroids.append(Ncentroid)

epochs=50

for _ in range(epochs):
    CentroidstoPixels={}
    for pixel in pixels:
        Score = np.linalg.norm(np.array(centroids) - pixel, axis=1)
        closestidx = np.argmin(Score)
        if tuple(centroids[closestidx]) in CentroidstoPixels:
            CentroidstoPixels[tuple(centroids[closestidx])].append(pixel)
        else:
            CentroidstoPixels[tuple(centroids[closestidx])]=[pixel]

    #reassign
    ncent = []
    for pts in CentroidstoPixels.values():
        parr = np.array(pts)
        new_centroid = parr.mean(axis=0)
        ncent.append(new_centroid)

    #if the change in centroid is within a thresh no need to change further
    if np.linalg.norm(np.array(ncent) - np.array(centroids))<1:
        break
    centroids = ncent

for k,v in CentroidstoPixels.items():
    for imgpt in v:
        img[int(imgpt[0]), int(imgpt[1])] = [k[2], k[3], k[4]]

final=np.hstack([ori,img])
plt.imshow(final)
plt.show()
