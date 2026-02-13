'''
Mean Shift Clustering with Spatial Features

We represent each pixel by its coordinates and its color: (Y, X, R, G, B).
This creates a 5D feature space where "closeness" depends on both where a pixel
is and what color it is.

We set a threshold (e.g., thresh = 30). This defines a hypersphere
around a data point. Only pixels within this radial distance contribute to the next move.

For every single pixel in the image, we look at all other pixels within the threshold.
We calculate the average position and color of these neighboring pixels

newcentroid = 1/N * Sum of pixels within that range

We "shift" the current point to this new mean.
This is an iterative process where the point "climbs" the
density gradient toward the peak (the mode). It stops if the shift is
tiny (less than 0.1 units) or after 10 iterations.

Once the point finds its local peak, the original pixel's color is
replaced with the color values (R, G, B) of that peak.

Because nearby pixels usually climb to the same peak, they end up with the same color.
This naturally groups the image into smooth, flattened color regions, preserving edges better than simple K-Means.
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

#Load-Image
img=cv.imread("lena.png")
size=150
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
img=cv.resize(img,(size,size))
ori=img.copy()

#Stacking Based on Color and Spatial
yrds, xrds = np.indices((size,size))
pixels = np.dstack((yrds, xrds, img)).reshape(-1, 5)

thresh=30
for px in pixels:
    centroid=px
    for _ in range(10):
        distances = np.linalg.norm(pixels - centroid, axis=1)
        mask = distances < thresh
        within = pixels[mask]
        if len(within) == 0:
            break
        ncentroid = np.mean(within, axis=0)
        if np.linalg.norm(ncentroid - centroid) < 0.1:
            break
        centroid = ncentroid
    img[px[0],px[1]]=[centroid[2],centroid[3],centroid[4]]

#stack & show
final=np.hstack([ori,img])
plt.imshow(final)
plt.show()
