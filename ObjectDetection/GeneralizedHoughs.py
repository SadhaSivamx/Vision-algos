import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import LogNorm

# Initializing the Image
Size = 500
img = cv.imread("Src/B1.png", 0)
img = cv.resize(img, (Size, Size))

# Finding the Edges
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

# Finding the Angle
anglerad = np.arctan2(sobely, sobelx)
angledeg = np.rad2deg(anglerad)

# Edges for Centroid
edges = cv.Canny(img, 100, 200)


def getcentroid():
    M = cv.moments(edges)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return (cX, cY)


# Creating R table
centx, centy = getcentroid()
ycoords, xcoords = np.where(edges > 0)
rtable = defaultdict(list)

for y, x in zip(ycoords, xcoords):
    # shift from centroid to ypoint and xpoint
    dy = y - centy
    dx = x - centx

    # radial distance and Alpha calculations
    r = np.hypot(dx, dy)
    alpha = np.arctan2(dy, dx)
    phi = int(np.round(angledeg[y, x]))

    # update the Table
    rtable[phi].append((r, alpha))

# Search images
Searchimg = cv.imread("Src/B2.png", 0)
Searchimg = cv.resize(Searchimg, (Size, Size))

# Finding the Edges
searchx = cv.Sobel(Searchimg, cv.CV_64F, 1, 0, ksize=3)
searchy = cv.Sobel(Searchimg, cv.CV_64F, 0, 1, ksize=3)

# Angle in deg
anglerads = np.arctan2(searchy, searchx)
angledegs = np.rad2deg(anglerads)
scales = np.arange(0.25, 1.25, 0.1)

#Defining Scoreboard and voting
Scoreboard = np.zeros((len(scales), Size, Size), dtype=np.int32)
edg = cv.Canny(Searchimg, 100, 200)
ypts, xpts = np.where(edg > 0)

#voting
for y, x in zip(ypts, xpts):
    phi = int(np.round(angledegs[y, x]))

    if phi in rtable:
        for (r, alpha) in rtable[phi]:
            for sidx, scale in enumerate(scales):

                #moving inwards
                scaledr = r * scale
                Xc = int(np.round(x + (scaledr * -np.cos(alpha))))
                Yc = int(np.round(y + (scaledr * -np.sin(alpha))))

                # check if available
                if 0 <= Xc < Size and 0 <= Yc < Size:
                    Scoreboard[sidx, Yc, Xc] += 1

#for visualization
Best2D = np.max(Scoreboard, axis=0)
plt.imshow((Best2D + 1) ** 2, cmap="plasma", norm=LogNorm())
plt.axis("off")
plt.colorbar()
plt.title("Hough Accumulator")
plt.show()
