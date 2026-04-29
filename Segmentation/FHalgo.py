import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from collections import defaultdict
from Ds import Disjointset

imgfile = "img.png"
size = (250, 250)

realimg = cv.imread(imgfile)
realimg = cv.resize(realimg, size)
realimg = cv.cvtColor(realimg, cv.COLOR_BGR2RGB)
img = cv.cvtColor(realimg, cv.COLOR_RGB2GRAY)

n, m = img.shape

def findrect(arr):
    xcoords = [p[1] for p in arr]
    ycoords = [p[0] for p in arr]

    tl = (min(xcoords), min(ycoords))
    br = (max(xcoords), max(ycoords))
    return tl, br


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")

Connections = []
for i in range(n):
    for j in range(m):
        current_val = int(img[i, j])
        if j + 1 < m:
            weight = abs(current_val - int(img[i, j + 1]))
            Connections.append((weight, (i, j), (i, j + 1)))
        if i + 1 < n:
            weight = abs(current_val - int(img[i + 1, j]))
            Connections.append((weight, (i, j), (i + 1, j)))
Connections.sort(key=lambda x: x[0])

k = 1000
ds = Disjointset(n, m, k)
for Conc in Connections:
    ds.merge(*Conc)

memory = np.zeros((n, m))
colormap = {}
rectangle = defaultdict(list)
ccolor = 1

for i in range(n):
    for j in range(m):
        root = ds.findparent(i, j)

        if root not in colormap:
            colormap[root] = ccolor
            ccolor += 1

        memory[i, j] = colormap[root]
        rectangle[root].append([i, j])

print(f"Found {ccolor - 1} distinct segments.")

memory = (memory / np.max(memory)) * 255

plt.subplot(1, 2, 2)
plt.title(f"Segmented (k={k})")
plt.imshow(memory, cmap="plasma")
plt.tight_layout()
plt.show()

for r in rectangle:
    pt1, pt2 = findrect(rectangle[r])
    w = pt2[0] - pt1[0]
    h = pt2[1] - pt1[1]

    if w * h >= 1000:
        cv.rectangle(realimg, pt1, pt2, color=(255, 0, 0), thickness=1)

plt.figure(figsize=(6, 6))
plt.title("Bounding Boxes")
plt.imshow(realimg)
plt.show()
