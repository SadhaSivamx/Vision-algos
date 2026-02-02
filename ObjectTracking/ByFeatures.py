import cv2 as cv
import numpy as np
import os

''' 
the template is matched over a map generated from SIFT feature classifications.
Specifically, if a detected feature's descriptor is found to be similar 
to the object's descriptor library, we assign it a value of +1
 
otherwise, it is assigned a value of -1.
By convolving a fixed-size grid (the template) over this map and 
summing the underlying values, the result reaches a maximum in regions 
with a high density of positive matches. 

This peak effectively defines the second Region of Interest (ROI), 
localizing the target based on feature-space evidence rather 
than raw pixel intensities. 
'''

Base="Frames/"

# configfile
Configimg = cv.imread(Base+"frame_0000.jpg", 0)
height, width = Configimg.shape
print("Starting...")
# Select Roi
x, y, w, h = cv.selectROI("Template", Configimg, fromCenter=False, showCrosshair=True)
# side = int((w + h) / 2)
# w, h = side, side
cv.destroyWindow("Template")

# sift
sift = cv.SIFT_create()

def Seperate(img):
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # seperation
    obj = []
    bg = []

    # obj|Bg
    for i, kp in enumerate(keypoints):
        kpx, kpy = kp.pt
        if x <= kpx <= (x + w) and y <= kpy <= (y + h):
            obj.append(descriptors[i])
        else:
            bg.append(descriptors[i])

    # Desriptors
    obj = np.array(obj, dtype=np.float32)
    bg = np.array(bg, dtype=np.float32)

    return obj, bg


# FlannMatcher Setup
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=100)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Scale factors for search
scales = np.linspace(0.75, 1.25, 7)

for i in sorted(os.listdir(Base)):
    # secondframe
    Seccol = cv.imread(Base + i)
    if Seccol is None: continue

    obj, bg = Seperate(Configimg)

    Secimg = cv.cvtColor(Seccol, cv.COLOR_BGR2GRAY)
    seckeys, secdesc = sift.detectAndCompute(Secimg, None)

    # Score and Match
    zeroimg = np.zeros((height, width), dtype=np.float32)

    # Match each Descriptor
    if secdesc is not None and obj is not None and len(obj) > 0:
        for i in range(len(secdesc)):
            curr = secdesc[i].reshape(1, -1)

            # Distances to Nearest Neighbors in both libraries
            distobj = flann.match(curr, obj)[0].distance
            distbg = flann.match(curr, bg)[0].distance

            kpx, kpy = map(int, seckeys[i].pt)
            # Boundary check for safety
            if 0 <= kpy < height and 0 <= kpx < width:
                if distobj < distbg:
                    zeroimg[kpy, kpx] = 1.0
                else:
                    zeroimg[kpy, kpx] = -1.0

        # Apply Smoothing for more robust match
        zeroimg = cv.GaussianBlur(zeroimg, (15, 15), 0)

        bestval = -float('inf')
        bestroi = (x, y, w, h)

        # Iterate through scales to find best match W,H
        for sc in scales:
            nw, nh = int(w * sc), int(h * sc)
            if nw < 50 or nh < 50 or nw > width or nh > height:
                continue

            # Matchme with updated scale template
            template = np.ones((nh, nw), dtype=np.float32)
            res = cv.matchTemplate(zeroimg, template, cv.TM_CCORR_NORMED)

            _, mval, _, mloc = cv.minMaxLoc(res)

            distpenalty = 1e-3 * ((mloc[0] - x) ** 2 + (mloc[1] - y) ** 2)
            currentscore = mval - distpenalty - 1e-5*((1-sc)**2)

            if currentscore > bestval:
                bestval = currentscore
                bestroi = (mloc[0], mloc[1], nw, nh)

        # Change state to best found Scale and Location
        (x, y, w, h) = bestroi
        topleft = (x, y)
        bottomright = (x + w, y + h)
        maxval = bestval

        # Draw
        cv.rectangle(Seccol, topleft, bottomright, (0, 255, 0), 3)
        cv.putText(Seccol, f"Match: {maxval:.2f}", (topleft[0], topleft[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Change frame reference
        Configimg = Secimg

        # Show
        cv.imshow("SIFT-Guided Detection", Seccol)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
