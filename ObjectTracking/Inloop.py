import cv2 as cv
import numpy as np
import os
from yolomodel import getroi
import time

'''
Detection-based Tracking...

We assume that the movement of the object will be small
therefore, we add penalties as it moves away from its previous position.
This spatial penalty is calculated as:

r = x^2 + y^2

As the object moves back and forth, its scale may vary. 
To account for this, we apply scale variations to the Object of Interest 
and add a penalty to retain the current scale. 
This scale penalty is defined as:

s = (1 - sc)^2 (This squared term ensures the penalty remains positive).

Total Result Calculation: 

Result = result - r - s

The Drift Problem:  

A common issue in this approach is Template Drift, 
where the tracker gradually loses the target due to accumulated errors. 

To resolve this, we employ a Deep Learning model on an extended Region of Interest (ROI).
This allows us to periodically correct the template and re-center the tracker, 
optimizing both processing speed and tracking accuracy.
'''

#Files
files = sorted([f for f in os.listdir("Frames") if f.endswith(('.jpg', '.png'))])

#configfile
Configimg=cv.imread("Frames/frame_0000.jpg",0)
height,width=Configimg.shape

#Select Roi
x,y,w,h=cv.selectROI("Template", Configimg, fromCenter=False, showCrosshair=True)
side = int((w + h) / 2)
w, h = side, side
cv.destroyWindow("Template")

#Resizer
scales = np.linspace(0.75, 1.25, 10)

starttime = time.time()
print("Program Starts At : {}".format(starttime))

for idx, filename in enumerate(files):

    # Match
    Matchcol = cv.imread("Frames/" + filename)
    Matchimg = cv.cvtColor(Matchcol, cv.COLOR_BGR2GRAY)

    #Update with YOLO
    if idx % 5 == 0 and idx != 0:
        pad = 250

        #check for Problem..
        winx1 = max(0, x - pad)
        winy1 = max(0, y - pad)
        winx2 = min(width, (x + w) + pad)
        winy2 = min(height, (y + h) + pad)

        #ROI Selection...
        LocalSection = Matchcol[winy1:winy2, winx1:winx2]

        if LocalSection.size != 0:
            nlocal = getroi(LocalSection)

            if nlocal:
                lx, ly, lw, lh = nlocal
                x = winx1 + lx
                y = winy1 + ly
                w = lw
                h = lh

                Configimg = Matchimg
                continue

    x, y = max(0, x), max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)

    # Template
    TemplateMain = Configimg[y:y + h, x:x + w]
    if TemplateMain.size == 0:
        continue

    #init
    bestval=float("-inf")
    bestroi=(x,y,w,h)

    for sc in scales:
        #Scaling for variation in Distance from Camera
        nh,nw=int(h*sc),int(w*sc)
        if nw < 50 or nh < 50 or nw > width or nh > height:
            continue

        #resize and Match
        Template = cv.resize(TemplateMain, (nw, nh))
        result = cv.matchTemplate(Matchimg, Template, cv.TM_CCOEFF_NORMED)
        resheight,reswidth=result.shape

        #penalty for Distance
        ymap, xmap = np.ogrid[:resheight, :reswidth]
        distsq = (xmap - x) ** 2 + (ymap - y) ** 2

        #Penalty
        penalty = 1e-5
        scalepenality = 50.0

        finalsq = (result) - (distsq * penalty) - (scalepenality*abs(1-sc)**2)

        #Max Square
        _, mval , _, maxloc = cv.minMaxLoc(finalsq)

        if mval>bestval:
            bestval=mval
            bestroi=(maxloc[0],maxloc[1],nw,nh)

    #Update
    (x, y, w, h) = bestroi
    Configimg = Matchimg

    #Visualization
    viz = Matchcol.copy()
    cv.rectangle(viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(viz, f"Match: {bestval:.2f}", (x, y - 10),
    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv.imshow("Template Tracker", viz)

    #break
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

totaltime = time.time() - starttime
print(f"Template Tracking + DL took {totaltime/60:.4f} Mins")

cv.destroyAllWindows()