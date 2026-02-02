import cv2 as cv
import numpy as np
import os

'''
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

Total Result = result - r - s
Problem : The template might Drift in a wrong direction..
'''

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

for i in os.listdir("Frames/"):
    # Match
    Matchcol = cv.imread("Frames/" + i)
    Matchimg = cv.cvtColor(Matchcol, cv.COLOR_BGR2GRAY)

    #init
    bestval=float("-inf")
    bestroi=(x,y,w,h)

    # Template
    TemplateMain = Configimg[y:y + h, x:x + w]

    for sc in scales:
        #Scaling for variation in Distance from Camera
        nh,nw=int(h*sc),int(w*sc)
        if nw < 10 or nh < 10 or nw > width or nh > height:
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

cv.destroyAllWindows()
