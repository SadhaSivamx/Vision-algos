'''
Equation Consider :
Y = 2X^2 + 2X -10

Update Rule :
Y = Y - Alpha * DelY/DelX

DelY/DelX = (4*X)+2
'''

import numpy as np
import matplotlib.pyplot as plt

#Function
def func(x):
    return (2*(x**2))+(2*x)-10

def visualize(x,y,xp,yp):
    plt.plot(x, y, c="b")
    plt.scatter(xp, yp, c="r")
    plt.xlim((-3, 3))
    plt.ylim((-15, 20))
    plt.title("Error Curve..")
    plt.show()

#Error Curve
x=np.linspace(-10,10,50,endpoint=True)
y=func(x)

#Point of Interest
xp=2.2
yp=func(xp)

#iteration
ep=20
al=0.03

#show
visualize(x,y,xp,yp)

#update
for _ in range(1,ep):
    xp-=al*((4*xp)+2)
yp = func(xp)

#show
visualize(x,y,xp,yp)
