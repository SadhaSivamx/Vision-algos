import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#Func
def Func(X,Y):
    return X**2 + Y**2

def Visualize(X,Y,Z,Xp,Yp,Zp,idx):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma_r',alpha=0.7)
    ax.scatter(Xp, Yp, Zp, c="r",s=50)
    ax.set_title('Elliptic Paraboloid: Z = X**2 + Y**2')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(elev=45, azim=70)
    plt.savefig('img{}.png'.format(idx))
    plt.clf()

#Points
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = Func(X,Y)

#Visualize & Save
Xp=-4
Yp=-3.5
Zp=Func(Xp,Yp)
Visualize(X,Y,Z,Xp,Yp,Zp,0)

#Iteration
ep=20
al=0.03

#Update
for _ in range(1,ep):
    Xp-=al*(2*Xp)
    Yp-=al*(2*Yp)
    Zp = Func(Xp,Yp)
    Visualize(X, Y, Z, Xp, Yp, Zp, _)
print("Done...")




