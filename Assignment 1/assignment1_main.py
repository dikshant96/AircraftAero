import ta_functions as ff
import ta_class as fc
import numpy as np
import matplotlib.pyplot as plt

#Generating NACA 4 digit airfoil
pos, yc = ff.gen_NACAfoil(100,0,0,9)

#Get camber of the airfoil
n = 41
x, y = ff.get_camber(pos,n)

#Get pansels from the discretisation
panels = []
for i in range(1,len(x)):
    panels.append(fc.Panel(x[i-1],y[i-1],x[i],y[i]))

#Get influence coefficients as A matrix LHS
A = ff.get_coeffMatrix(panels,n)

#Get RHS of the equation (free stream effect)
alfa = 5*(np.pi/180)
U = 10
RHS = ff.get_RHS(U,alfa,panels,n)

#Solve for gamma distribution
Gamma = np.linalg.solve(A,RHS)

dL = Gamma*U

dP = np.zeros((n-1,1))
Cp = np.zeros((n-1,1))
Cp2 = np.zeros((n-1,1))
ut = ff.post_process(U,alfa,Gamma,panels,n)
for i in range(n-1):
    dP[i] = U*Gamma[i]/panels[i].ds
    Cp[i] = dP[i]/(U)
    Cp2[i] = 1 - ((ut[i]**2)/(U**2))

xv = [panels[i].vpoint.x for i in range(n-1)]

#plt.scatter(xv, Cp)
plt.scatter(xv, Cp2)
plt.show()