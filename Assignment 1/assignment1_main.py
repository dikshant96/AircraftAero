import ta_functions as ff
import ta_class as fc
import numpy as np
import matplotlib.pyplot as plt

#Generating NACA 4 digit airfoil
pos, yc = ff.gen_NACAfoil(100,1,4,9)

#Get camber of the airfoil
n = 21
bool = 'cosine'
x, y = ff.get_camber(pos,n,bool)
x2, y2 = ff.get_camber(pos,n,'uniform')

#Get pansels from the discretisation
panels = []
for i in range(1,len(x)):
    panels.append(fc.Panel(x[i-1],y[i-1],x[i],y[i]))

#Get influence coefficients as A matrix LHS
A = ff.get_coeffMatrix(panels,n)

#Get RHS of the equation (free stream effect)
alfa = 2*(np.pi/180)
U = 10
RHS = ff.get_RHS(U,alfa,panels,n)

#Solve for gamma distribution
Gamma = np.linalg.solve(A,RHS)

dL = Gamma*U
Cl = sum(dL)[0]/(0.5*U*U)

dP = np.zeros((n-1,1))
Cp = np.zeros((n-1,1))
for i in range(n-1):
    dP[i] = U*Gamma[i]/panels[i].ds
    Cp[i] = dP[i]/(0.5*U*U)


xv = [panels[i].vpoint.x for i in range(n-1)]

plt.figure(1)
plt.plot(pos[:,0],pos[:,1],'-')
plt.plot(x,y,'-x')
plt.legend(['Airfoil','Camber'])
plt.xlabel('x [m]')
plt.ylabel('y [m]')


#plt.scatter(xv, Cp)
#plt.scatter(xv, Cp2
#plt.show()