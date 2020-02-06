import ta_functions as ff
import numpy as np
import matplotlib.pyplot as plt

#Generating NACA 4 digit airfoil
pos, yc = ff.gen_NACAfoil(100,2,4,8)

#Get camber of the airfoil
x, y = ff.get_camber(pos,30)



plt.scatter(pos[:,0],pos[:,1])
plt.plot(x,y)
plt.show()
