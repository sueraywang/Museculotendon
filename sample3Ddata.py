import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from testMuscle import *

# Create a grid of points on the xy-plane
lMtilde = np.random.uniform(low = .45, high= 1.85, size = 50)
lTtilde = np.random.uniform(low = .99, high= 1.04, size = 50)
act = np.random.uniform(low = 0, high= 1, size = 50)

X, Y, Z= np.meshgrid(lMtilde, lTtilde, act)

values = np.zeros((len(lMtilde),len(lTtilde),len(act)))

# Evaluate the function on the grid
for i in range (len(lMtilde)):
     for j in range (len(lTtilde)):
         for k in range (len(act)):
             values[i][j][k] = calcVelTilde(lMtilde[i], lTtilde[j], act[k], params, curves)



# Create a figure and 3D axes
fig = plt.figure()
ax = plt.subplot(projection="3d")
ax.scatter(X, Y, Z, s=10, alpha=.5, c=values, cmap="RdBu")
plt.show()

# Add labels
ax.set_xlabel('lMtilde')
ax.set_ylabel('lTtilde')
ax.set_zlabel('activation')

plt.show()