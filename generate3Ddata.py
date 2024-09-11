import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from testMuscle import *
import csv

sample_size = 50

lMtilde = np.linspace(.45, 1.85, sample_size)
lTtilde = np.linspace(.99, 1.04, sample_size)
act = np.linspace(0, 1, sample_size)

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