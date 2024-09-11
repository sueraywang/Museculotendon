import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from testMuscle import *

# activation
act = 0
sample_size = 100

# Create a grid of points on the xy-plane
lMtilde = np.linspace(.45, 1.85, sample_size)
lTtilde = np.linspace(.99, 1.04, sample_size)

#Evaluate the function
X, Y = np.meshgrid(lMtilde, lTtilde)
vMTilde = []

for x, y in zip(X, Y):
    for x1, y1 in zip(x, y):
        vMTilde.append(calcVelTilde(x1, y1, act, params, curves))

Z = np.reshape(np.array(vMTilde), (-1, sample_size))

# print(Z)

# Create a contour plot
plt.contourf(X, Y, Z, levels=np.linspace(-1.5, 1.5, 100))
plt.colorbar(label='vMtilde')
plt.title('2D Slice of vMtilde')
plt.xlabel('lMtilde')
plt.ylabel('lTtilde')
plt.show()
