import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from testMuscle import *

# activation
act = 1

# Create a grid of points on the xy-plane
lMtilde = np.random.uniform(low = .45, high= 1.85, size = 1000)
lTtilde = np.random.uniform(low = .99, high= 1.04, size = 1000)

#Evaluate the function
vMtilde = list()
i = 0
for i in range(len(lMtilde)): 
    z = calcVelTilde(lMtilde[i], lTtilde[i], act, params, curves)
    vMtilde.append(z)

# Create a contour plot
plt.tricontourf(lMtilde, lTtilde, vMtilde)
plt.colorbar(label='vMtilde')
plt.title('2D Slice of vMtilde')
plt.xlabel('lMtilde')
plt.ylabel('lTtilde')
plt.show()
