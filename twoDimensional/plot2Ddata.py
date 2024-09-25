import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../python-nn')
from testMuscle import *

# activation
act = 1
sample_size = 100

""" # Read in data
DF = pd.read_csv('velocityData_2D_a=0.csv')
# Organize data for plot
X = DF.head(sample_size).drop(DF.columns[[0]], axis=1).to_numpy()
DF = DF.iloc[sample_size:]
Y = DF.head(sample_size).drop(DF.columns[[0]], axis=1).to_numpy()
DF = DF.iloc[sample_size:]
Z = DF.head(sample_size).drop(DF.columns[[0]], axis=1).to_numpy()
DF = DF.iloc[sample_size:] """


# Create a grid of points on the xy-plane
lMtilde = np.linspace(.30, 2.00, sample_size)
lTtilde = np.linspace(.985, 1.075, sample_size)

#Evaluate the function
X, Y = np.meshgrid(lMtilde, lTtilde)
vMTilde = []

for x, y in zip(X.ravel(), Y.ravel()):
        vMTilde.append(calcVelTilde(x, y, act, params, curves))
Z = np.reshape(np.array(vMTilde), X.shape)

# print(Z)

# Create a contour plot
plt.contourf(X, Y, Z, levels=np.linspace(-1.5, 1.5, 11))
plt.colorbar(label='vMtilde', ticks=np.linspace(-1.5, 1.5, 11)) 
plt.title('2D slice of vMtilde when activation = 1')
plt.xlabel('lMtilde')
plt.ylabel('lTtilde')
plt.show()
