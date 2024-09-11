import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from testMuscle import *

# Generate data
act = 0
sample_size = 1000

lMtilde = np.linspace(.45, 1.85, sample_size)
lTtilde = np.linspace(.99, 1.04, sample_size)

#Evaluate the function
X, Y = np.meshgrid(lMtilde, lTtilde)
vMTilde = []

for x, y in zip(X, Y):
    for x1, y1 in zip(x, y):
        vMTilde.append(calcVelTilde(x1, y1, act, params, curves))

#reshare Z to matrix m*n, m = len(lMtilde),  l = len(lTtilde)
Z = np.reshape(np.array(vMTilde), (-1, sample_size))

data = np.vstack([X, Y, Z])
DF = pd.DataFrame(data) 
DF.to_csv("velocityData_2D_a=0.csv")

# data = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
# DF = pd.read_csv('velocityData.csv')
# x = DF[["0", "1"]].to_numpy()
# y = DF[["2"]].to_numpy().T[0]
# print(x)
# print(y)
