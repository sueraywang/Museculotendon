import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from testMuscle import *
import csv

# Generate data
act = 1
lMtilde = np.random.uniform(low = .45, high= 1.85, size = 100)
lTtilde = np.random.uniform(low = .99, high= 1.04, size = 100)
X, Y = np.meshgrid(lMtilde, lTtilde)
values = []

for x, y in zip(X, Y):
    for x1, y1 in zip(x, y):
        values.append(calcVelTilde(x1, y1, act, params, curves))

#reshare Z to matrix m*n, m = len(lMtilde),  l = len(lTtilde)
Z = np.reshape(np.array(values), (-1, len(lTtilde)))

data = np.vstack([X, Y, Z])
DF = pd.DataFrame(data) 
DF.to_csv("velocityData.csv")

# data = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
# DF = pd.read_csv('velocityData.csv')
# x = DF[["0", "1"]].to_numpy()
# y = DF[["2"]].to_numpy().T[0]
# print(x)
# print(y)
