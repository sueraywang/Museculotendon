import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from testMuscle import *
import csv
import random

sample_size = 1000

lMtilde = list(np.random.uniform(.40, 1.90, sample_size))
lTtilde = list(np.random.uniform(.99, 1.07, sample_size))
act = list(np.random.uniform(0, 1, sample_size))

r_lM = []
r_lT = []
r_act = []
vMtilde = []

# Evaluate the function on the grid
while (len(vMtilde) < 10000):
    lM = random.choice(lMtilde)
    lT = random.choice(lTtilde)
    a = random.choice(act)
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= 1.5) & (v >= -1.5)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

while (len(vMtilde) < 10500):
    lM = random.choice(lMtilde)
    lT = random.choice(lTtilde)
    a = 0
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= 1.5) & (v >= -1.5)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

while (len(vMtilde) < 11000):
    lM = random.choice(lMtilde)
    lT = random.choice(lTtilde)
    a = 1
    v = calcVelTilde(lM, lT, a, params, curves)
    if ((v <= 1.5) & (v >= -1.5)) : 
        r_lM.append(lM)
        r_lT.append(lT)
        r_act.append(a)
        vMtilde.append(v)

DF = pd.DataFrame(np.column_stack([r_lM, r_lT, r_act, vMtilde]), 
                               columns=['lMtilde', 'lTtilde', 'act', 'vMtilde'])
DF.to_csv("random3DVelocityData.csv")